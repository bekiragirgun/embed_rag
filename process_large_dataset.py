#!/usr/bin/env python3
"""
Large Dataset Processing Script
Process 40GB+ PDF collections with optimized indexing strategy
+ Resume capability: Skip already processed PDFs
+ Thermal management: 30min work / 10min cooldown cycles
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from main_pipeline import RAGPipeline

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "large_dataset_processing.log",
    format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="DEBUG"
)


def process_pdf_directory(
    pdf_dir: str,
    create_index_after: bool = True,
    index_lists: int = 2000,
    recursive: bool = True,
    work_minutes: int = 30,
    cooldown_minutes: int = 10
):
    """
    Process all PDFs in a directory with optimized strategy for large datasets

    Args:
        pdf_dir: Path to directory containing PDF files
        create_index_after: Whether to create index after loading (default: True)
        index_lists: Number of lists for IVFFlat index (default: 2000 for ~5M vectors)
        recursive: Whether to search subdirectories (default: True)
        work_minutes: Minutes to work before cooldown (default: 30)
        cooldown_minutes: Minutes to cooldown (default: 10)

    Strategy:
        1. Disable index during data loading (fast inserts)
        2. Skip already processed PDFs (resume capability)
        3. Work/cooldown cycles for thermal management
        4. Create index ONCE at the end (30-120 min one-time operation)
    """

    print("\n" + "="*70)
    print("LARGE DATASET PROCESSING - 40GB+ PDF Collection")
    print("="*70 + "\n")

    # Validate PDF directory
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return

    # Find PDF files (recursive or not)
    pdf_files = []

    if recursive:
        logger.info("Searching for PDFs recursively (including subdirectories)...")
        for root, dirs, files in os.walk(pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        logger.info("Searching for PDFs in top-level directory only...")
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith('.pdf')
        ]

    if not pdf_files:
        logger.error(f"No PDF files found in: {pdf_dir}")
        return

    # Show directory structure
    if recursive:
        subdirs = set(os.path.dirname(pdf) for pdf in pdf_files)
        logger.info(f"Found {len(pdf_files)} PDF files in {len(subdirs)} directories")
        logger.info(f"Root directory: {pdf_dir}\n")
    else:
        logger.info(f"Found {len(pdf_files)} PDF files")
        logger.info(f"Directory: {pdf_dir}\n")

    # Initialize pipeline WITHOUT index (critical for 40GB datasets)
    logger.info("Initializing pipeline (index creation: DISABLED)")
    logger.info("  → Fast inserts without index overhead")
    logger.info("  → Index will be created AFTER all data is loaded\n")

    pipeline = RAGPipeline(create_index=False)

    # Get already processed PDFs from database (RESUME CAPABILITY)
    logger.info("Checking for already processed PDFs...")
    from vector_store import PaperEmbedding
    import sqlalchemy as sa

    session = pipeline.vector_store.Session()
    try:
        # Get unique paper IDs (using PDF filename as paper_id)
        processed_ids = session.query(PaperEmbedding.paper_id).distinct().all()
        processed_set = set([pid[0] for pid in processed_ids])

        logger.info(f"✓ Found {len(processed_set)} already processed PDFs")
        logger.info(f"  Will skip these and process remaining {len(pdf_files) - len(processed_set)} PDFs\n")
    finally:
        session.close()

    # Filter out already processed PDFs
    pdf_files_filtered = []
    for pdf_path in pdf_files:
        # Use filename as paper_id (same as in main_pipeline.py)
        paper_id = Path(pdf_path).stem
        if paper_id not in processed_set:
            pdf_files_filtered.append(pdf_path)

    if not pdf_files_filtered:
        logger.info("✓ All PDFs already processed! Nothing to do.")
        return

    logger.info(f"📋 Processing queue: {len(pdf_files_filtered)} PDFs")
    print()

    # Process all PDFs with thermal management
    start_time = datetime.now()
    cycle_start_time = datetime.now()
    successful = 0
    failed = 0
    skipped = len(processed_set)

    for i, pdf_path in enumerate(pdf_files_filtered, 1):
        # Thermal Management: Check if work cycle is complete
        cycle_elapsed = (datetime.now() - cycle_start_time).total_seconds()
        if cycle_elapsed >= work_minutes * 60:
            logger.info(f"\n{'='*70}")
            logger.info(f"🌡️  THERMAL COOLDOWN")
            logger.info(f"{'='*70}")
            logger.info(f"Worked for {cycle_elapsed/60:.1f} minutes")
            logger.info(f"Cooling down for {cooldown_minutes} minutes...")
            logger.info(f"Processed so far: {skipped + successful} / {len(pdf_files)} PDFs")
            logger.info(f"Resume at: {datetime.now() + timedelta(minutes=cooldown_minutes)}")
            logger.info(f"{'='*70}\n")

            # Countdown cooldown
            for remaining in range(cooldown_minutes, 0, -1):
                print(f"\r⏳ Cooldown: {remaining} minutes remaining...", end='', flush=True)
                time.sleep(60)

            print("\r✓ Cooldown complete! Resuming...                    \n")
            cycle_start_time = datetime.now()  # Reset cycle timer

        # Get relative path for display
        if recursive:
            rel_path = os.path.relpath(pdf_path, pdf_dir)
        else:
            rel_path = os.path.basename(pdf_path)

        total_processed = skipped + successful + failed
        logger.info(f"[{total_processed + 1}/{len(pdf_files)}] Processing: {rel_path}")

        try:
            result = pipeline.process_pdf(pdf_path)

            if result.get("status") == "success":
                successful += 1
                extraction = result.get("extraction", {})
                logger.info(f"  ✓ Success: {extraction.get('total_chunks', 0)} chunks")
            else:
                failed += 1
                logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            failed += 1
            logger.error(f"  ✗ Exception: {str(e)}")

        # Progress update every 10 files
        if i % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            total_processed = skipped + successful + failed
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(pdf_files_filtered) - i) / rate if rate > 0 else 0
            logger.info(f"\nProgress: {total_processed}/{len(pdf_files)} ({total_processed/len(pdf_files)*100:.1f}%)")
            logger.info(f"  New PDFs processed: {successful + failed}")
            logger.info(f"  Rate: {rate:.2f} files/sec")
            logger.info(f"  Estimated time remaining: {remaining/3600:.1f} hours\n")

    elapsed_time = (datetime.now() - start_time).total_seconds()

    # Summary
    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)
    print(f"Already processed (skipped): {skipped}")
    print(f"Newly processed (success):   {successful}")
    print(f"Failed:                      {failed}")
    print(f"Total in database:           {skipped + successful}")
    print(f"Time (this session): {elapsed_time/3600:.2f} hours")
    if successful + failed > 0:
        print(f"Rate: {(successful + failed)/elapsed_time:.2f} files/sec")

    # Get final stats
    stats = pipeline.vector_store.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total embeddings: {stats['total_embeddings']:,}")
    print(f"  Unique papers: {stats['unique_papers']:,}")
    print(f"  By type: {stats['by_type']}")
    print("="*70 + "\n")

    # Create index if requested
    if create_index_after and stats['total_embeddings'] >= 1000:
        print("Creating IVFFlat index...")
        print(f"  Vectors: {stats['total_embeddings']:,}")
        print(f"  Lists: {index_lists}")
        print("  This may take 30-120 minutes for millions of vectors...\n")

        try:
            index_start = datetime.now()
            pipeline.create_index(lists=index_lists)
            index_time = (datetime.now() - index_start).total_seconds()

            print(f"✓ Index created successfully in {index_time/60:.1f} minutes")
            print("  Search will now be ~16x faster!\n")

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            logger.info("You can create it later with: pipeline.create_index(lists=2000)")

    elif stats['total_embeddings'] < 1000:
        logger.info("Dataset too small (<1000 vectors) - index not recommended")
        logger.info("Sequential scan will be faster for this size")

    else:
        logger.info("Index creation skipped (create_index_after=False)")
        logger.info("Create manually later with: pipeline.create_index(lists=2000)")

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70 + "\n")


def test_search(pipeline: RAGPipeline):
    """Test search functionality"""

    print("\n" + "="*70)
    print("TESTING SEMANTIC SEARCH")
    print("="*70 + "\n")

    queries = [
        "rough set theory",
        "transportation optimization",
        "minimize cost",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        results = pipeline.search(query, limit=3)

        if results:
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['chunk_type']}] Similarity: {result['similarity']:.2%}")
                print(f"     {result['content'][:100]}...")
        else:
            print("  No results found")
        print()


if __name__ == "__main__":

    print("\n" + "="*70)
    print("RAG PIPELINE - PDF Collection Processing")
    print("="*70 + "\n")

    # ============================================================
    # ASK USER FOR PDF DIRECTORY
    # ============================================================

    print("PDF dosyalarınız nerede?\n")
    print("Örnekler:")
    print("  - /Users/bekiragirgun/Downloads/Documents/aktarılan")
    print("  - /path/to/your/research/papers")
    print("  - /Volumes/ExternalDrive/PDFs\n")

    pdf_directory = input("PDF klasörü yolu: ").strip()

    # Remove quotes if user copy-pasted with quotes
    pdf_directory = pdf_directory.strip('"').strip("'")

    # Validate directory
    if not pdf_directory:
        print("\n❌ Hata: Klasör yolu boş olamaz")
        sys.exit(1)

    if not os.path.exists(pdf_directory):
        print(f"\n❌ Hata: Klasör bulunamadı: {pdf_directory}")
        sys.exit(1)

    if not os.path.isdir(pdf_directory):
        print(f"\n❌ Hata: Bu bir klasör değil: {pdf_directory}")
        sys.exit(1)

    # ============================================================
    # ASK ABOUT RECURSIVE SCANNING
    # ============================================================

    print("\n" + "-"*70)
    print("Klasör Tarama Ayarları")
    print("-"*70 + "\n")

    print("Alt klasörlerdeki PDF'ler de işlensin mi?")
    print("  evet: Tüm alt klasörleri tara (önerilen - derin klasör yapıları için)")
    print("  hayır: Sadece ana klasördeki PDF'leri işle\n")

    recursive_input = input("[evet/hayır] veya Enter (varsayılan: evet): ").strip().lower()

    if recursive_input in ['h', 'hayır', 'no', 'n']:
        recursive = False
    else:
        recursive = True  # Default to recursive

    # Count PDFs (recursive or not)
    pdf_files = []

    if recursive:
        print("\n🔍 Alt klasörler taranıyor...")
        for root, dirs, files in os.walk(pdf_directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        pdf_files = [
            os.path.join(pdf_directory, f)
            for f in os.listdir(pdf_directory)
            if f.lower().endswith('.pdf')
        ]

    if not pdf_files:
        if recursive:
            print(f"\n❌ Hata: Bu klasörde ve alt klasörlerinde PDF dosyası yok: {pdf_directory}")
        else:
            print(f"\n❌ Hata: Bu klasörde PDF dosyası yok: {pdf_directory}")
        sys.exit(1)

    # Show summary
    if recursive:
        subdirs = set(os.path.dirname(pdf) for pdf in pdf_files)
        print(f"\n✓ {len(pdf_files)} PDF dosyası bulundu")
        print(f"  ({len(subdirs)} farklı klasörde)")
    else:
        print(f"\n✓ {len(pdf_files)} PDF dosyası bulundu")

    # ============================================================
    # ASK ABOUT INDEX CREATION
    # ============================================================

    print("\n" + "-"*70)
    print("Index Ayarları")
    print("-"*70 + "\n")

    print(f"Dataset boyutu: {len(pdf_files)} PDF")

    if len(pdf_files) < 100:
        print("  → Küçük dataset: Index önerilmez (sequential scan daha hızlı)")
        default_index = False
        recommended = "hayır"
    elif len(pdf_files) < 1000:
        print("  → Orta dataset: Index opsiyonel")
        default_index = False
        recommended = "hayır"
    else:
        print("  → Büyük dataset: Index önerilir (aramaları ~16x hızlandırır)")
        default_index = True
        recommended = "evet"

    print(f"\nTüm veriler yüklendikten SONRA index oluşturulsun mu? (önerilen: {recommended})")
    create_index_input = input("[evet/hayır] veya Enter (varsayılan): ").strip().lower()

    if create_index_input in ['e', 'evet', 'yes', 'y']:
        create_index_after = True
    elif create_index_input in ['h', 'hayır', 'no', 'n']:
        create_index_after = False
    else:
        create_index_after = default_index

    # Index lists configuration
    if create_index_after:
        print("\nIndex lists sayısı (varsayılan: 2000 - 5M vektör için)")
        print("  100K-1M vektör:   500-1000 lists")
        print("  1M-10M vektör:    1000-3000 lists")
        print("  10M+ vektör:      3000-5000 lists")

        lists_input = input("\nLists sayısı (Enter = varsayılan 2000): ").strip()

        if lists_input:
            try:
                index_lists = int(lists_input)
            except ValueError:
                print("⚠️  Geçersiz sayı, varsayılan 2000 kullanılacak")
                index_lists = 2000
        else:
            index_lists = 2000
    else:
        index_lists = 2000

    # ============================================================
    # THERMAL MANAGEMENT SETTINGS
    # ============================================================

    print("\n" + "-"*70)
    print("Termal Yönetim (GPU Soğutma)")
    print("-"*70 + "\n")

    print("GPU'nun aşırı ısınmaması için çalışma/dinlenme döngüsü:")
    print("  Varsayılan: 30 dakika çalış, 10 dakika dinlen")
    print("  Önerilen: M4 Max için güvenli ve etkili\n")

    work_input = input("Çalışma süresi (dakika) [Enter = 30]: ").strip()
    cooldown_input = input("Dinlenme süresi (dakika) [Enter = 10]: ").strip()

    work_minutes = int(work_input) if work_input else 30
    cooldown_minutes = int(cooldown_input) if cooldown_input else 10

    print(f"\n✓ Termal döngü: {work_minutes}dk çalış / {cooldown_minutes}dk dinlen")

    # ============================================================
    # CONFIRMATION
    # ============================================================

    print("\n" + "="*70)
    print("ÖZET")
    print("="*70)
    print(f"Klasör:          {pdf_directory}")
    print(f"PDF sayısı:      {len(pdf_files)}")
    print(f"Alt klasörler:   {'Evet (recursive)' if recursive else 'Hayır (sadece ana klasör)'}")
    print(f"Termal döngü:    {work_minutes}dk çalış / {cooldown_minutes}dk dinlen")
    print(f"Resume:          Evet (işlenenleri skip eder)")
    print(f"Index:           {'Evet' if create_index_after else 'Hayır'}")
    if create_index_after:
        print(f"Index lists:     {index_lists}")
    print("="*70 + "\n")

    confirm = input("Devam edilsin mi? [evet/hayır]: ").strip().lower()

    if confirm not in ['e', 'evet', 'yes', 'y']:
        print("\nİşlem iptal edildi.")
        sys.exit(0)

    # ============================================================
    # PROCESS PDFs
    # ============================================================

    process_pdf_directory(
        pdf_dir=pdf_directory,
        create_index_after=create_index_after,
        index_lists=index_lists,
        recursive=recursive,
        work_minutes=work_minutes,
        cooldown_minutes=cooldown_minutes
    )
