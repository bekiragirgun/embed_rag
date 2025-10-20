"""
Hızlı PDF işleme - 5-10 makale için
"""
import os
from main_pipeline import RAGPipeline

# PDF klasörü
pdf_dir = "/Users/bekiragirgun/Downloads/Documents/aktarılan"

# İlk 10 PDF'i bul
pdfs = []
for file in os.listdir(pdf_dir):
    if file.endswith('.pdf'):
        pdfs.append(os.path.join(pdf_dir, file))
    if len(pdfs) >= 10:
        break

print(f"Found {len(pdfs)} PDFs to process")

# Pipeline başlat (MPS ile!)
pipeline = RAGPipeline(create_index=False)

# İşle
for i, pdf_path in enumerate(pdfs, 1):
    print(f"\n[{i}/{len(pdfs)}] Processing: {os.path.basename(pdf_path)}")
    result = pipeline.process_pdf(pdf_path)
    if result.get('status') == 'success':
        print(f"✓ {result['extraction']['total_chunks']} chunks")

print("\n✓ Processing complete!")
print(f"Total papers: {pipeline.vector_store.get_stats()['unique_papers']}")
