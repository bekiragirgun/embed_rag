#!/usr/bin/env python3
"""
Probes Optimization Script
Tests different IVFFlat probes values to find optimal recall/speed tradeoff
"""

import sys
import json
from typing import List, Dict
from datetime import datetime
import numpy as np
from loguru import logger

from main_pipeline import RAGPipeline

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


def test_semantic_similarity_with_probes(pipeline: RAGPipeline, probes: int) -> Dict:
    """
    Test semantic similarity with a specific probes value

    Args:
        pipeline: RAG pipeline instance
        probes: IVFFlat probes value to test

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing with probes = {probes}")

    # Benzer anlamlı sorgu çiftleri
    query_pairs = [
        ("rough set theory", "rough sets approach"),
        ("minimize cost", "cost minimization"),
        ("transportation problem", "shipping optimization"),
        ("mathematical programming", "optimization mathematics"),
        ("fuzzy logic", "fuzzy reasoning"),
    ]

    overlap_scores = []

    for q1, q2 in query_pairs:
        # Create query embeddings
        q1_embedding = pipeline.embedding_model.embed_ensemble([q1], normalize=True)[0]
        q2_embedding = pipeline.embedding_model.embed_ensemble([q2], normalize=True)[0]

        # Her iki sorgu için top-3 sonuç al (probes parametresiyle)
        r1 = pipeline.vector_store.search_similar(
            query_embedding=q1_embedding.tolist(),
            limit=3,
            threshold=0.0,
            probes=probes
        )
        r2 = pipeline.vector_store.search_similar(
            query_embedding=q2_embedding.tolist(),
            limit=3,
            threshold=0.0,
            probes=probes
        )

        if not r1 or not r2:
            continue

        # Overlap: İki sorgunun sonuçlarında kaç ortak chunk var?
        ids_1 = set(r['id'] for r in r1)
        ids_2 = set(r['id'] for r in r2)
        overlap = len(ids_1 & ids_2)
        overlap_percentage = overlap / 3 * 100

        overlap_scores.append(overlap_percentage)

    avg_overlap = np.mean(overlap_scores) if overlap_scores else 0.0

    return {
        "probes": probes,
        "avg_overlap_percentage": float(avg_overlap),
        "individual_scores": [float(s) for s in overlap_scores],
        "pairs_tested": len(overlap_scores)
    }


def main():
    """Ana test fonksiyonu"""

    print("\n" + "="*70)
    print("PROBES OPTIMIZATION TEST")
    print("="*70 + "\n")

    # Pipeline'ı başlat
    logger.info("Loading RAG Pipeline...")
    pipeline = RAGPipeline(create_index=False)

    # Veritabanı durumunu kontrol et
    stats = pipeline.vector_store.get_stats()
    logger.info(f"Database ready: {stats['total_embeddings']} embeddings")
    print()

    # Test edilecek probes değerleri
    probes_values = [10, 15, 20, 30, 50]

    logger.info(f"Testing {len(probes_values)} different probes values...")
    logger.info("This will take approximately 5-10 minutes.\n")

    results = []

    # Her probes değerini test et
    for i, probes in enumerate(probes_values, 1):
        logger.info(f"[{i}/{len(probes_values)}] Testing probes = {probes}")

        result = test_semantic_similarity_with_probes(pipeline, probes)
        results.append(result)

        logger.info(f"  → Semantic Similarity: {result['avg_overlap_percentage']:.1f}%")
        print()

    # Sonuçları analiz et
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70 + "\n")

    # Tabloyu göster
    logger.info("Probes │ Semantic Similarity │ Change from Baseline")
    logger.info("──────┼────────────────────┼─────────────────────")

    baseline_score = results[0]['avg_overlap_percentage']  # probes=10 baseline
    best_probes = 10
    best_score = baseline_score

    for result in results:
        probes = result['probes']
        score = result['avg_overlap_percentage']
        change = score - baseline_score

        # En iyi skoru takip et
        if score > best_score:
            best_score = score
            best_probes = probes

        # Formatlanmış çıktı
        change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
        marker = " ★" if probes == best_probes else ""
        logger.info(f"  {probes:2d}  │      {score:5.1f}%       │ {change_str:>11}{marker}")

    print()

    # Öneriler
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\n")

    improvement = best_score - baseline_score

    if improvement > 5:
        logger.info(f"✓ Significant improvement found!")
        logger.info(f"  Best probes: {best_probes}")
        logger.info(f"  Score: {best_score:.1f}% (+{improvement:.1f}% from baseline)")
        logger.info(f"\n  ACTION: Update vector_store.py default probes to {best_probes}")
    elif improvement > 2:
        logger.info(f"✓ Moderate improvement found")
        logger.info(f"  Best probes: {best_probes}")
        logger.info(f"  Score: {best_score:.1f}% (+{improvement:.1f}% from baseline)")
        logger.info(f"\n  ACTION: Consider updating default probes to {best_probes}")
    else:
        logger.info(f"⚠️  Minimal improvement")
        logger.info(f"  Best probes: {best_probes}")
        logger.info(f"  Score: {best_score:.1f}% (+{improvement:.1f}% from baseline)")
        logger.info(f"\n  ACTION: Keep current probes=10 (negligible difference)")

    # Detaylı sonuçları kaydet
    output_file = "probes_optimization_results.json"
    output_data = {
        "test": "probes_optimization",
        "timestamp": datetime.now().isoformat(),
        "baseline_probes": 10,
        "baseline_score": baseline_score,
        "best_probes": best_probes,
        "best_score": best_score,
        "improvement": improvement,
        "results": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDetailed results saved: {output_file}")
    print()


if __name__ == "__main__":
    main()
