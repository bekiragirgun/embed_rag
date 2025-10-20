#!/usr/bin/env python3
"""
Hybrid Weights Optimization Script
Tests different BM25 vs Embedding weight combinations for optimal hybrid search
"""

import sys
import json
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from loguru import logger

from main_pipeline import RAGPipeline

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


def test_semantic_similarity_with_hybrid_weights(
    pipeline: RAGPipeline,
    bm25_weight: float,
    embedding_weight: float
) -> Dict:
    """
    Test semantic similarity with specific hybrid weights

    Args:
        pipeline: RAG pipeline instance
        bm25_weight: Weight for BM25 (keyword) component
        embedding_weight: Weight for embedding (semantic) component

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing weights: BM25={bm25_weight:.2f}, Embedding={embedding_weight:.2f}")

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
        # Use hybrid search with specific weights
        r1 = pipeline.search_hybrid(
            query=q1,
            limit=3,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            use_reranking=False,  # Disable reranking for consistent comparison
            threshold=0.0
        )
        r2 = pipeline.search_hybrid(
            query=q2,
            limit=3,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            use_reranking=False,
            threshold=0.0
        )

        if not r1 or not r2:
            continue

        # Overlap calculation
        ids_1 = set(r['id'] for r in r1)
        ids_2 = set(r['id'] for r in r2)
        overlap = len(ids_1 & ids_2)
        overlap_percentage = overlap / 3 * 100

        overlap_scores.append(overlap_percentage)

    avg_overlap = np.mean(overlap_scores) if overlap_scores else 0.0

    return {
        "bm25_weight": bm25_weight,
        "embedding_weight": embedding_weight,
        "avg_overlap_percentage": float(avg_overlap),
        "individual_scores": [float(s) for s in overlap_scores],
        "pairs_tested": len(overlap_scores)
    }


def generate_hybrid_weight_combinations() -> List[Tuple[float, float]]:
    """
    Generate weight combinations for BM25 vs Embedding

    Returns:
        List of (bm25_weight, embedding_weight) tuples that sum to 1.0
    """
    combinations = []

    # 1. Baseline (current default)
    combinations.append((0.3, 0.7))  # Current default

    # 2. Equal balance
    combinations.append((0.5, 0.5))

    # 3. Embedding-dominant variations
    combinations.extend([
        (0.2, 0.8),  # Strong embedding
        (0.1, 0.9),  # Very strong embedding
        (0.15, 0.85),
    ])

    # 4. BM25-dominant variations
    combinations.extend([
        (0.4, 0.6),  # Moderate BM25
        (0.6, 0.4),  # Strong BM25
        (0.7, 0.3),  # Very strong BM25
    ])

    # 5. Edge cases
    combinations.extend([
        (0.0, 1.0),  # Pure embedding (no BM25)
        (1.0, 0.0),  # Pure BM25 (no embedding)
    ])

    # 6. Fine-grained exploration around default
    combinations.extend([
        (0.25, 0.75),
        (0.35, 0.65),
    ])

    return combinations


def main():
    """Ana test fonksiyonu"""

    print("\n" + "="*70)
    print("HYBRID WEIGHTS OPTIMIZATION TEST")
    print("="*70 + "\n")

    # Pipeline'ı başlat
    logger.info("Loading RAG Pipeline...")
    pipeline = RAGPipeline(create_index=False)

    # Veritabanı durumunu kontrol et
    stats = pipeline.vector_store.get_stats()
    logger.info(f"Database ready: {stats['total_embeddings']} embeddings")
    print()

    # Weight kombinasyonları
    weight_combinations = generate_hybrid_weight_combinations()

    logger.info(f"Testing {len(weight_combinations)} weight combinations...")
    logger.info("This will take approximately 10-15 minutes.\\n")

    results = []

    # Her kombinasyonu test et
    for i, (bm25_w, emb_w) in enumerate(weight_combinations, 1):
        logger.info(f"[{i}/{len(weight_combinations)}] Testing combination {i}")

        result = test_semantic_similarity_with_hybrid_weights(
            pipeline,
            bm25_weight=bm25_w,
            embedding_weight=emb_w
        )
        results.append(result)

        logger.info(f"  → Semantic Similarity: {result['avg_overlap_percentage']:.1f}%")
        print()

    # Sonuçları analiz et
    print("="*70)
    print("RESULTS SUMMARY (Top 10)")
    print("="*70 + "\\n")

    # Sırala (en yüksek'ten en düşük'e)
    sorted_results = sorted(results, key=lambda x: x['avg_overlap_percentage'], reverse=True)

    # Tablo başlığı
    logger.info(" Rank │  BM25  │ Embedding │ Similarity │ Improvement")
    logger.info("──────┼────────┼───────────┼────────────┼─────────────")

    baseline_score = None
    for r in results:
        if abs(r['bm25_weight'] - 0.3) < 0.01:  # Current default
            baseline_score = r['avg_overlap_percentage']
            break

    if baseline_score is None:
        baseline_score = results[0]['avg_overlap_percentage']

    best_score = sorted_results[0]['avg_overlap_percentage']
    best_weights = (sorted_results[0]['bm25_weight'], sorted_results[0]['embedding_weight'])

    # Top 10 göster
    for i, result in enumerate(sorted_results[:10], 1):
        bm25_w = result['bm25_weight']
        emb_w = result['embedding_weight']
        score = result['avg_overlap_percentage']
        improvement = score - baseline_score

        marker = " ★" if i == 1 else ""
        improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"

        logger.info(
            f"  {i:2d}  │ {bm25_w:.2f}  │   {emb_w:.2f}    │   {score:5.1f}%   │ {improvement_str:>6}{marker}"
        )

    print()

    # Detaylı analiz
    print("="*70)
    print("DETAILED ANALYSIS")
    print("="*70 + "\\n")

    improvement = best_score - baseline_score

    logger.info(f"Baseline (BM25=0.3, Emb=0.7):  {baseline_score:.1f}%")
    logger.info(f"Best combination:              {best_score:.1f}%")
    logger.info(f"Improvement:                   +{improvement:.1f}%")
    print()

    logger.info("Best weights:")
    logger.info(f"  BM25 (keyword):        {best_weights[0]:.2f}")
    logger.info(f"  Embedding (semantic):  {best_weights[1]:.2f}")
    print()

    # Öneriler
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\\n")

    if improvement >= 5:
        logger.info(f"✓ Significant improvement found!")
        logger.info(f"  Improvement: +{improvement:.1f}%")
        logger.info(f"\\n  ACTION: Update main_pipeline.py search_hybrid() defaults to:")
        logger.info(f"    bm25_weight={best_weights[0]:.2f}")
        logger.info(f"    embedding_weight={best_weights[1]:.2f}")
    elif improvement >= 2:
        logger.info(f"✓ Moderate improvement found")
        logger.info(f"  Improvement: +{improvement:.1f}%")
        logger.info(f"\\n  ACTION: Consider updating weights (marginal gain)")
    else:
        logger.info(f"⚠️  Minimal improvement")
        logger.info(f"  Improvement: +{improvement:.1f}%")
        logger.info(f"\\n  ACTION: Keep current weights (0.3/0.7 is optimal)")

    # Detaylı sonuçları kaydet
    output_file = "hybrid_weights_results.json"
    output_data = {
        "test": "hybrid_weights_optimization",
        "timestamp": datetime.now().isoformat(),
        "baseline_weights": {"bm25": 0.3, "embedding": 0.7},
        "baseline_score": baseline_score,
        "best_score": best_score,
        "best_weights": {
            "bm25": best_weights[0],
            "embedding": best_weights[1]
        },
        "improvement": improvement,
        "all_results": sorted_results,
        "total_combinations_tested": len(results)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\\nDetailed results saved: {output_file}")
    print()


if __name__ == "__main__":
    main()
