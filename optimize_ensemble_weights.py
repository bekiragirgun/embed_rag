#!/usr/bin/env python3
"""
Ensemble Weights Optimization
Tests different weight combinations for the 3 embedding models
"""

import sys
import json
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from loguru import logger
from itertools import product

from main_pipeline import RAGPipeline

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


def test_semantic_similarity_with_weights(
    pipeline: RAGPipeline,
    weights: Tuple[float, float, float]
) -> Dict:
    """
    Test semantic similarity with specific ensemble weights

    Args:
        pipeline: RAG pipeline instance
        weights: Tuple of (specter2_weight, e5_weight, scibert_weight)

    Returns:
        Dictionary with test results
    """
    specter2_w, e5_w, scibert_w = weights
    logger.info(f"Testing weights: SPECTER2={specter2_w:.2f}, E5={e5_w:.2f}, SciBERT={scibert_w:.2f}")

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
        # Create query embeddings with specific weights
        q1_embedding = pipeline.embedding_model.embed_ensemble([q1], weights=weights, normalize=True)[0]
        q2_embedding = pipeline.embedding_model.embed_ensemble([q2], weights=weights, normalize=True)[0]

        # Search
        r1 = pipeline.vector_store.search_similar(
            query_embedding=q1_embedding.tolist(),
            limit=3,
            threshold=0.0,
            probes=10
        )
        r2 = pipeline.vector_store.search_similar(
            query_embedding=q2_embedding.tolist(),
            limit=3,
            threshold=0.0,
            probes=10
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
        "weights": {
            "specter2": specter2_w,
            "e5": e5_w,
            "scibert": scibert_w
        },
        "avg_overlap_percentage": float(avg_overlap),
        "individual_scores": [float(s) for s in overlap_scores],
        "pairs_tested": len(overlap_scores)
    }


def generate_weight_combinations() -> List[Tuple[float, float, float]]:
    """
    Generate smart grid of weight combinations

    Returns:
        List of (specter2_w, e5_w, scibert_w) tuples that sum to 1.0
    """
    combinations = []

    # 1. Baseline (equal weights)
    combinations.append((1/3, 1/3, 1/3))

    # 2. Single model dominance (0.6, 0.2, 0.2)
    combinations.extend([
        (0.6, 0.2, 0.2),  # SPECTER2 dominant (scientific focus)
        (0.2, 0.6, 0.2),  # E5 dominant (general purpose)
        (0.2, 0.2, 0.6),  # SciBERT dominant (scientific BERT)
    ])

    # 3. Two-model combinations (0.5, 0.5, 0.0)
    combinations.extend([
        (0.5, 0.5, 0.0),  # SPECTER2 + E5
        (0.5, 0.0, 0.5),  # SPECTER2 + SciBERT
        (0.0, 0.5, 0.5),  # E5 + SciBERT
    ])

    # 4. Stronger dominance (0.7, 0.15, 0.15)
    combinations.extend([
        (0.7, 0.15, 0.15),  # Strong SPECTER2
        (0.15, 0.7, 0.15),  # Strong E5
        (0.15, 0.15, 0.7),  # Strong SciBERT
    ])

    # 5. Scientific papers bias (more SPECTER2 + SciBERT, less E5)
    combinations.extend([
        (0.5, 0.2, 0.3),  # Balanced scientific
        (0.4, 0.2, 0.4),  # Even scientific
        (0.6, 0.1, 0.3),  # SPECTER2 heavy
    ])

    # 6. Exploration: Different ratios
    combinations.extend([
        (0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3),
        (0.3, 0.3, 0.4),
    ])

    return combinations


def main():
    """Ana test fonksiyonu"""

    print("\n" + "="*70)
    print("ENSEMBLE WEIGHTS GRID SEARCH")
    print("="*70 + "\n")

    # Pipeline'ı başlat
    logger.info("Loading RAG Pipeline...")
    pipeline = RAGPipeline(create_index=False)

    # Veritabanı durumu
    stats = pipeline.vector_store.get_stats()
    logger.info(f"Database ready: {stats['total_embeddings']} embeddings")
    print()

    # Weight kombinasyonları
    weight_combinations = generate_weight_combinations()

    logger.info(f"Testing {len(weight_combinations)} weight combinations...")
    logger.info("This will take approximately 30-45 minutes.\n")

    results = []

    # Her kombinasyonu test et
    for i, weights in enumerate(weight_combinations, 1):
        logger.info(f"[{i}/{len(weight_combinations)}] Testing combination {i}")

        result = test_semantic_similarity_with_weights(pipeline, weights)
        results.append(result)

        logger.info(f"  → Semantic Similarity: {result['avg_overlap_percentage']:.1f}%")
        print()

    # Sonuçları analiz et
    print("="*70)
    print("RESULTS SUMMARY (Top 10)")
    print("="*70 + "\n")

    # Sırala (en yüksek'ten en düşük'e)
    sorted_results = sorted(results, key=lambda x: x['avg_overlap_percentage'], reverse=True)

    # Tablo başlığı
    logger.info(" Rank │ SPEC2 │  E5   │ SciBERT │ Similarity │ Improvement")
    logger.info("──────┼───────┼───────┼─────────┼────────────┼─────────────")

    baseline_score = results[0]['avg_overlap_percentage']  # Equal weights baseline
    best_score = sorted_results[0]['avg_overlap_percentage']
    best_weights = sorted_results[0]['weights']

    # Top 10 göster
    for i, result in enumerate(sorted_results[:10], 1):
        w = result['weights']
        score = result['avg_overlap_percentage']
        improvement = score - baseline_score

        marker = " ★" if i == 1 else ""
        improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"

        logger.info(
            f"  {i:2d}  │ {w['specter2']:.2f}  │ {w['e5']:.2f}  │  {w['scibert']:.2f}   │   {score:5.1f}%   │ {improvement_str:>6}{marker}"
        )

    print()

    # Detaylı analiz
    print("="*70)
    print("DETAILED ANALYSIS")
    print("="*70 + "\n")

    improvement = best_score - baseline_score

    logger.info(f"Baseline (equal weights):   {baseline_score:.1f}%")
    logger.info(f"Best combination:           {best_score:.1f}%")
    logger.info(f"Improvement:                +{improvement:.1f}%")
    print()

    logger.info("Best weights:")
    logger.info(f"  SPECTER2:  {best_weights['specter2']:.2f}")
    logger.info(f"  E5-Large:  {best_weights['e5']:.2f}")
    logger.info(f"  SciBERT:   {best_weights['scibert']:.2f}")
    print()

    # Öneriler
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\n")

    if improvement >= 5:
        logger.info(f"✓ Significant improvement found!")
        logger.info(f"  Improvement: +{improvement:.1f}%")
        logger.info(f"\n  ACTION: Update ensemble_embeddings.py weights to:")
        logger.info(f"    'specter2': {best_weights['specter2']:.2f}")
        logger.info(f"    'e5': {best_weights['e5']:.2f}")
        logger.info(f"    'scibert': {best_weights['scibert']:.2f}")
    elif improvement >= 2:
        logger.info(f"✓ Moderate improvement found")
        logger.info(f"  Improvement: +{improvement:.1f}%")
        logger.info(f"\n  ACTION: Consider updating weights (marginal gain)")
    else:
        logger.info(f"⚠️  Minimal improvement")
        logger.info(f"  Improvement: +{improvement:.1f}%")
        logger.info(f"\n  ACTION: Keep equal weights (no significant benefit)")

    # Detaylı sonuçları kaydet
    output_file = "ensemble_weights_results.json"
    output_data = {
        "test": "ensemble_weights_optimization",
        "timestamp": datetime.now().isoformat(),
        "baseline_score": baseline_score,
        "best_score": best_score,
        "best_weights": best_weights,
        "improvement": improvement,
        "all_results": sorted_results,
        "total_combinations_tested": len(results)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDetailed results saved: {output_file}")
    print()


if __name__ == "__main__":
    main()
