#!/usr/bin/env python3
"""
Test Multi-Query Search
Quick test to verify multi-query search is working
"""

import sys
from loguru import logger

from main_pipeline import RAGPipeline

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


def main():
    """Test multi-query search"""

    print("\n" + "="*70)
    print("MULTI-QUERY SEARCH TEST")
    print("="*70 + "\n")

    # Initialize pipeline
    logger.info("Loading RAG Pipeline...")
    pipeline = RAGPipeline(create_index=False)

    # Check database
    stats = pipeline.vector_store.get_stats()
    logger.info(f"Database ready: {stats['total_embeddings']} embeddings")
    print()

    # Test queries
    test_queries = [
        "rough set theory",
        "minimize transportation cost",
        "fuzzy logic programming"
    ]

    for query in test_queries:
        print("=" * 70)
        print(f"Query: {query}")
        print("=" * 70 + "\n")

        # Multi-query search
        results = pipeline.search_multi_query(
            query=query,
            limit=3,
            max_variants=5,
            use_reranking=False  # Disable for faster testing
        )

        # Print results
        print(f"\nResults ({len(results)} documents):")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. RRF Score: {result.get('rrf_score', 0):.4f} | "
                  f"Matched {result.get('num_variants_matched', 0)} variants")
            print(f"   Type: {result.get('chunk_type', 'unknown')}")
            print(f"   Content: {result.get('content', '')[:100]}...")

        print("\n")

    print("=" * 70)
    print("✓ Multi-query search test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
