#!/usr/bin/env python3
"""
Quick search test to debug similarity scores
"""

import sys
from loguru import logger
from main_pipeline import RAGPipeline

# Configure verbose logging
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="<level>{level: <8}</level> | <level>{message}</level>")

def main():
    print("\n" + "="*70)
    print("SEARCH DEBUG TEST")
    print("="*70 + "\n")

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Check database stats first
    stats = pipeline.vector_store.get_stats()
    print(f"\nDatabase Stats:")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Unique papers: {stats['unique_papers']}")
    print(f"  By type: {stats['by_type']}\n")

    if stats['total_embeddings'] == 0:
        print("❌ No embeddings in database! Run main_pipeline.py first.\n")
        return

    # Test queries
    test_queries = [
        ("What is rough set theory?", None),
        ("Minimize transportation cost", None),
        ("Transportation constraints", None),
        ("table data", "table"),  # Filter for tables only
    ]

    for query, chunk_type in test_queries:
        print("\n" + "-"*70)
        print(f"Query: '{query}'")
        if chunk_type:
            print(f"Filter: chunk_type='{chunk_type}'")
        print("-"*70 + "\n")

        # Search with threshold=0.0 to get all results
        results = pipeline.search(query, limit=5, chunk_type=chunk_type, threshold=0.0)

        if results:
            print(f"\n✅ Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['chunk_type'].upper()}] Similarity: {result['similarity']:.4f}")
                print(f"   Content: {result['content'][:150]}...")
                print(f"   Paper: {result['paper_id']}")
                print()
        else:
            print("❌ No results found!\n")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()
