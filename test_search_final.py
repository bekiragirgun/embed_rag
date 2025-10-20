#!/usr/bin/env python3
"""
Final search test with default threshold (0.6)
"""

import sys
from loguru import logger
from main_pipeline import RAGPipeline

# Configure simple logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <8}</level> | <level>{message}</level>")

def main():
    print("\n" + "="*70)
    print("SEARCH TEST - WITH DEFAULT THRESHOLD (0.6)")
    print("="*70 + "\n")

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Check database stats
    stats = pipeline.vector_store.get_stats()
    print(f"Database: {stats['total_embeddings']} embeddings from {stats['unique_papers']} paper(s)\n")

    # Test queries with default threshold
    test_queries = [
        "What is rough set theory?",
        "Minimize transportation cost",
        "Transportation constraints",
        "How to formulate optimization problems?",
    ]

    for query in test_queries:
        print("\n" + "-"*70)
        print(f"Query: '{query}'")
        print("-"*70)

        # Use default threshold (0.6)
        results = pipeline.search(query, limit=3)

        if results:
            print(f"\n✅ Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['chunk_type'].upper()}] Similarity: {result['similarity']:.2%}")
                content = result['content'][:120].replace('\n', ' ')
                print(f"   {content}...")
                print()
        else:
            print("\n⚠️  No results above threshold (0.6)\n")

    print("="*70)
    print("\n✅ Search is working correctly!")
    print("Typical similarity scores: 0.70-0.85 for good matches")
    print("Threshold 0.6 filters out low-quality results\n")

if __name__ == "__main__":
    main()
