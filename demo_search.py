#!/usr/bin/env python3
"""
Final search demonstration with threshold 0.5
Shows the RAG system in action
"""

import sys
from loguru import logger
from main_pipeline import RAGPipeline

# Clean logging for demo
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{message}</level>")

def main():
    print("\n" + "="*80)
    print("  RAG SYSTEM - SCIENTIFIC PAPER SEMANTIC SEARCH DEMO")
    print("="*80 + "\n")

    # Initialize
    print("🔧 Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Stats
    stats = pipeline.vector_store.get_stats()
    print(f"📊 Database: {stats['total_embeddings']} embeddings ({stats['by_type']})")
    print(f"📄 Papers: {stats['unique_papers']}\n")

    # Demo queries
    queries = [
        ("What is rough set theory?", None),
        ("Minimize transportation cost", None),
        ("Transportation constraints", None),
        ("Optimization formulation", None),
        ("Tables about costs", "table"),
    ]

    for query, chunk_type in queries:
        print("─" * 80)
        print(f"🔍 Query: '{query}'")
        if chunk_type:
            print(f"   Filter: {chunk_type} chunks only")
        print("─" * 80 + "\n")

        results = pipeline.search(query, limit=3, chunk_type=chunk_type)

        if results:
            for i, result in enumerate(results, 1):
                similarity_pct = result['similarity'] * 100
                color = "🟢" if similarity_pct >= 75 else "🟡" if similarity_pct >= 65 else "🔴"

                print(f"{color} Result {i}: [{result['chunk_type'].upper()}] {similarity_pct:.1f}% match")

                # Clean content
                content = result['content'][:150]
                content = content.replace('\n', ' ').strip()
                print(f"   {content}...")

                # Show metadata
                meta = result.get('metadata', {})
                if 'page' in meta:
                    print(f"   📄 Page {meta['page']}")
                print()
        else:
            print("⚠️  No results above similarity threshold (50%)\n")

    print("=" * 80)
    print("\n✅ RAG System Demo Complete!")
    print("\n💡 Tips:")
    print("   • Similarity scores 70-85% indicate strong matches")
    print("   • Adjust threshold parameter to control result quality")
    print("   • Filter by chunk_type: 'text', 'table', 'formula', 'theorem'")
    print("   • Ensemble embeddings (SPECTER2 + E5-Large + SciBERT) provide robust semantic matching\n")

if __name__ == "__main__":
    main()
