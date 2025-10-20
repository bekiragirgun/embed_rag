#!/usr/bin/env python3
import sys
from loguru import logger
from main_pipeline import RAGPipeline

logger.remove()
logger.add(sys.stdout, level="DEBUG", format="<level>{level: <8}</level> | <level>{message}</level>")

pipeline = RAGPipeline()

# Test with debug to see actual similarity values
query = "What is rough set theory?"
print(f"\nQuery: '{query}'")
print(f"Testing with threshold=0.0 (show all results):\n")

results = pipeline.search(query, limit=3, threshold=0.0)
print(f"\n{len(results)} results found\n")

for i, r in enumerate(results, 1):
    print(f"{i}. Similarity: {r['similarity']:.4f} ({r['similarity']*100:.1f}%)")
    print(f"   Type: {r['chunk_type']}")
    print(f"   Content: {r['content'][:100]}...\n")

print(f"\nNow testing with threshold=0.5:\n")
results = pipeline.search(query, limit=3, threshold=0.5)
print(f"\n{len(results)} results found\n")
