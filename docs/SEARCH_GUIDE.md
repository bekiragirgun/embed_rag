# Search Guide - Semantic Similarity Tuning

## Quick Reference

```python
from main_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Basic search (default threshold=0.5)
results = pipeline.search("What is rough set theory?", limit=5)

# Adjust threshold for quality
results = pipeline.search(query, threshold=0.7)  # Higher = stricter
results = pipeline.search(query, threshold=0.3)  # Lower = more results

# Filter by chunk type
results = pipeline.search("tables about costs", chunk_type="table")
results = pipeline.search("formula for optimization", chunk_type="formula")
```

## Understanding Similarity Scores

### Typical Score Ranges (Ensemble Embeddings)

| Score Range | Quality | Interpretation |
|------------|---------|----------------|
| 0.85+ | Excellent | Near-perfect semantic match |
| 0.75-0.85 | Very Good | Strong relevance |
| 0.65-0.75 | Good | Moderate relevance |
| 0.50-0.65 | Fair | Weak relevance |
| < 0.50 | Poor | Likely irrelevant |

### Test Results from Our System

**Query: "What is rough set theory?"**
- Top result: 0.7382 (73.8%)
- Type: TABLE
- Comment: Tables often have lower scores than text for conceptual queries

**Query: "Minimize transportation cost"**
- Top result: 0.7790 (77.9%)
- Type: TABLE
- Comment: Tables perform well for specific domain queries

**Query: "Tables about costs"** (filtered to tables)
- Top result: 0.8333 (83.3%)
- Type: TABLE
- Comment: Explicit filtering + matching content type = high scores

## Threshold Selection Guide

### Conservative (Precision-focused)
```python
threshold = 0.7  # Only very relevant results
```
**Use when:**
- You need high-quality results only
- False positives are costly
- Working with critical applications

### Balanced (Default)
```python
threshold = 0.5  # Good balance
```
**Use when:**
- General purpose search
- Want reasonable result count
- OK with some lower-quality matches

### Exploratory (Recall-focused)
```python
threshold = 0.3  # See more results
```
**Use when:**
- Exploring the dataset
- Missing expected results
- Debugging search behavior

### Debug Mode
```python
threshold = 0.0  # All results
```
**Use when:**
- Checking what's in the database
- Understanding score distribution
- Troubleshooting search issues

## Chunk Type Filtering

Filter by the type of content you're looking for:

```python
# Find text paragraphs
results = pipeline.search("rough set theory", chunk_type="text")

# Find mathematical formulas
results = pipeline.search("optimization equation", chunk_type="formula")

# Find data tables
results = pipeline.search("cost matrix", chunk_type="table")

# Find theorems
results = pipeline.search("properties of rough sets", chunk_type="theorem")
```

## Performance & Scaling

### Current Setup (36 embeddings)
- **Index:** Disabled (sequential scan)
- **Search time:** 1-5 ms
- **Reason:** IVFFlat requires >100 vectors for good recall

### Scaling Behavior

| Embedding Count | Index Strategy | Expected Latency |
|----------------|---------------|------------------|
| < 100 | No index (seq scan) | 1-10 ms |
| 100-1,000 | Simple IVFFlat | 5-20 ms |
| 1,000-100K | Optimized IVFFlat | 10-100 ms |
| 100K+ | HNSW recommended | 50-200 ms |

### Memory Usage

```
Single Embedding: 1024 dim × 4 bytes = 4 KB
1,000 embeddings: ~4 MB
100,000 embeddings: ~400 MB
1M embeddings: ~4 GB
```

**Our system (36 embeddings):**
- Embedding storage: 147 KB
- Model memory: 2-3 GB (SPECTER2 + E5-Large + SciBERT)
- Total: ~4-5 GB (well within 48 GB capacity)

## Common Issues & Solutions

### Issue: No results returned

**Check 1: Database has data**
```python
stats = pipeline.vector_store.get_stats()
print(stats)  # Should show embeddings
```

**Check 2: Threshold too high**
```python
# Try lower threshold
results = pipeline.search(query, threshold=0.0)
# If results appear, adjust threshold down
```

**Check 3: Query-document mismatch**
```python
# Try broader queries
instead of: "specific formula for XYZ"
try: "optimization formula" or "mathematical equation"
```

### Issue: Too many low-quality results

**Solution: Raise threshold**
```python
results = pipeline.search(query, threshold=0.7)  # Stricter filtering
```

### Issue: Missing expected results

**Solution: Lower threshold + examine scores**
```python
# Get all results with scores
results = pipeline.search(query, limit=20, threshold=0.0)
for r in results:
    print(f"{r['similarity']:.2%} - {r['content'][:100]}")
```

## Advanced Usage

### Hybrid Search (coming soon)
Combine keyword and semantic search for best results:

```python
# Planned feature
results = pipeline.hybrid_search(
    query="rough set theory",
    semantic_weight=0.7,
    keyword_weight=0.3
)
```

### Re-ranking (coming soon)
Use different models to re-rank top results:

```python
# Planned feature
results = pipeline.search_with_reranking(
    query="...",
    first_stage_limit=20,
    rerank_limit=5,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)
```

## Model Information

### Ensemble Components

**SPECTER2** (allenai/specter2_base)
- Dimension: 768
- Specialty: Scientific papers, citations
- Training: 1.9M papers from S2ORC

**E5-Large** (intfloat/multilingual-e5-large)
- Dimension: 1024
- Specialty: Multilingual, general purpose
- Training: Large-scale contrastive learning

**SciBERT** (allenai/scibert_scivocab_uncased)
- Dimension: 768
- Specialty: Scientific vocabulary
- Training: 1.14M papers from Semantic Scholar

**Ensemble Strategy:**
- Weighted average: SPECTER2 (40%) + E5-Large (35%) + SciBERT (25%)
- Final dimension: 1024 (padding + normalization)
- Distance metric: Cosine similarity

## Tips & Best Practices

1. **Start with default threshold (0.5)** - adjust based on results
2. **Use chunk type filtering** when you know what you're looking for
3. **Check similarity scores** to understand result quality
4. **Lower threshold for exploration** - raise for production use
5. **Monitor search latency** as dataset grows
6. **Plan for index creation** at 100+ embeddings
7. **Consider query formulation** - more specific != better scores
8. **Test with various queries** to find optimal threshold

## Examples

### Example 1: Find relevant text chunks
```python
results = pipeline.search(
    "What are the properties of rough sets?",
    limit=5,
    chunk_type="text",
    threshold=0.6
)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['similarity']:.0%} | {result['content'][:100]}...")
```

### Example 2: Find tables with specific data
```python
results = pipeline.search(
    "transportation costs between cities",
    limit=3,
    chunk_type="table",
    threshold=0.5
)
```

### Example 3: Broad exploration
```python
results = pipeline.search(
    "optimization",
    limit=10,
    threshold=0.3  # Lower threshold for exploration
)
```

### Example 4: Debug mode
```python
# See all results with scores
all_results = pipeline.search(query, limit=50, threshold=0.0)

# Analyze score distribution
scores = [r['similarity'] for r in all_results]
print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
print(f"Mean: {sum(scores)/len(scores):.2f}")
```
