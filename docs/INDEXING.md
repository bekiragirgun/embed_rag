# Vector Indexing Guide

## Overview

This guide explains how to manage pgvector indexes for large-scale document embeddings (40GB+ datasets).

## Index Strategy

### IVFFlat Index

pgvector uses **IVFFlat** (Inverted File with Flat Compression) for approximate nearest neighbor search:

- **Without index**: Sequential scan - O(N) time complexity
- **With IVFFlat**: Approximate search - O(sqrt(N)) time complexity
- **Trade-off**: Speed vs. recall (may miss some results)

### When to Use Index

| Dataset Size | Recommendation | Reason |
|--------------|----------------|--------|
| < 1K vectors | **No index** | Sequential scan is fast enough |
| 1K - 100K | **Optional** | Index helps but not critical |
| 100K - 1M | **Recommended** | Significant speed improvement |
| 1M+ vectors | **Essential** | Sequential scan becomes impractical |

## Configuration Parameters

### `CREATE_INDEX` (default: `false`)

Controls whether to create index during pipeline initialization.

**For large datasets (40GB+):**
- Set to `false` during data loading
- Create index AFTER all data is loaded
- Prevents index rebuilds after each batch

### `INDEX_LISTS` (default: `1000`)

Number of clusters for IVFFlat index.

**Recommended values:**
- **100K-1M vectors**: 500-1000 lists
- **1M-10M vectors**: 1000-3000 lists
- **10M+ vectors**: 3000-5000 lists

**Formula**: `sqrt(num_vectors)` to `num_vectors / 10`

**Trade-offs:**
- **Too few lists**: Slower search, better recall
- **Too many lists**: Faster search, lower recall
- **Sweet spot**: Balance between speed and accuracy

## Usage Examples

### Small Dataset (< 100K vectors)

No index needed - sequential scan is fast:

```python
from main_pipeline import RAGPipeline

# Default settings - no index
pipeline = RAGPipeline()

# Process papers
pipeline.process_pdf("paper1.pdf")
pipeline.process_pdf("paper2.pdf")
# ... more papers ...

# Search works fast without index
results = pipeline.search("rough set theory")
```

### Medium Dataset (100K - 1M vectors)

Create index during initialization:

```python
# Create index automatically
pipeline = RAGPipeline(
    create_index=True,
    index_lists=800  # For ~500K vectors
)

# Process papers
for pdf_file in pdf_files:
    pipeline.process_pdf(pdf_file)

# Search uses index automatically
results = pipeline.search("optimization problem")
```

### Large Dataset (40GB+, millions of vectors)

**Step 1: Load all data without index**

```python
# Disable index during loading
pipeline = RAGPipeline(create_index=False)

# Process all PDFs
import os
pdf_dir = "/path/to/pdfs"
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pipeline.process_pdf(os.path.join(pdf_dir, pdf_file))
        print(f"Processed: {pdf_file}")
```

**Step 2: Create index after loading**

```python
# Create index with appropriate lists parameter
# For 5M vectors: use 2000-2500 lists
pipeline.create_index(lists=2000)
```

**Why this approach?**
- Index creation can take 30 minutes to 2 hours for large datasets
- Creating index once is much faster than rebuilding after each batch
- Reduces memory pressure during data loading

## Monitoring Index Performance

### Check Index Existence

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'paper_embeddings';
```

### Compare Query Performance

```python
import time

query = "transportation optimization"

# Without index (if testing)
start = time.time()
results = pipeline.search(query, limit=10)
no_index_time = time.time() - start

# With index
start = time.time()
results = pipeline.search(query, limit=10)
index_time = time.time() - start

print(f"Without index: {no_index_time:.3f}s")
print(f"With index: {index_time:.3f}s")
print(f"Speedup: {no_index_time/index_time:.1f}x")
```

### Expected Performance

For 1M vectors on M4 Max (48GB RAM):

| Operation | Sequential Scan | IVFFlat (1000 lists) |
|-----------|-----------------|----------------------|
| Search 10 results | ~800ms | ~50ms |
| Search 100 results | ~900ms | ~80ms |
| Index creation | N/A | ~15 minutes |

## Advanced: Custom Index Strategies

### Rebuild Index with Different Parameters

```python
# Connect to existing database
pipeline = RAGPipeline(create_index=False)

# Get current vector count
stats = pipeline.vector_store.get_stats()
num_vectors = stats['total_embeddings']

# Calculate optimal lists
import math
optimal_lists = int(math.sqrt(num_vectors))

# Recreate index
pipeline.create_index(lists=optimal_lists)
```

### Drop Index (for debugging)

```python
from vector_store import VectorStore

store = VectorStore()
with store.engine.connect() as conn:
    conn.execute("DROP INDEX IF EXISTS ix_embedding")
    conn.commit()
print("Index dropped - using sequential scan")
```

### Manual Index Creation (SQL)

```sql
-- Drop old index
DROP INDEX IF EXISTS ix_embedding;

-- Create new index with 2000 lists
CREATE INDEX ix_embedding ON paper_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists=2000);

-- Analyze table for query planner
ANALYZE paper_embeddings;
```

## Troubleshooting

### Index Creation Times Out

**Solution**: Increase PostgreSQL work memory:

```sql
-- Check current setting
SHOW work_mem;

-- Increase temporarily (128MB)
SET work_mem = '128MB';

-- Create index
CREATE INDEX ix_embedding ON paper_embeddings ...;
```

### Search Returns Fewer Results Than Expected

**Cause**: IVFFlat is approximate - may miss some results.

**Solutions**:
1. Lower `lists` parameter (improves recall, slower search)
2. Use sequential scan for critical queries:
   ```sql
   SET enable_indexscan = OFF;
   -- Run query
   SET enable_indexscan = ON;
   ```

### Out of Memory During Index Creation

**Solution**: Create index in smaller batches or increase system memory.

For M4 Max with 48GB RAM, index creation for 10M vectors should work fine.

## Best Practices Summary

1. ✅ **Disable index during bulk loading** (`create_index=false`)
2. ✅ **Create index after all data is loaded** (`pipeline.create_index()`)
3. ✅ **Use auto-calculated lists** for most cases
4. ✅ **Monitor search performance** and adjust if needed
5. ✅ **Consider recall requirements** - some queries need 100% recall
6. ⚠️ **Don't recreate index frequently** - very expensive operation
7. ⚠️ **Test with your queries** before production deployment

## Environment Variables

Add to `.env`:

```bash
# Disable auto-indexing for large datasets
CREATE_INDEX=false

# Number of lists (auto-calculated if not set)
INDEX_LISTS=1000

# Search threshold (0.0 = return all, 1.0 = perfect match only)
SIMILARITY_THRESHOLD=0.0
```

## References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [IVFFlat Algorithm](https://github.com/pgvector/pgvector#ivfflat)
- [Query Performance Tips](https://github.com/pgvector/pgvector#query-performance)
