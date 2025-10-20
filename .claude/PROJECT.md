# EmbedRAG Project - Claude Context

## Project Overview
Scientific paper embedding and retrieval system for M4 Max (48GB RAM, Apple Silicon).

## Tech Stack
- **Embeddings:** SPECTER2 + E5-Large + SciBERT → 1024-dim ensemble
- **Database:** PostgreSQL + pgvector (Docker, port 5433)
- **Processing:** DeepDoc (text, formulas, tables)
- **Platform:** macOS (Apple Silicon optimized)

## Key Files
- `process_large_dataset.py` - Interactive PDF processor (RECOMMENDED)
- `main_pipeline.py` - Programmatic pipeline
- `vector_store.py` - Database with optional IVFFlat indexing
- `ensemble_embeddings.py` - Ensemble embedding model

## Important Configurations
- Database URL: `postgresql://postgres:postgres@localhost:5433/embed_rag`
- Create index: **Disabled by default** (enable after loading all data)
- Search threshold: 0.0 (return all with scores, typical good: 0.71-0.83)

## Documentation
All docs in `docs/` folder:
- `QUICKSTART_TR.md` - Turkish quick start (5 min)
- `INDEX.md` - Documentation index
- `INDEXING.md` - Index strategy for 40GB+ datasets
- `SEARCH_GUIDE.md` - Search optimization
- `CHANGELOG.md` - Recent updates
- `research/` - MLX, QWEN research notes

## Critical Notes
1. **Database:** Always use port 5433 (has pgvector), NOT 5432
2. **Index:** Create AFTER loading all data for 40GB datasets
3. **Search:** Threshold 0.0 → returns all with scores for analysis
4. **Turkish:** Interactive script asks for PDF folder path

## Recent Updates
- ✅ **2025-01-19:** Recursive directory scanning for nested PDF folders
- ✅ **2025-01-18:** Fixed search returning 0 results (removed index for small datasets)
- ✅ **2025-01-18:** Made IVFFlat index optional (default: disabled)
- ✅ **2025-01-18:** Created interactive Turkish script
- ✅ **2025-01-18:** Organized docs into `docs/` folder
- ✅ **2025-01-18:** Lowered search threshold to 0.0

## User Profile
- Platform: Macbook Pro M4 Max 48GB RAM, 40 GPU, 1TB SSD
- Dataset: ~40GB of scientific papers
- Language: Turkish (use Turkish in comments when appropriate)
- Focus: Research, embeddings, RAG systems
