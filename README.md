# EmbedRAG - Scientific Paper Embedding & Retrieval System

A comprehensive RAG (Retrieval Augmented Generation) system for processing scientific papers with mathematical content using ensemble embeddings and semantic search.

> **🇹🇷 Türkçe kullanıcılar için:** [5 Dakikada Başlangıç Rehberi](docs/QUICKSTART_TR.md) | [Doküman İndeksi](docs/INDEX.md)

## ✨ Features

- **Multi-Modal Document Processing**
  - Text extraction with semantic awareness
  - Formula detection and LaTeX parsing
  - Table structure recognition
  - Metadata preservation (page, section, type)

- **Ensemble Embedding Models**
  - SPECTER2 (768-dim): Scientific paper specialist
  - E5-Large (1024-dim): Multilingual + high performance
  - SciBERT (768-dim): Scientific vocabulary expert
  - Combined: 1024-dim normalized embeddings

- **Vector Database (PostgreSQL + pgvector)**
  - Efficient cosine similarity search
  - Metadata filtering and advanced queries
  - Persistent storage with indexing
  - Batch operations support

- **End-to-End Pipeline**
  - PDF → Chunks → Embeddings → Vector Store
  - Single command orchestration
  - Progress tracking and logging
  - Error handling and recovery

## 🎯 Use Cases

1. **Scientific Literature Mining**
   - Find similar papers in your collection
   - Discover related mathematical formulations
   - Cross-reference equations and theorems

2. **Research Assistant**
   - Answer questions about paper content
   - Retrieve relevant sections by semantic meaning
   - Find specific formulas and their contexts

3. **Document Management**
   - Organize large paper collections
   - Deduplicate similar research
   - Build knowledge graphs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│          PDF Scientific Papers                       │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   DeepDoc Processor │  Extract components
        │  (OCR + Layout)     │  • Text chunks
        └──────────┬──────────┘  • Formulas (LaTeX)
                   │             • Tables (Markdown)
        ┌──────────▼──────────────────────────┐
        │  Ensemble Embedding Model            │
        │  ┌──────┐ ┌────────┐ ┌─────────┐   │
        │  │SPECTER2│ E5-Large│ SciBERT │   │
        │  └──────┘ └────────┘ └─────────┘   │
        │         ↓ Combined ↓                │
        │      1024-dim Embeddings           │
        └──────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   Vector Store      │  PostgreSQL + pgvector
        │ ┌────────────────┐  │  • Cosine similarity
        │ │ Embeddings DB  │  │  • Metadata filtering
        │ │ Chunk Metadata │  │  • IVFFLAT indexing
        │ └────────────────┘  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Semantic Search    │  Top-K retrieval
        │  Results with       │  Similarity scores
        │  Metadata           │  Filtered results
        └─────────────────────┘
```

## 📊 Data Flow Example

### Input: Scientific Paper (PDF)
```
A class of rough multiple objective programming and its application
to solid transportation problem
- 21 pages
- Multiple formulas
- 5 tables
```

### Processing:
```json
{
  "text_chunks": 45,
  "formula_chunks": 12,
  "table_chunks": 5,
  "total_embeddings": 62
}
```

### Output: Searchable Database
```
Query: "minimize transportation cost"
        ↓
Results:
1. [FORMULA] Similarity: 94%
   "∑c_ijk·x_ijk → minimize total transportation cost"

2. [TEXT] Similarity: 87%
   "The objective is to minimize the total cost of transportation..."

3. [TABLE] Similarity: 82%
   "Table showing transportation costs by route..."
```

## 🚀 Quick Start

### Prerequisites
```bash
# macOS with Docker (recommended)
brew install docker
```

### Installation

1. **Clone & Setup**
   ```bash
   cd /Users/bekiragirgun/Projects/embed_rag
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Database (Docker)**
   ```bash
   ./docker-up.sh
   # PostgreSQL with pgvector runs on port 5433
   ```

3. **Process Your PDFs**
   ```bash
   export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/embed_rag"
   python process_large_dataset.py
   ```

   Script will ask:
   - PDF klasörü yolu? (PDF folder path)
   - Alt klasörlerdeki PDF'ler de işlensin mi? (Process subdirectories?)
   - Index oluşturulsun mu? (Create index after loading?)

**Detailed setup:** See [SETUP.md](docs/SETUP.md)

## 📖 Usage Examples

### Interactive Processing (Recommended - Easy)

```bash
# Set database connection
export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/embed_rag"

# Run interactive script
python process_large_dataset.py

# Script will ask:
# 1. PDF klasörü yolu? → /path/to/your/pdfs
# 2. Alt klasörlerdeki PDF'ler de işlensin mi? → evet (recommended for nested folders)
# 3. Index oluşturulsun mu? → evet (for large datasets)
# 4. Index lists sayısı? → 2000 (default, for ~5M vectors)
```

### Programmatic Usage (Advanced)

**Small Datasets (< 100 PDFs):**
```python
from main_pipeline import RAGPipeline

# Initialize (no index needed)
pipeline = RAGPipeline()

# Process paper
result = pipeline.process_pdf("path/to/paper.pdf", paper_id="paper_001")

# Search
results = pipeline.search("rough set theory", limit=5)
```

**Large Dataset Processing (40GB+):**
```python
from main_pipeline import RAGPipeline
import os

# Disable index during data loading
pipeline = RAGPipeline(create_index=False)

# Process all PDFs
pdf_dir = "/path/to/pdfs"
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        result = pipeline.process_pdf(os.path.join(pdf_dir, pdf_file))
        print(f"Processed: {pdf_file}")

# Create index AFTER all data is loaded (one-time operation)
# For ~5M vectors, use lists=2000
pipeline.create_index(lists=2000)

# Now search is fast
results = pipeline.search("optimization problem", limit=10)
```

**Why this approach?**
- Index creation can take 30-120 minutes for millions of vectors
- Creating once is faster than rebuilding after each batch
- See [INDEXING.md](docs/INDEXING.md) for detailed strategy

### Component Usage

#### 1. Extract Paper Components

```python
from deepdoc_test import DeepDocProcessor

processor = DeepDocProcessor("paper.pdf")
chunks = processor.extract_from_pdf()

print(f"Text chunks: {len(chunks['text'])}")
print(f"Formulas: {len(chunks['formula'])}")
print(f"Tables: {len(chunks['table'])}")
```

#### 2. Generate Embeddings

```python
from ensemble_embeddings import EnsembleEmbeddingModel

model = EnsembleEmbeddingModel()

# Embed single text
embedding = model.embed_by_type("Minimize transportation cost", chunk_type="formula")

# Batch embed
chunks = [
    {"id": "1", "content": "text 1", "type": "text"},
    {"id": "2", "content": "formula here", "type": "formula"}
]
result = model.batch_embed(chunks, batch_size=32)
```

#### 3. Vector Database Operations

```python
from vector_store import VectorStore

store = VectorStore()

# Insert embeddings
store.insert_batch(embeddings_list)

# Search
results = store.search_similar(
    query_embedding=query_vec,
    limit=10,
    chunk_type="text",
    threshold=0.5
)

# Stats
stats = store.get_stats()
print(f"Total embeddings: {stats['total_embeddings']}")
```

## 🧪 Testing

### Test Individual Components

```bash
# Test PDF extraction
python deepdoc_test.py

# Test embedding models
python ensemble_embeddings.py

# Test vector database
python vector_store.py

# Full pipeline test
python main_pipeline.py
```

### Expected Test Results

- ✅ DeepDoc: Extract 45-60 chunks per 20-page paper
- ✅ Embeddings: Generate 1024-dim vectors in <5s per 100 chunks
- ✅ Vector Store: Insert 100 embeddings in <2s
- ✅ Search: Return top-5 results in <100ms

## 📁 Project Structure

```
embed_rag/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env                           # Configuration (local)
├── docker-compose.yml             # PostgreSQL + pgvector setup
├── docker-up.sh                   # Start database script
│
├── docs/                          # 📚 Documentation
│   ├── QUICKSTART_TR.md           # 🇹🇷 Quick start guide (Turkish)
│   ├── SETUP.md                   # Detailed setup guide
│   ├── INDEXING.md                # Vector indexing guide (40GB datasets)
│   ├── SEARCH_GUIDE.md            # Semantic search tuning guide
│   └── CHANGELOG.md               # Recent updates and fixes
│
├── process_large_dataset.py      # ⭐ Interactive PDF processing (RECOMMENDED)
│   ├─ User-friendly prompts
│   ├─ Automatic validation
│   ├─ Progress tracking
│   └─ Smart index configuration
│
├── main_pipeline.py              # Programmatic pipeline
│   ├─ RAGPipeline class
│   ├─ End-to-end workflow
│   ├─ Optional indexing
│   └─ Search interface
│
├── deepdoc_test.py               # PDF processing module
│   ├─ DeepDocProcessor class
│   ├─ Formula extraction
│   ├─ Table processing
│   └─ JSON export
│
├── ensemble_embeddings.py        # Embedding module
│   ├─ EnsembleEmbeddingModel class
│   ├─ SPECTER2 + E5-Large + SciBERT
│   ├─ Type-aware embedding
│   └─ Batch processing
│
├── vector_store.py               # Vector database module
│   ├─ VectorStore class
│   ├─ PostgreSQL + pgvector
│   ├─ Optional IVFFlat indexing
│   └─ Similarity search
│
├── model_cache/                  # Downloaded models (~5GB)
├── pipeline.log                  # Execution logs
└── venv/                          # Virtual environment
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://postgres@localhost:5432/embed_rag
DB_HOST=localhost
DB_PORT=5432

# Models
EMBEDDING_MODEL_DEVICE=cuda  # or 'cpu'
MODEL_CACHE_DIR=./model_cache
EMBEDDING_DIM=1024

# Pipeline
BATCH_SIZE=32
VECTOR_SEARCH_LIMIT=10
SIMILARITY_THRESHOLD=0.0  # 0.0 = return all results with scores

# Vector Index (NEW - for large datasets)
CREATE_INDEX=false  # Disable during bulk loading
INDEX_LISTS=1000    # Number of clusters for IVFFlat

# Logging
LOG_LEVEL=INFO
LOG_FILE=pipeline.log
```

## 📊 Performance Metrics

### Processing Speed (M4 Max, 48GB RAM)

| Task | Time | Throughput |
|------|------|-----------|
| PDF Extraction (20 pages) | ~2s | - |
| Embedding 100 chunks | ~5s | 20 chunks/s |
| Vector DB Insert (100) | ~1.5s | 67 insert/s |
| Semantic Search (1M vectors) | ~50ms | - |

### Resource Usage

| Component | Memory | CPU | GPU |
|-----------|--------|-----|-----|
| DeepDoc | 500MB | 20% | - |
| SPECTER2 | 2GB | 10% | ✓ |
| E5-Large | 2.5GB | 10% | ✓ |
| SciBERT | 2GB | 10% | ✓ |
| PostgreSQL | 1GB | 5% | - |
| **Total** | **~10GB** | **55%** | ✓ |

## 🎓 Scientific Background

### Why Ensemble Models?

1. **SPECTER2**: Trained on 7M scientific papers → understands domain-specific language and mathematical notation
2. **E5-Large**: 100+ languages including Turkish → multilingual queries
3. **SciBERT**: Specialized scientific vocabulary → formula and theorem understanding

Combined embedding = best of all three models!

### Supported Paper Types

- ✅ Mathematics & Optimization (Rough Sets, Fuzzy Logic, Linear Programming)
- ✅ Computer Science & AI
- ✅ Physics & Engineering
- ✅ Biology & Medicine
- ✅ Economics & Finance

## 📖 Documentation

- **[QUICKSTART_TR.md](docs/QUICKSTART_TR.md)** - 🇹🇷 **Hızlı başlangıç** (5 dakikada başlayın)
- **[INDEXING.md](docs/INDEXING.md)** - Vector indexing strategy for large datasets (40GB+)
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Recent fixes and updates
- **[SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md)** - Complete guide to semantic search tuning
- **[SETUP.md](docs/SETUP.md)** - Detailed installation and configuration

### Recent Updates (2025-01-18)

✅ **NEW:** Optional IVFFlat indexing for large datasets (40GB+)
✅ **Fixed:** Search returning zero results
✅ **Fixed:** Search threshold too high (lowered to 0.0)
✅ **Improved:** Debug logging for search operations
✅ **Added:** Manual index creation after bulk loading
✅ **NEW:** Interactive Turkish script (process_large_dataset.py)

**Key changes:**
- Index creation now **disabled by default** (enable with `create_index=True`)
- New `pipeline.create_index(lists=2000)` method for post-loading indexing
- Threshold changed from 0.5 to 0.0 (return all results with scores)
- Auto-calculated `lists` parameter using sqrt(N) heuristic
- See [INDEXING.md](docs/INDEXING.md) for 40GB dataset strategies

## 🐛 Troubleshooting

See [SETUP.md - Troubleshooting](docs/SETUP.md#-troubleshooting) section

**Common Issues:**
1. **No search results** → Check [SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md#common-issues--solutions)
2. **Slow search** → Review index strategy in [CHANGELOG.md](docs/CHANGELOG.md)
3. **Low similarity scores** → See [SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md#understanding-similarity-scores)

## 📚 References

- **SPECTER2**: [Allen AI - SPECTER](https://github.com/allenai/SPECTER)
- **E5 Embeddings**: [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- **SciBERT**: [Allen AI - SciBERT](https://github.com/allenai/scibert)
- **pgvector**: [pgvector Documentation](https://github.com/pgvector/pgvector)
- **DeepDoc**: [RAGFlow DeepDoc](https://github.com/infiniflow/ragflow/tree/main/deepdoc)

## 📄 License

MIT License

## 👨‍💻 Development

### Adding New Embedding Models

Edit `ensemble_embeddings.py`:
```python
self.new_model = SentenceTransformer(
    "model-name",
    device=self.device,
    cache_folder=self.cache_dir
)
```

### Batch Processing Multiple Papers

Edit `main_pipeline.py`:
```python
pdf_files = glob.glob("/path/to/pdfs/*.pdf")
for pdf_path in pdf_files:
    result = pipeline.process_pdf(pdf_path)
```

### Custom Search Filters

Edit `vector_store.py`:
```python
results = store.search_similar(
    query_embedding,
    chunk_type="formula",
    paper_id="specific_paper",
    threshold=0.6
)
```

## 📞 Support

For issues, questions, or suggestions:
1. Check [SETUP.md](docs/SETUP.md) troubleshooting section
2. Review test outputs in `pipeline.log`
3. Verify database connection with `python vector_store.py`
4. 🇹🇷 Quick start guide: [QUICKSTART_TR.md](docs/QUICKSTART_TR.md)

---

**Created**: October 2024
**Platform**: macOS (Monterey+)
**Hardware**: Optimized for M4 Max with 48GB RAM
