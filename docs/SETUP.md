# EmbedRAG - Setup Guide

Scientific paper embedding & retrieval system with PostgreSQL + pgvector

## 📋 Prerequisites

- macOS with Homebrew
- Python 3.10+
- PostgreSQL 14+
- 48GB RAM (optimal for ensemble models)
- 10GB disk space for models

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PostgreSQL + pgvector (macOS)

```bash
# Install PostgreSQL
brew install postgresql

# Install pgvector extension
brew install pgvector

# Start PostgreSQL service
brew services start postgresql

# Create database
createdb embed_rag

# Enable pgvector extension
psql embed_rag -c "CREATE EXTENSION IF NOT EXISTS vector"

# Verify
psql embed_rag -c "SELECT extname FROM pg_extension WHERE extname = 'vector'"
```

**Output should be:**
```
 extname
---------
 vector
(1 row)
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

⚠️ **First run will download ~5GB of models** (SPECTER2, E5-Large, SciBERT)

### 4. Set up Environment

```bash
cp .env.example .env
# Edit .env if needed (default values should work)
```

### 5. Verify Setup

```bash
# Quick activation
source activate.sh

# Test components individually
python deepdoc_test.py
python ensemble_embeddings.py
python vector_store.py
```

---

## 📊 System Architecture

```
PDF Input
    ↓
┌─────────────────────┐
│   DeepDoc Processor │  Extract: text, formulas, tables
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Ensemble Embedding  │  SPECTER2 (768) + E5-Large (1024) + SciBERT (768)
│     Model           │  → Combined: 1024-dim normalized
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Vector Store       │  PostgreSQL + pgvector
│  (pgvector)         │  Cosine similarity search
└─────────────────────┘
    ↓
Semantic Search Results
```

---

## 🔧 Component Testing

### Test 1: PDF Extraction (DeepDoc)

```bash
python deepdoc_test.py
```

**Expected Output:**
```
DEEPDOC EXTRACTION SUMMARY
──────────────────────────────────────
Document: A class of rough multiple objective programming...
Pages: 21

Extracted Components:
  • Text chunks: 45
  • Formulas: 12
  • Tables: 5
```

### Test 2: Embedding Models

```bash
python ensemble_embeddings.py
```

**Expected Output:**
```
ENSEMBLE EMBEDDING TEST
──────────────────────────────────────
Generated embeddings shape: (4, 1024)
Each embedding dimension: 1024

Sample 1:
  Text: The rough set theory provides...
  Embedding norm: 1.0000
  First 5 dims: [0.12, 0.34, ...]
```

### Test 3: Vector Store

```bash
python vector_store.py
```

**Expected Output:**
```
VECTOR STORE TEST
──────────────────────────────────────
Database Stats:
  Total embeddings: 0
  Unique papers: 0
  By type: {}

✓ Database connected and ready
```

### Test 4: Full Pipeline

```bash
python main_pipeline.py
```

**Expected Output:**
```
RAG PIPELINE - Scientific Paper Embedding & Retrieval
══════════════════════════════════════════════════════

Processing: A class of rough multiple objective...
Paper ID: rough_programming

┌─ STEP 1: Extract chunks (DeepDoc)
├─ Text chunks: 45
├─ Formula chunks: 12
└─ Table chunks: 5
└─ ✓ Extraction complete

┌─ STEP 2: Generate embeddings (Ensemble)
├─ Embedding 45 text chunks...
├─ Embedding 12 formula chunks...
├─ Embedding 5 table chunks...
└─ ✓ Generated 62 embeddings

┌─ STEP 3: Store embeddings (PostgreSQL + pgvector)
├─ Inserted: 62 records
├─ Database total: 62 embeddings
└─ ✓ Storage complete

Testing semantic search...

Query: What is rough set theory?
──────────────────────────────────
1. [TEXT] (Similarity: 92%)
   Content: The rough set theory provides an effective tool...
```

---

## 🧪 Manual Testing

### Test Specific Components

```python
# Python REPL
from ensemble_embeddings import EnsembleEmbeddingModel
from vector_store import VectorStore

# Test embeddings
model = EnsembleEmbeddingModel()
emb = model.embed_ensemble(["test text"], normalize=True)
print(f"Embedding shape: {emb.shape}")

# Test vector store
store = VectorStore()
stats = store.get_stats()
print(f"Database: {stats}")
```

---

## 📝 Processing Pipeline

### Step 1: Extract Paper Components

```bash
from deepdoc_test import DeepDocProcessor

processor = DeepDocProcessor("path/to/paper.pdf")
chunks = processor.extract_from_pdf()
# Returns: {text: [...], formula: [...], table: [...], metadata: {...}}
```

### Step 2: Generate Embeddings

```bash
from ensemble_embeddings import EnsembleEmbeddingModel

model = EnsembleEmbeddingModel()
embeddings = model.batch_embed(chunks, batch_size=32)
# Each chunk now has 1024-dim embedding
```

### Step 3: Store in Vector DB

```bash
from vector_store import VectorStore

store = VectorStore()
store.insert_batch(embeddings)
```

### Step 4: Semantic Search

```bash
results = store.search_similar(
    query_embedding=query_vec,
    limit=10,
    chunk_type="text"
)
```

---

## 🔍 Troubleshooting

### PostgreSQL Connection Error

```
Error: psycopg2.OperationalError: could not connect to server
```

**Solution:**
```bash
# Check PostgreSQL is running
brew services list

# Start if not running
brew services start postgresql

# Check pgvector extension
psql embed_rag -c "SELECT extname FROM pg_extension"
```

### Model Download Issues

```
Error: Failed to download model from Hugging Face
```

**Solution:**
```bash
# Clear cache and retry
rm -rf model_cache/
# Models will re-download on next run

# Or set custom cache directory
export HF_HOME=/path/to/large/disk
```

### Memory Issues

```
RuntimeError: CUDA out of memory
```

**Solution:**
Edit `.env`:
```
EMBEDDING_MODEL_DEVICE=cpu  # Use CPU instead
BATCH_SIZE=8               # Reduce batch size
```

---

## 📦 Project Structure

```
embed_rag/
├── requirements.txt          # Python dependencies
├── .env.example             # Configuration template
├── .env                     # Configuration (gitignored)
├── activate.sh              # Quick activation script
│
├── deepdoc_test.py          # PDF extraction & chunking
├── ensemble_embeddings.py   # SPECTER2 + E5-Large + SciBERT
├── vector_store.py          # PostgreSQL + pgvector
├── main_pipeline.py         # Full orchestration
│
├── model_cache/             # Downloaded models (~5GB)
├── pipeline.log             # Execution logs
├── chunks.json              # Extracted chunks (temporary)
└── embeddings.json          # Embeddings (temporary)
```

---

## 🎯 Next Steps

1. **Process Your Papers**
   ```bash
   python main_pipeline.py
   ```

2. **Batch Processing**
   Edit `main_pipeline.py` to loop through multiple PDFs

3. **Advanced Search**
   Implement hybrid search (semantic + keyword)

4. **RAG Integration**
   Connect to LLM for question-answering

---

## 📚 Resources

- [SPECTER2 Documentation](https://github.com/allenai/SPECTER)
- [E5 Embeddings](https://huggingface.co/intfloat/multilingual-e5-large)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## 📄 License

MIT License - See LICENSE file
