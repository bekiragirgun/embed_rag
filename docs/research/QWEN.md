# EmbedRAG - Scientific Paper Embedding & Retrieval System

## Overview

EmbedRAG is a comprehensive RAG (Retrieval Augmented Generation) system designed for processing scientific papers with mathematical content using ensemble embeddings and semantic search. It combines multiple state-of-the-art embedding models with a PostgreSQL vector database to create a powerful system for scientific literature mining and analysis.

The project features multi-modal document processing for text, formulas, and tables, with specialized handling for scientific content including LaTeX formulas and structured data. It uses an ensemble of embedding models (SPECTER2, E5-Large, and SciBERT) to achieve high-quality semantic representations.

## Architecture

The system follows a pipeline architecture:
1. **PDF Processing**: Uses DeepDoc and pdfplumber to extract text, formulas (LaTeX), and tables from scientific papers
2. **Ensemble Embedding**: Combines SPECTER2, E5-Large, and SciBERT models to create 1024-dimensional embeddings
3. **Vector Storage**: Stores embeddings in PostgreSQL with pgvector extension for efficient similarity search
4. **Semantic Search**: Provides semantic search capabilities across the stored paper chunks

## Key Components

- **`main_pipeline.py`**: Orchestration of the complete RAG pipeline
- **`deepdoc_test.py`**: PDF processing with focus on mathematical content extraction
- **`ensemble_embeddings.py`**: Ensemble of SPECTER2, E5-Large, and SciBERT models
- **`vector_store.py`**: PostgreSQL + pgvector implementation for embedding storage and retrieval

## Building and Running

### Prerequisites
- Python 3.10+
- PostgreSQL with pgvector extension
- Docker (for containerized setup)

### Quick Setup
1. Clone the repository and set up virtual environment:
   ```bash
   cd /Users/bekiragirgun/Projects/embed_rag
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure the database:
   ```bash
   createdb embed_rag
   psql embed_rag -c "CREATE EXTENSION IF NOT EXISTS vector"
   ```

3. Run the complete pipeline:
   ```bash
   python main_pipeline.py
   ```

### Docker-based Setup
The project includes a `docker-compose.yml` that sets up PostgreSQL with pgvector and an embedding server. Run with:
```bash
docker-compose up -d
```

## Configuration

The system uses environment variables defined in `.env`:
- `DATABASE_URL`: PostgreSQL connection string
- `EMBEDDING_MODEL_DEVICE`: Device for running models (cuda/cpu)
- `MODEL_CACHE_DIR`: Directory for cached models
- `BATCH_SIZE`: Processing batch size
- `SIMILARITY_THRESHOLD`: Threshold for semantic search results

## Usage Examples

### Process a PDF
```python
from main_pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.process_pdf("path/to/paper.pdf", paper_id="paper_001")
```

### Search for content
```python
results = pipeline.search("rough set theory", limit=5)
```

### Individual component usage:
```python
# Extract paper components
from deepdoc_test import DeepDocProcessor
processor = DeepDocProcessor("paper.pdf")
chunks = processor.extract_from_pdf()

# Generate embeddings
from ensemble_embeddings import EnsembleEmbeddingModel
model = EnsembleEmbeddingModel()
embedding = model.embed_by_type("Minimize transportation cost", chunk_type="formula")

# Vector database operations
from vector_store import VectorStore
store = VectorStore()
store.insert_batch(embeddings_list)
```

## Development Conventions

- The project uses ensemble embeddings with weighted combination to leverage different models' strengths
- LaTeX formulas are processed separately with context preservation
- Each paper chunk includes metadata like page number and content type
- The system normalizes embeddings for cosine similarity search
- Uses logging with loguru for detailed pipeline tracking

## Project Structure

- `main_pipeline.py`: Main orchestration
- `deepdoc_test.py`: PDF processing and extraction
- `ensemble_embeddings.py`: Embedding model ensemble
- `vector_store.py`: Database interface
- `requirements.txt`: Dependencies
- `docker-compose.yml`: Container configuration
- `model_cache/`: Downloaded models (~5GB)
- `pipeline.log`: Execution logs
- `SETUP.md`: Detailed setup guide
- `QUICKSTART.md`: Quick start guide

## Performance Metrics

- PDF Extraction (20 pages): ~2s
- Embedding 100 chunks: ~5s (20 chunks/s)
- Vector DB Insert (100): ~1.5s (67 insert/s)
- Semantic Search (1M vectors): ~50ms
- Total resource usage: ~10GB memory, 55% CPU

## Supported Content Types

- Scientific papers with mathematical formulas
- LaTeX equation extraction
- Table structures with markdown conversion
- Text content with semantic preservation
- Multiple scientific domains (Mathematics, CS, Physics, Biology, Economics)

This comprehensive RAG system is optimized for macOS (with M4 Max support) and handles scientific papers with complex mathematical content efficiently.