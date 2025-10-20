# RAG System - Linux Deployment Guide

This guide is for deploying the RAG system on Ubuntu 25 (64-bit) for heavy processing tasks.

## Prerequisites

- Ubuntu 25 (64-bit)
- Docker & Docker Compose installed
- UV package manager (auto-installed by setup script)
- At least 16GB RAM
- 50GB+ free disk space
- PDF files on USB drive

**Note:** UV will automatically handle Python version management. No need to pre-install Python!

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd embed_rag
git checkout feature/linux-reprocessing
```

### 2. Run Setup Script

```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

This will:
- ✅ Verify system requirements (Docker, UV)
- ✅ Create necessary directories
- ✅ Set up Python environment with UV (auto-detects or installs Python)
- ✅ Install dependencies (much faster with UV!)
- ✅ Start Docker services (PostgreSQL + Embedding Server)

### 3. Copy PDF Files

```bash
# From USB drive
cp -r /media/usb/pdfs/* ./data/pdfs/

# Or from network
rsync -avz --progress user@remote:/path/to/pdfs/ ./data/pdfs/
```

### 4. Configure Environment

```bash
# Edit .env file
nano .env

# Required settings:
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/embed_rag
EMBEDDING_SERVER_URL=http://localhost:8000
```

### 5. Run Data Quality Analysis

```bash
# Activate UV environment
source .venv/bin/activate

# Or use UV directly (no activation needed!)
uv run python3 analyze_data_quality.py
```

This will show current data quality and identify issues.

### 6. Process PDFs

```bash
# Full reprocessing (1,252 papers, 2-4 hours)
# Option 1: With activation
source .venv/bin/activate
nohup python3 main_pipeline.py > processing.log 2>&1 &

# Option 2: Direct with UV (recommended!)
nohup uv run python3 main_pipeline.py > processing.log 2>&1 &
```

**Progress Monitoring:**
```bash
# In another terminal
tail -f processing.log

# Watch live updates
watch -n 10 'tail -n 20 processing.log'
```

### 7. Validate Results

```bash
# Run quality tests
uv run python3 test_embedding_quality.py

# Check semantic similarity (target: 78%+)
uv run python3 test_multi_query.py
```

### 8. Export Results

```bash
# Create database dump
docker exec embed_rag_pgvector pg_dump -U postgres embed_rag > paper_embeddings_v2.sql

# Create archive
tar -czf results_$(date +%Y%m%d).tar.gz \
    paper_embeddings_v2.sql \
    embedding_quality_report.json \
    pdf_quality_scores.json \
    pipeline.log
```

### 9. Transfer to macOS

```bash
# From Linux machine
scp results_20251020.tar.gz user@macos-machine:/path/to/destination/

# On macOS
cd /path/to/embed_rag
tar -xzf results_20251020.tar.gz
psql $DATABASE_URL < paper_embeddings_v2.sql
```

## UV Package Manager Benefits

Why UV is better than traditional pip/venv:

**Speed:**
```bash
# UV is 10-100x faster than pip
uv pip install torch  # Seconds instead of minutes!
```

**Python Version Management:**
```bash
# Use specific Python version
uv venv --python 3.11
uv venv --python 3.12

# UV auto-downloads Python if not installed!
```

**Lock Files:**
```bash
# Create lock file for reproducible installs
uv pip compile requirements.txt -o requirements.lock

# Install from lock file
uv pip sync requirements.lock
```

**No Activation Needed:**
```bash
# Traditional way
source .venv/bin/activate
python3 script.py

# UV way (no activation!)
uv run python3 script.py
```

**Global Cache:**
- UV caches packages globally
- Installations share dependencies
- Saves disk space and time

## Troubleshooting

### Docker Issues

```bash
# Check services
docker ps

# View logs
docker logs embed_rag_pgvector
docker logs embed_rag_embeddings

# Restart services
docker-compose restart
```

### Memory Issues

If processing fails with OOM errors:

```bash
# Edit main_pipeline.py and reduce batch size
# Line ~50: batch_size=8  # Reduce from 16 to 8
```

### Database Connection Issues

```bash
# Test connection
psql postgresql://postgres:postgres@localhost:5433/embed_rag

# Reset database
docker-compose down -v
docker-compose up -d
```

## Performance Tips

### 1. Use tmpfs for temporary files (if enough RAM)

```bash
sudo mount -t tmpfs -o size=8G tmpfs /tmp/embed_rag
export TMPDIR=/tmp/embed_rag
```

### 2. Optimize PostgreSQL

Edit `docker-compose.yml`:

```yaml
environment:
  - POSTGRES_SHARED_BUFFERS=4GB
  - POSTGRES_WORK_MEM=256MB
  - POSTGRES_MAINTENANCE_WORK_MEM=1GB
```

### 3. Monitor resources

```bash
# CPU and memory
htop

# Disk I/O
iotop

# GPU (if available)
nvidia-smi -l 1
```

## What Gets Processed

The reprocessing pipeline fixes:

1. ✅ **Text Chunking**: Reduces chunks from 2000+ to max 1500 chars
2. ✅ **OCR Errors**: Cleans garbled characters in formulas
3. ✅ **Table Descriptions**: Generates detailed table summaries
4. ✅ **Embeddings**: Regenerates with ensemble model (E5 + SPECTER2 + SciBERT)

**Expected Results:**
- Semantic Similarity: 66.7% → 78%+
- Data Quality Score: 57.8 → 85%+
- Processing Time: ~2-4 hours for 1,252 papers

## Next Steps After Processing

1. ✅ Transfer results to macOS
2. ✅ Import database
3. ✅ Run final tests
4. ✅ Deploy to production

## Support

For issues, check:
- `pipeline.log` - Processing logs
- `docker logs` - Service logs
- GitHub Issues - Community support
