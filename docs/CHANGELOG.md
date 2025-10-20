# Changelog

## 2025-01-19 - Recursive Directory Scanning

### ✨ New Features

#### Recursive PDF Directory Scanning
**Feature:** `process_large_dataset.py` now supports recursive directory scanning
**Benefits:**
- Automatically finds PDFs in all subdirectories
- Perfect for nested folder structures (e.g., `/papers/2024/january/`, `/papers/2024/february/`)
- Intelligent folder structure display (shows number of directories processed)
- User-friendly prompt with Turkish language support

**Usage:**
```bash
python process_large_dataset.py

# Script will ask:
# 1. PDF klasörü yolu? → /path/to/your/papers
# 2. Alt klasörlerdeki PDF'ler de işlensin mi? → evet (default)
# 3. Index oluşturulsun mu? → evet/hayır
# 4. Index lists sayısı? → 2000
```

**Implementation:**
- Added `recursive: bool = True` parameter to `process_pdf_directory()`
- Uses `os.walk()` for recursive scanning vs `os.listdir()` for top-level only
- Shows relative paths during processing for better user experience
- Displays directory count in summary (e.g., "Found 450 PDFs in 23 directories")

**Example Output:**
```
Klasör Tarama Ayarları
----------------------------------------------------------------------

Alt klasörlerdeki PDF'ler de işlensin mi?
  evet: Tüm alt klasörleri tara (önerilen - derin klasör yapıları için)
  hayır: Sadece ana klasördeki PDF'leri işle

[evet/hayır] veya Enter (varsayılan: evet): evet

🔍 Alt klasörler taranıyor...

✓ 450 PDF dosyası bulundu
  (23 farklı klasörde)
```

---

## 2025-01-18 - Search & Performance Fixes

### 🐛 Fixed Issues

#### 1. Search Returning Zero Results
**Problem:** Semantic search was returning 0 results even with valid queries
**Root Cause:** IVFFlat vector index causing "low recall" with small datasets (<100 embeddings)
**Solution:**
- Modified `_create_vector_index()` to skip index creation for datasets < 100 embeddings
- Sequential scan is fast enough for small datasets and provides perfect recall
- Index will be automatically created when dataset grows > 100 embeddings

#### 2. Search Threshold Too High
**Problem:** Default threshold (0.3-0.5) was filtering out valid results
**Root Cause:** Ensemble embeddings produce similarity scores in 0.70-0.85 range for good matches
**Solution:**
- Lowered default threshold to 0.5 (was 0.6-0.3)
- Added detailed documentation about typical similarity ranges
- Added debug logging to show actual similarity scores

### 📊 Performance Characteristics

**Current Setup (36 embeddings):**
- Search latency: 1-5 ms
- Index: Disabled (sequential scan)
- Typical similarity: 0.71-0.83 for good matches

**Expected Scaling:**
- 100-1000 embeddings: Simple index, ~5-10 ms
- 1000+ embeddings: IVFFlat index, ~10-50 ms
- 1M+ embeddings: Optimized IVFFlat, ~50-100 ms

### 🔧 Technical Changes

**Files Modified:**
1. `vector_store.py`:
   - Updated `_create_vector_index()` with smart index creation logic
   - Added embedding count check before index creation
   - Improved debug logging for search operations

2. `main_pipeline.py`:
   - Lowered default search threshold from 0.3 to 0.5
   - Added debug logging for query embedding details
   - Enhanced search result reporting

3. `ensemble_embeddings.py`:
   - No changes (working correctly)
   - Embedding dimension: 1024
   - Normalization: L2 norm

### ✅ Test Results

```
Query: "What is rough set theory?"
✓ 3 results found
  - Similarity: 0.7382 (73.8%)
  - Similarity: 0.7382 (73.8%)
  - Similarity: 0.7212 (72.1%)

Query: "Minimize transportation cost"
✓ 3 results found
  - Similarity: 0.7790 (77.9%)
  - Similarity: 0.7790 (77.9%)
  - Similarity: 0.7545 (75.4%)

Query: "Tables about costs" (filtered to tables only)
✓ 3 results found
  - Similarity: 0.8333 (83.3%)
  - Similarity: 0.8333 (83.3%)
  - Similarity: 0.7973 (79.7%)
```

### 📝 Notes

**Embedding Size:**
- Dimension: 1024 (after ensemble combination)
- SPECTER2: 768 → padded to 1024
- E5-Large: 1024 (native)
- SciBERT: 768 → padded to 1024
- Final: Weighted average + L2 normalization

**Memory Usage:**
- CPU-based embedding generation (Apple Silicon optimized)
- 36 embeddings × 1024 dim × 4 bytes = ~147 KB
- Model memory: ~2-3 GB (SPECTER2 + E5-Large + SciBERT)
- Total RAM usage: ~4-5 GB (well within 48 GB capacity)

**Index Strategy:**
- < 100 embeddings: No index (sequential scan)
- 100-1000: Simple ivfflat (lists = count/10)
- 1000+: Optimized ivfflat (lists = 100)
- Threshold: 0.5 (captures 0.70-0.85 similarity range)

### 🎯 Next Steps

1. **Scaling:** Add more papers to test index creation at 100+ embeddings
2. **Optimization:** Consider HNSW index for very large datasets (10K+ embeddings)
3. **Features:** Add hybrid search (keyword + semantic)
4. **Monitoring:** Add search latency metrics
