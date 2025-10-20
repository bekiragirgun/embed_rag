# Hızlı Başlangıç Rehberi - 5 Dakikada Başlayın

## 📋 Gereksinimler

- macOS (Apple Silicon - M1/M2/M3/M4)
- Docker Desktop
- Python 3.10+
- 48GB RAM (önerilen, 16GB minimum)

## 🚀 5 Adımda Başlangıç

### 1. Docker'ı Başlatın

```bash
cd /Users/bekiragirgun/Projects/embed_rag
./docker-up.sh
```

**Çıktı:**
```
✓ pgvector container running on port 5433
✓ Database: embed_rag created
✓ Extension: pgvector enabled
```

### 2. Virtual Environment'ı Aktifleştirin

```bash
source venv/bin/activate
```

### 3. Database Bağlantısını Ayarlayın

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/embed_rag"
```

**Not:** Her yeni terminal oturumunda bu komutu tekrar çalıştırmalısınız.

### 4. PDF'leri İşleyin

```bash
python process_large_dataset.py
```

**Script size soracak:**

```
PDF klasörü yolu: /Users/bekiragirgun/Downloads/Documents/aktarılan

Klasör Tarama Ayarları
----------------------------------------------------------------------

Alt klasörlerdeki PDF'ler de işlensin mi?
  evet: Tüm alt klasörleri tara (önerilen - derin klasör yapıları için)
  hayır: Sadece ana klasördeki PDF'leri işle

[evet/hayır] veya Enter (varsayılan: evet): evet

🔍 Alt klasörler taranıyor...

✓ 150 PDF dosyası bulundu
  (12 farklı klasörde)

Index Ayarları
----------------------------------------------------------------------

Dataset boyutu: 150 PDF
  → Orta dataset: Index opsiyonel

Tüm veriler yüklendikten SONRA index oluşturulsun mu? (önerilen: hayır)
[evet/hayır] veya Enter (varsayılan): hayır

ÖZET
======================================================================
Klasör:        /Users/bekiragirgun/Downloads/Documents/aktarılan
PDF sayısı:    150
Alt klasörler: Evet (recursive)
Index:         Hayır
======================================================================

Devam edilsin mi? [evet/hayır]: evet
```

### 5. İşleme Başladı!

```
======================================================================
LARGE DATASET PROCESSING - 40GB+ PDF Collection
======================================================================

Found 150 PDF files
Directory: /Users/bekiragirgun/Downloads/Documents/aktarılan

Initializing pipeline (index creation: DISABLED)
  → Fast inserts without index overhead
  → Index will be created AFTER all data is loaded

[1/150] Processing: paper1.pdf
  ✓ Success: 45 chunks

[2/150] Processing: paper2.pdf
  ✓ Success: 52 chunks

...
```

## 📊 Beklenen Süreler (M4 Max, 48GB RAM)

| İşlem | Süre |
|-------|------|
| Docker başlatma | ~10 saniye |
| Pipeline init | ~30 saniye |
| PDF başına (20 sayfa) | ~8 saniye |
| 100 PDF | ~13 dakika |
| 1000 PDF | ~2.2 saat |
| Index oluşturma (1M vektör) | ~15 dakika |

## 🔍 İşleme Tamamlandıktan Sonra Arama

### Python ile:

```python
from main_pipeline import RAGPipeline

pipeline = RAGPipeline(create_index=False)

# Arama
results = pipeline.search("rough set theory", limit=5)

# Sonuçları göster
for i, result in enumerate(results, 1):
    print(f"{i}. [{result['chunk_type']}] {result['similarity']:.2%}")
    print(f"   {result['content'][:100]}...")
```

### Örnek Çıktı:

```
1. [text] 73.82%
   Rough set theory is a mathematical approach to data analysis introduced by Pawlak in 1982...

2. [formula] 71.45%
   Lower approximation: R_(X) = {x ∈ U | [x]_R ⊆ X}

3. [text] 69.21%
   The key concept in rough set theory is the equivalence class...
```

## ⚙️ Ayarlar

### Küçük Dataset (< 100 PDF):
- Index: **Hayır** (sequential scan daha hızlı)
- Beklenen işlem süresi: 10-15 dakika

### Orta Dataset (100-1000 PDF):
- Index: **Opsiyonel**
- Beklenen işlem süresi: 1-3 saat

### Büyük Dataset (1000+ PDF):
- Index: **Evet** (16x hızlandırma)
- Index lists: 2000 (varsayılan)
- Beklenen işlem süresi: 3-10 saat + 30 dakika index

## 🐛 Sorun Giderme

### "Database connection failed"

**Çözüm:**
```bash
# Docker çalışıyor mu kontrol et
docker ps | grep pgvector

# Eğer çalışmıyorsa:
./docker-up.sh

# DATABASE_URL'yi tekrar export et
export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/embed_rag"
```

### "No PDF files found"

**Çözüm:**
- Klasör yolunu tırnak işareti olmadan yazın
- Klasör yolunu Finder'dan kopyalayın: sağ tık → "Copy as Pathname"

### "Out of memory"

**Çözüm:**
- Batch size'ı küçültün (`.env` dosyasında `BATCH_SIZE=16`)
- GPU kullanımını kapatın (`.env` dosyasında `EMBEDDING_MODEL_DEVICE=cpu`)

## 📚 Daha Fazla Bilgi

- **Detaylı kurulum:** [SETUP.md](SETUP.md)
- **Index stratejisi:** [INDEXING.md](INDEXING.md)
- **Arama ayarları:** [SEARCH_GUIDE.md](SEARCH_GUIDE.md)
- **Son güncellemeler:** [CHANGELOG.md](CHANGELOG.md)

## 💡 İpuçları

1. **İlk kez kullanıyorsanız:** Küçük bir klasörle (5-10 PDF) test edin
2. **40GB dataset için:** Index'i mutlaka oluşturun, arama 16x hızlanır
3. **Her yeni terminal:** `export DATABASE_URL=...` komutunu tekrar çalıştırın
4. **Progress takibi:** `large_dataset_processing.log` dosyasını kontrol edin

## 🎯 Sonraki Adımlar

1. ✅ Pipeline'ı çalıştırın
2. ✅ İlk aramalarınızı yapın
3. ✅ Similarity threshold'u ayarlayın ([SEARCH_GUIDE.md](SEARCH_GUIDE.md))
4. ✅ API geliştirin (opsiyonel)
5. ✅ Production'a deploy edin (opsiyonel)

---

**Created:** 2025-01-18
**Platform:** macOS (Apple Silicon)
**Hardware:** M4 Max, 48GB RAM
