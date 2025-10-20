# EmbedRAG - Doküman İndeksi

## 📚 Başlangıç Rehberleri

### 🇹🇷 Türkçe
- **[QUICKSTART_TR.md](QUICKSTART_TR.md)** - 5 dakikada başlayın (önerilen)
  - Docker kurulumu
  - İlk PDF işleme
  - Arama testi
  - Sorun giderme

### 🇬🇧 English
- **[../README.md](../README.md)** - Project overview and quick start
- **[SETUP.md](SETUP.md)** - Detailed installation guide

## 🔧 Yapılandırma Rehberleri

### Index Stratejisi
- **[INDEXING.md](INDEXING.md)** - Vector index yönetimi
  - Küçük dataset (< 100 PDF): Index gerekmez
  - Orta dataset (100-1000 PDF): Opsiyonel
  - Büyük dataset (40GB+): Gerekli
  - Performans karşılaştırmaları
  - Index oluşturma zamanlaması

### Arama Ayarları
- **[SEARCH_GUIDE.md](SEARCH_GUIDE.md)** - Semantic arama optimizasyonu
  - Similarity threshold ayarlama
  - Score yorumlama (0.71-0.83 iyi)
  - Chunk type filtreleme
  - Common issues & solutions

## 📋 Referans Dokümanları

### Değişiklikler
- **[CHANGELOG.md](CHANGELOG.md)** - Son güncellemeler (2025-01-18)
  - Search fix (0 result → working)
  - Optional indexing (40GB datasets)
  - Interactive Turkish script
  - Threshold optimization

## 🎯 Kullanım Senaryolarına Göre

### Yeni Başlayanlar
1. [QUICKSTART_TR.md](QUICKSTART_TR.md) - İlk kurulum
2. [../README.md](../README.md) - Feature overview
3. Test: 5-10 PDF ile başlayın

### 40GB+ Dataset İşleme
1. [INDEXING.md](INDEXING.md) - Index stratejisi
2. [QUICKSTART_TR.md](QUICKSTART_TR.md) - Interactive script
3. Index: `process_large_dataset.py` kullanın

### Arama Optimizasyonu
1. [SEARCH_GUIDE.md](SEARCH_GUIDE.md) - Threshold tuning
2. [CHANGELOG.md](CHANGELOG.md) - Recent fixes
3. Test: Different thresholds (0.0-0.8)

### Sorun Giderme
1. [SETUP.md](SETUP.md) - Troubleshooting section
2. [QUICKSTART_TR.md](QUICKSTART_TR.md) - Common errors
3. [SEARCH_GUIDE.md](SEARCH_GUIDE.md) - Search issues

## 📖 Doküman Türlerine Göre

### Hızlı Referans (< 5 dakika)
- ✅ [QUICKSTART_TR.md](QUICKSTART_TR.md)
- ✅ [CHANGELOG.md](CHANGELOG.md)

### Detaylı Rehber (15-30 dakika)
- 📚 [SETUP.md](SETUP.md)
- 📚 [INDEXING.md](INDEXING.md)
- 📚 [SEARCH_GUIDE.md](SEARCH_GUIDE.md)

### Kapsamlı Doküman (> 30 dakika)
- 📖 [../README.md](../README.md)

## 🔗 Hızlı Linkler

| Konu | Link |
|------|------|
| 5 dakikada başla | [QUICKSTART_TR.md](QUICKSTART_TR.md) |
| 40GB dataset | [INDEXING.md](INDEXING.md) |
| Arama düşük | [SEARCH_GUIDE.md](SEARCH_GUIDE.md#understanding-similarity-scores) |
| Index oluştur | [INDEXING.md](INDEXING.md#large-dataset-40gb-millions-of-vectors) |
| Sorun giderme | [SETUP.md](SETUP.md#-troubleshooting) |
| Son güncellemeler | [CHANGELOG.md](CHANGELOG.md) |

## 💡 SSS (Sık Sorulan Sorular)

### "PDF'ler nerede gösterilir?"
→ [QUICKSTART_TR.md](QUICKSTART_TR.md#4-pdfleri-işleyin)

### "Index ne zaman oluşturulmalı?"
→ [INDEXING.md](INDEXING.md#when-to-use-index)

### "Arama sonuç vermiyor?"
→ [SEARCH_GUIDE.md](SEARCH_GUIDE.md#common-issues--solutions)

### "40GB dokümanım var, nasıl işlerim?"
→ [INDEXING.md](INDEXING.md#large-dataset-40gb-millions-of-vectors)

---

**Son güncelleme:** 2025-01-18
**Dil:** Türkçe (🇹🇷) / English (🇬🇧)
