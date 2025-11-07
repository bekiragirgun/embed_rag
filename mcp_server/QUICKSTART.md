# MCP Context Manager - Hızlı Başlangıç

## 🚀 5 Dakikada Kurulum

### 1. Setup Çalıştır

```bash
cd mcp_server
./setup.sh
```

Bu komut:
- ✅ Virtual environment oluşturur
- ✅ Bağımlılıkları yükler
- ✅ `~/.mcp_contexts` dizinini oluşturur
- ✅ Config dosyasını hazırlar

### 2. Claude Config'e Ekle

**macOS/Linux:**
```bash
# Config dosyasını aç
nano ~/.config/claude/config.json

# Veya setup script'in oluşturduğunu kullan:
cat claude-config.json
```

**Config içeriği:**
```json
{
  "mcpServers": {
    "context-manager": {
      "command": "/FULL/PATH/TO/mcp_server/venv/bin/python",
      "args": ["/FULL/PATH/TO/mcp_server/server.py"],
      "env": {
        "CONTEXT_STORAGE_DIR": "~/.mcp_contexts",
        "PYTHONPATH": "/FULL/PATH/TO/mcp_server"
      }
    }
  }
}
```

> ⚠️ **Önemli:** Path'leri tam path ile değiştir! `setup.sh` bunu otomatik yapar.

### 3. Claude'ı Restart Et

Tamamen kapat ve yeniden aç.

### 4. Test Et

Claude'da şunu söyle:
> "Can you list available MCP tools?"

Görmelisin:
- `store_conversation`
- `get_context`
- `search_context`
- `clear_session`
- `list_projects`
- `get_stats`
- `start_session`

## ✅ Kullanıma Hazır!

Artık Claude ile çalışırken:
- ✅ Konuşmalar otomatik saklanır
- ✅ Her proje için ayrı context
- ✅ Token bitti? Session otomatik temizlenir
- ✅ Geçmiş konuşmalarda arama yapabilirsin

## 📝 İlk Kullanım

### Otomatik (Önerilen)

Sadece normal çalış, Claude otomatik yönetir:

```bash
cd ~/Projects/my_app
claude code
```

Claude:
1. Projeyi tespit eder (`my_app`)
2. Session başlatır
3. Konuşmaları saklar

### Manuel Test

Claude'a şunu söyle:
> "Store this conversation and show me the stats"

Claude otomatik olarak:
```typescript
// 1. Konuşmayı saklar
store_conversation({
  role: "user",
  content: "Store this conversation and show me the stats"
})

// 2. Stats'ı gösterir
get_stats({ type: "session" })
```

## 🔍 Yaygın Komutlar

### Context Al

Claude'a:
> "Show me our conversation history"

### Arama Yap

Claude'a:
> "Search our previous conversations about authentication"

### Session Temizle

Claude'a:
> "Clear the current session, I want to start fresh"

### İstatistikler

Claude'a:
> "Show me statistics for this project"

## 🐛 Sorun Giderme

### 1. Tool'lar Görünmüyor

```bash
# Config doğru mu?
cat ~/.config/claude/config.json

# Path'ler tam mı? (❌ ./server.py ✅ /full/path/server.py)
```

### 2. Server Çalışmıyor

```bash
# Log kontrol
tail -f ~/.mcp_contexts/server.log

# Manuel test
cd mcp_server
source venv/bin/activate
python server.py
```

### 3. Context Saklanmıyor

```bash
# Dizin var mı?
ls -la ~/.mcp_contexts

# İzinler doğru mu?
chmod 755 ~/.mcp_contexts
```

## 📚 Daha Fazla

Detaylı dokümantasyon için: [README.md](README.md)

## 🎉 Başarılı Kurulum?

Şunu dene:

1. Yeni bir projede Claude aç
2. Claude'a: "Store this message and show stats"
3. Claude otomatik olarak context yönetir
4. `~/.mcp_contexts` altında DB dosyalarını gör

**Tebrikler! Artık tüm projelerinde context yönetimi aktif! 🚀**
