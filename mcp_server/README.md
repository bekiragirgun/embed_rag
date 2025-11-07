# MCP Context Manager

**Tüm CLI tabanlı LLM araçlarında kullanabileceğin proje bazlı context yönetim sistemi.**

## 🎯 Özellikler

- ✅ **Proje Bazlı Saklama** - Her proje için ayrı SQLite veritabanı
- ✅ **Otomatik Session Yönetimi** - Token bitince session otomatik temizlenir
- ✅ **Local Storage** - Tüm konuşmalar `~/.mcp_contexts` altında
- ✅ **Git Repo Detection** - Proje otomatik tespit edilir
- ✅ **STDIO Transport** - Claude Code, Cursor, Continue.dev gibi araçlarla uyumlu
- ✅ **Zero Configuration** - Setup bir kere, her projede otomatik çalışır

## 🚀 Kurulum

### 1. Bağımlılıkları Yükle

```bash
cd mcp_server
./setup.sh
```

Setup script:
- Python virtual environment oluşturur
- Bağımlılıkları yükler
- `~/.mcp_contexts` dizinini oluşturur
- Config dosyasını hazırlar

### 2. Claude Desktop'a Ekle

**macOS:** `~/.config/claude/config.json`
**Linux:** `~/.config/claude/config.json`
**Windows:** `%APPDATA%\Claude\config.json`

```json
{
  "mcpServers": {
    "context-manager": {
      "command": "/FULL/PATH/TO/mcp_server/venv/bin/python",
      "args": [
        "/FULL/PATH/TO/mcp_server/server.py"
      ],
      "env": {
        "CONTEXT_STORAGE_DIR": "~/.mcp_contexts",
        "PYTHONPATH": "/FULL/PATH/TO/mcp_server"
      }
    }
  }
}
```

> 💡 Setup script `claude-config.json` dosyasını otomatik oluşturur, oradan kopyalayabilirsin.

### 3. Claude Desktop'ı Yeniden Başlat

MCP server aktif olduğunda Claude'da yeni tool'lar görünecek:
- `store_conversation`
- `get_context`
- `search_context`
- `clear_session`
- `list_projects`
- `get_stats`
- `start_session`

## 📖 Kullanım

### Otomatik Kullanım (Önerilen)

Claude ile çalışırken otomatik olarak:
1. Proje tespit edilir (git repo veya klasör adı)
2. Session başlatılır
3. Konuşmalar saklanır
4. Token limiti aşıldığında session temizlenir

**Hiçbir şey yapman gerekmez!** Claude otomatik olarak context'i yönetir.

### Manuel Tool Kullanımı

#### 1. Konuşma Sakla

```typescript
// Claude otomatik yapar, ama manuel de kullanabilirsin
store_conversation({
  role: "user",
  content: "How do I implement authentication?",
  token_count: 10
})
```

#### 2. Context Al

```typescript
// Mevcut session'ın tüm konuşmalarını al
get_context({
  limit: 50  // Son 50 mesaj
})
```

#### 3. Context'te Ara

```typescript
// Geçmiş konuşmalarda ara
search_context({
  query: "authentication",
  limit: 10
})
```

#### 4. Session Temizle

```typescript
// Token bitti, yeni session başla
clear_session({})

// Sonraki mesaj yeni session'da başlar
```

#### 5. Proje İstatistikleri

```typescript
// Mevcut session istatistikleri
get_stats({
  type: "session"
})

// Proje istatistikleri
get_stats({
  type: "project"
})
```

#### 6. Projeleri Listele

```typescript
// Tüm projeleri ve boyutlarını gör
list_projects({})
```

## 🗂️ Veri Yapısı

```
~/.mcp_contexts/
├── 3f4a1b2c_my_project.db      # Her proje için ayrı DB
├── a5d3e9f1_another_project.db
├── server.log                   # Server logs
└── ...

Her DB içinde:
- conversations table: Tüm mesajlar
- sessions table: Session metadata
```

### Conversations Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment ID |
| session_id | TEXT | Session identifier |
| timestamp | DATETIME | Message timestamp |
| role | TEXT | user/assistant/system |
| content | TEXT | Message content |
| metadata | JSON | Extra metadata |
| token_count | INTEGER | Token count |

### Sessions Table

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | Primary key |
| project_id | TEXT | Project identifier |
| created_at | DATETIME | Session start time |
| last_active | DATETIME | Last message time |
| total_tokens | INTEGER | Total tokens used |
| is_active | BOOLEAN | Active session flag |

## 🎨 Kullanım Senaryoları

### Senaryo 1: Yeni Proje Başlat

```bash
cd ~/Projects/my_new_app
claude code
```

Claude otomatik olarak:
1. Projeyi tespit eder (`my_new_app`)
2. Yeni session başlatır
3. Konuşmaları saklar

### Senaryo 2: Token Limiti Doldu

Claude şöyle söyler:
> "Token limit reached. Starting new session..."

```typescript
// Claude otomatik yapar:
clear_session({})

// Yeni session başlar, eski context kayıtlı
```

### Senaryo 3: Geçmiş Konuşmaları Ara

```bash
# Önceki projede authentication hakkında ne konuşmuştuk?
```

Claude:
```typescript
search_context({
  query: "authentication",
  limit: 5
})
```

### Senaryo 4: Proje Değiştir

```bash
cd ~/Projects/another_project
claude code
```

Claude otomatik olarak `another_project` context'ine geçer.

## 🔧 Yapılandırma

### Environment Variables

```bash
# Contexts dizini (default: ~/.mcp_contexts)
export CONTEXT_STORAGE_DIR="~/my_contexts"

# Python path (setup.sh otomatik set eder)
export PYTHONPATH="/path/to/mcp_server"
```

### Custom Storage Directory

```json
{
  "mcpServers": {
    "context-manager": {
      "env": {
        "CONTEXT_STORAGE_DIR": "/custom/path/contexts"
      }
    }
  }
}
```

## 🐛 Troubleshooting

### Server çalışmıyor

1. **Log kontrol et:**
   ```bash
   tail -f ~/.mcp_contexts/server.log
   ```

2. **Manuel test:**
   ```bash
   cd mcp_server
   source venv/bin/activate
   python server.py
   ```

3. **Bağımlılıkları kontrol et:**
   ```bash
   pip list | grep mcp
   ```

### Context saklanmıyor

1. **Dizin var mı:**
   ```bash
   ls -la ~/.mcp_contexts
   ```

2. **Yazma izni var mı:**
   ```bash
   chmod 755 ~/.mcp_contexts
   ```

3. **Proje tespit ediliyor mu:**
   ```python
   from context_manager import ContextManager
   cm = ContextManager()
   print(cm.detect_project())
   ```

### Claude tool'ları göremiyor

1. **Config doğru mu:**
   ```bash
   cat ~/.config/claude/config.json
   ```

2. **Path'ler mutlak mı:**
   - ❌ `./server.py`
   - ✅ `/full/path/to/server.py`

3. **Claude restart edildi mi:**
   - Quit Claude completely
   - Reopen

## 🧪 Test

### Manuel Test

```bash
cd mcp_server
source venv/bin/activate

# Test storage
python storage.py

# Test context manager
python context_manager.py
```

### MCP Server Test

```bash
# Test server STDIO
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python server.py
```

## 📊 İstatistikler ve İzleme

### Session İstatistikleri

```typescript
get_stats({ type: "session" })

// Sonuç:
{
  "session_id": "my_project_a3f8b1c2",
  "created_at": "2025-01-18 10:30:00",
  "last_active": "2025-01-18 11:45:00",
  "total_tokens": 15420,
  "is_active": true,
  "message_count": 42
}
```

### Proje İstatistikleri

```typescript
get_stats({ type: "project" })

// Sonuç:
{
  "project_id": "my_project",
  "session_count": 5,
  "message_count": 234,
  "total_tokens": 89340,
  "active_sessions": 1
}
```

### Tüm Projeler

```typescript
list_projects({})

// Sonuç:
{
  "projects": [
    {
      "project_id": "my_project",
      "db_file": "~/.mcp_contexts/a3f8b1c2_my_project.db",
      "size_mb": 2.4
    },
    {
      "project_id": "another_project",
      "db_file": "~/.mcp_contexts/f9e3a7d1_another_project.db",
      "size_mb": 1.8
    }
  ],
  "count": 2
}
```

## 🔒 Güvenlik ve Gizlilik

- ✅ **Tüm veriler local** - Hiçbir şey buluta gönderilmez
- ✅ **SQLite encryption** - Gerekirse DB şifrelenebilir
- ✅ **No external dependencies** - Sadece yerel dosya sistemi
- ✅ **Per-project isolation** - Projeler birbirinden izole

### DB Şifreleme (İsteğe Bağlı)

```bash
pip install sqlcipher3
```

`storage.py` içinde:
```python
from sqlcipher3 import dbapi2 as sqlite3

conn = sqlite3.connect(db_path)
conn.execute("PRAGMA key='your-secret-key'")
```

## 🚀 İleri Seviye Kullanım

### Custom Session ID

```typescript
start_session({
  session_id: "my_custom_session",
  project_id: "my_project"
})
```

### Cross-Project Search

```typescript
// Tüm projelerde ara (manuel script gerekli)
const projects = list_projects()
for (const project of projects) {
  search_context({
    project_id: project.project_id,
    query: "authentication"
  })
}
```

### Token Threshold Warning

```typescript
// Session stats'ı kontrol et
const stats = get_stats({ type: "session" })

if (stats.total_tokens > 180000) {
  console.log("⚠️ Token limit yaklaşıyor, session temizle!")
  clear_session({})
}
```

## 📚 API Referansı

### store_conversation

Konuşma mesajı saklar.

**Parameters:**
- `role` (string, required): "user" | "assistant" | "system"
- `content` (string, required): Mesaj içeriği
- `token_count` (integer, optional): Token sayısı
- `metadata` (object, optional): Extra metadata
- `project_id` (string, optional): Proje ID (otomatik tespit)
- `cwd` (string, optional): Working directory

**Returns:**
```json
{
  "status": "success",
  "project_id": "my_project",
  "session_id": "my_project_a3f8b1c2"
}
```

### get_context

Mevcut session'ın context'ini getirir.

**Parameters:**
- `limit` (integer, optional): Max mesaj sayısı
- `session_id` (string, optional): Özel session
- `project_id` (string, optional): Proje ID
- `cwd` (string, optional): Working directory

**Returns:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-01-18 10:30:00",
      "token_count": 5
    }
  ],
  "count": 1,
  "project_id": "my_project",
  "session_id": "my_project_a3f8b1c2"
}
```

### search_context

Context'te arama yapar.

**Parameters:**
- `query` (string, required): Arama sorgusu
- `session_id` (string, optional): Session filtresi
- `limit` (integer, optional): Max sonuç (default: 10)
- `project_id` (string, optional): Proje ID
- `cwd` (string, optional): Working directory

**Returns:**
```json
{
  "results": [
    {
      "role": "assistant",
      "content": "Authentication can be implemented using...",
      "timestamp": "2025-01-18 10:35:00",
      "session_id": "my_project_a3f8b1c2"
    }
  ],
  "count": 1,
  "query": "authentication"
}
```

### clear_session

Mevcut session'ı sonlandırır.

**Parameters:**
- `session_id` (string, optional): Session ID
- `project_id` (string, optional): Proje ID
- `cwd` (string, optional): Working directory

**Returns:**
```json
{
  "status": "success",
  "message": "Session my_project_a3f8b1c2 ended",
  "session_id": "my_project_a3f8b1c2"
}
```

### list_projects

Tüm projeleri listeler.

**Parameters:** None

**Returns:**
```json
{
  "projects": [
    {
      "project_id": "my_project",
      "db_file": "~/.mcp_contexts/a3f8b1c2_my_project.db",
      "size_mb": 2.4
    }
  ],
  "count": 1
}
```

### get_stats

İstatistikleri getirir.

**Parameters:**
- `type` (string, optional): "session" | "project" (default: "session")
- `session_id` (string, optional): Session ID (session stats için)
- `project_id` (string, optional): Proje ID
- `cwd` (string, optional): Working directory

**Returns:**

**Session stats:**
```json
{
  "session_id": "my_project_a3f8b1c2",
  "created_at": "2025-01-18 10:30:00",
  "total_tokens": 15420,
  "is_active": true,
  "message_count": 42
}
```

**Project stats:**
```json
{
  "project_id": "my_project",
  "session_count": 5,
  "message_count": 234,
  "total_tokens": 89340,
  "active_sessions": 1
}
```

### start_session

Yeni session başlatır.

**Parameters:**
- `project_id` (string, optional): Proje ID (otomatik tespit)
- `session_id` (string, optional): Custom session ID (otomatik oluşturulur)
- `cwd` (string, optional): Working directory

**Returns:**
```json
{
  "status": "success",
  "session_id": "my_project_a3f8b1c2",
  "project_id": "my_project"
}
```

## 🤝 Diğer CLI Araçlarla Kullanım

### Cursor

`~/.cursor/config.json`:
```json
{
  "mcpServers": {
    "context-manager": {
      "command": "/path/to/mcp_server/venv/bin/python",
      "args": ["/path/to/mcp_server/server.py"]
    }
  }
}
```

### Continue.dev

`.continue/config.json`:
```json
{
  "mcpServers": [
    {
      "name": "context-manager",
      "command": "/path/to/mcp_server/venv/bin/python",
      "args": ["/path/to/mcp_server/server.py"]
    }
  ]
}
```

### Qwen CLI

Qwen'in MCP desteği eklendiğinde aynı config kullanılabilir.

## 📝 Notlar

- Her proje otomatik olarak tespit edilir (git repo veya klasör adı)
- Session'lar otomatik oluşturulur ve yönetilir
- Token limiti aşıldığında session otomatik temizlenir
- Tüm veriler local'de saklanır (`~/.mcp_contexts`)
- SQLite kullanıldığı için hafif ve hızlı

## 📄 Lisans

MIT License

## 🙋 Destek

Issues için:
1. Log dosyasını kontrol et: `~/.mcp_contexts/server.log`
2. Test script'lerini çalıştır: `python storage.py`
3. Config'i doğrula: `cat ~/.config/claude/config.json`

## 🎉 Özet

1. **Kurulum:** `./setup.sh` → Claude config'e ekle → Restart
2. **Kullanım:** Claude ile normal çalış, otomatik saklanır
3. **Token bitti:** `clear_session()` → Yeni session başlar
4. **Arama:** `search_context()` → Geçmiş konuşmalarda ara
5. **Stats:** `get_stats()` → İstatistikleri gör

**Artık her projende konuşmalarını local'de saklayabilir, token bittiğinde temizleyebilirsin!** 🚀
