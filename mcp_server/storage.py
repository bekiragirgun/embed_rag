"""
Storage layer for conversation context management.
Supports project-based SQLite databases with automatic session tracking.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib


class ContextStorage:
    """
    Manages conversation storage in SQLite.
    Each project gets its own database file.
    """

    def __init__(self, storage_dir: str = ".contexts"):
        """
        Initialize storage manager.

        Args:
            storage_dir: Directory to store all project databases
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_db_path(self, project_id: str) -> Path:
        """Get database path for a project."""
        # Sanitize project_id for filename
        safe_id = hashlib.md5(project_id.encode()).hexdigest()[:8]
        return self.storage_dir / f"{safe_id}_{project_id.replace('/', '_')}.db"

    def _init_db(self, conn: sqlite3.Connection):
        """Initialize database schema."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                token_count INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_tokens INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON conversations(session_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON conversations(timestamp)
        """)

        conn.commit()

    def get_connection(self, project_id: str) -> sqlite3.Connection:
        """Get database connection for a project."""
        db_path = self._get_db_path(project_id)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Initialize if new database
        self._init_db(conn)

        return conn

    def store_message(
        self,
        project_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: int = 0
    ):
        """
        Store a conversation message.

        Args:
            project_id: Unique project identifier
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Additional metadata
            token_count: Token count for this message
        """
        conn = self.get_connection(project_id)

        try:
            # Ensure session exists
            conn.execute("""
                INSERT OR IGNORE INTO sessions (session_id, project_id)
                VALUES (?, ?)
            """, (session_id, project_id))

            # Store message
            conn.execute("""
                INSERT INTO conversations
                (session_id, role, content, metadata, token_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                json.dumps(metadata) if metadata else None,
                token_count
            ))

            # Update session stats
            conn.execute("""
                UPDATE sessions
                SET last_active = CURRENT_TIMESTAMP,
                    total_tokens = total_tokens + ?
                WHERE session_id = ?
            """, (token_count, session_id))

            conn.commit()

        finally:
            conn.close()

    def get_session_context(
        self,
        project_id: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for current session.

        Args:
            project_id: Project identifier
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of messages in chronological order
        """
        conn = self.get_connection(project_id)

        try:
            query = """
                SELECT role, content, metadata, timestamp, token_count
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query, (session_id,))

            messages = []
            for row in cursor.fetchall():
                msg = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "token_count": row["token_count"]
                }
                if row["metadata"]:
                    msg["metadata"] = json.loads(row["metadata"])
                messages.append(msg)

            return messages

        finally:
            conn.close()

    def search_context(
        self,
        project_id: str,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history by content.

        Args:
            project_id: Project identifier
            query: Search query
            session_id: Optional session filter
            limit: Maximum results

        Returns:
            List of matching messages
        """
        conn = self.get_connection(project_id)

        try:
            if session_id:
                sql = """
                    SELECT role, content, metadata, timestamp, session_id
                    FROM conversations
                    WHERE session_id = ? AND content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                params = (session_id, f"%{query}%", limit)
            else:
                sql = """
                    SELECT role, content, metadata, timestamp, session_id
                    FROM conversations
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                params = (f"%{query}%", limit)

            cursor = conn.execute(sql, params)

            results = []
            for row in cursor.fetchall():
                msg = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"]
                }
                if row["metadata"]:
                    msg["metadata"] = json.loads(row["metadata"])
                results.append(msg)

            return results

        finally:
            conn.close()

    def end_session(self, project_id: str, session_id: str):
        """
        Mark a session as ended (token limit reached).
        """
        conn = self.get_connection(project_id)

        try:
            conn.execute("""
                UPDATE sessions
                SET is_active = 0
                WHERE session_id = ?
            """, (session_id,))

            conn.commit()

        finally:
            conn.close()

    def get_session_stats(
        self,
        project_id: str,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get session statistics."""
        conn = self.get_connection(project_id)

        try:
            cursor = conn.execute("""
                SELECT
                    session_id,
                    created_at,
                    last_active,
                    total_tokens,
                    is_active,
                    (SELECT COUNT(*) FROM conversations WHERE session_id = ?) as message_count
                FROM sessions
                WHERE session_id = ?
            """, (session_id, session_id))

            row = cursor.fetchone()
            if not row:
                return None

            return {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "last_active": row["last_active"],
                "total_tokens": row["total_tokens"],
                "is_active": bool(row["is_active"]),
                "message_count": row["message_count"]
            }

        finally:
            conn.close()

    def list_projects(self) -> List[Dict[str, str]]:
        """List all projects with stored conversations."""
        projects = []

        for db_file in self.storage_dir.glob("*.db"):
            # Extract project ID from filename
            parts = db_file.stem.split("_", 1)
            if len(parts) == 2:
                project_id = parts[1].replace("_", "/")
                projects.append({
                    "project_id": project_id,
                    "db_file": str(db_file),
                    "size_mb": db_file.stat().st_size / (1024 * 1024)
                })

        return projects

    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """Get statistics for a project."""
        conn = self.get_connection(project_id)

        try:
            cursor = conn.execute("""
                SELECT
                    COUNT(DISTINCT session_id) as session_count,
                    COUNT(*) as message_count,
                    SUM(token_count) as total_tokens,
                    COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_sessions
                FROM conversations c
                LEFT JOIN sessions s ON c.session_id = s.session_id
            """)

            row = cursor.fetchone()

            return {
                "project_id": project_id,
                "session_count": row["session_count"] or 0,
                "message_count": row["message_count"] or 0,
                "total_tokens": row["total_tokens"] or 0,
                "active_sessions": row["active_sessions"] or 0
            }

        finally:
            conn.close()


if __name__ == "__main__":
    # Test the storage
    storage = ContextStorage()

    # Store some test messages
    storage.store_message(
        project_id="test_project",
        session_id="session_001",
        role="user",
        content="How do I implement a custom MCP server?",
        token_count=10
    )

    storage.store_message(
        project_id="test_project",
        session_id="session_001",
        role="assistant",
        content="Here's how to implement a custom MCP server...",
        metadata={"model": "claude-sonnet-4.5"},
        token_count=100
    )

    # Get session context
    context = storage.get_session_context("test_project", "session_001")
    print(f"Found {len(context)} messages")

    # Get stats
    stats = storage.get_session_stats("test_project", "session_001")
    print(f"Session stats: {stats}")

    # Search
    results = storage.search_context("test_project", "MCP")
    print(f"Search found {len(results)} results")
