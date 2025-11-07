"""
Context manager for CLI-based LLM tools.
Manages project-based conversation contexts with session tracking.
"""
import os
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path
from storage import ContextStorage


class ContextManager:
    """
    High-level context management for CLI LLM tools.
    Automatically detects project and manages sessions.
    """

    def __init__(self, storage_dir: str = "~/.mcp_contexts"):
        """
        Initialize context manager.

        Args:
            storage_dir: Directory for storing all contexts
        """
        self.storage = ContextStorage(storage_dir)
        self.current_session_id: Optional[str] = None
        self.current_project_id: Optional[str] = None

    def detect_project(self, cwd: Optional[str] = None) -> str:
        """
        Auto-detect current project from working directory.

        Args:
            cwd: Current working directory (default: os.getcwd())

        Returns:
            Project identifier (typically git repo name or directory name)
        """
        if cwd is None:
            cwd = os.getcwd()

        cwd_path = Path(cwd).resolve()

        # Try to find git root
        current = cwd_path
        while current != current.parent:
            if (current / ".git").exists():
                return current.name

            current = current.parent

        # Fallback to directory name
        return cwd_path.name

    def start_session(
        self,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a new session or resume existing one.

        Args:
            project_id: Project identifier (auto-detected if None)
            session_id: Session identifier (auto-generated if None)

        Returns:
            Session ID
        """
        if project_id is None:
            project_id = self.detect_project()

        if session_id is None:
            session_id = f"{project_id}_{uuid.uuid4().hex[:8]}"

        self.current_project_id = project_id
        self.current_session_id = session_id

        return session_id

    def store_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: int = 0
    ):
        """
        Store a message in the current session.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            token_count: Token count for this message
        """
        if not self.current_session_id or not self.current_project_id:
            raise ValueError("No active session. Call start_session() first.")

        self.storage.store_message(
            project_id=self.current_project_id,
            session_id=self.current_session_id,
            role=role,
            content=content,
            metadata=metadata,
            token_count=token_count
        )

    def get_context(
        self,
        limit: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for current or specified session.

        Args:
            limit: Maximum number of messages
            session_id: Specific session (default: current)

        Returns:
            List of messages
        """
        if not self.current_project_id:
            raise ValueError("No active project. Call start_session() first.")

        target_session = session_id or self.current_session_id
        if not target_session:
            raise ValueError("No active session.")

        return self.storage.get_session_context(
            project_id=self.current_project_id,
            session_id=target_session,
            limit=limit
        )

    def search_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history.

        Args:
            query: Search query
            session_id: Optional session filter (searches all if None)
            limit: Maximum results

        Returns:
            List of matching messages
        """
        if not self.current_project_id:
            raise ValueError("No active project.")

        return self.storage.search_context(
            project_id=self.current_project_id,
            query=query,
            session_id=session_id,
            limit=limit
        )

    def end_session(self, session_id: Optional[str] = None):
        """
        End current session (e.g., when token limit reached).

        Args:
            session_id: Session to end (default: current)
        """
        if not self.current_project_id:
            return

        target_session = session_id or self.current_session_id
        if target_session:
            self.storage.end_session(
                project_id=self.current_project_id,
                session_id=target_session
            )

        # Clear current session if it was ended
        if target_session == self.current_session_id:
            self.current_session_id = None

    def get_session_stats(
        self,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics for current or specified session.

        Args:
            session_id: Session identifier (default: current)

        Returns:
            Session statistics or None
        """
        if not self.current_project_id:
            return None

        target_session = session_id or self.current_session_id
        if not target_session:
            return None

        return self.storage.get_session_stats(
            project_id=self.current_project_id,
            session_id=target_session
        )

    def list_projects(self) -> List[Dict[str, str]]:
        """List all projects with conversations."""
        return self.storage.list_projects()

    def get_project_stats(
        self,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a project.

        Args:
            project_id: Project identifier (default: current)

        Returns:
            Project statistics
        """
        target_project = project_id or self.current_project_id
        if not target_project:
            raise ValueError("No active project.")

        return self.storage.get_project_stats(target_project)


if __name__ == "__main__":
    # Test the context manager
    manager = ContextManager(storage_dir=".test_contexts")

    # Start session
    session_id = manager.start_session(project_id="test_project")
    print(f"Started session: {session_id}")

    # Store messages
    manager.store_message("user", "Hello, how are you?", token_count=5)
    manager.store_message("assistant", "I'm doing great!", token_count=4)

    # Get context
    context = manager.get_context()
    print(f"Context has {len(context)} messages")

    # Get stats
    stats = manager.get_session_stats()
    print(f"Session stats: {stats}")

    # Search
    results = manager.search_context("hello")
    print(f"Search found {len(results)} results")

    # End session
    manager.end_session()
    print("Session ended")
