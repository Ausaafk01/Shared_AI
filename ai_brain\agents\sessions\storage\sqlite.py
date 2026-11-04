"""
Persistent SQLite storage implementation for Session records.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, List, Optional, Union

from ai_brain.agents.exceptions import SessionError
from ai_brain.agents.sessions.models import SessionRecord, SessionStatus
from ai_brain.agents.sessions.storage.base import SessionStore

logger = logging.getLogger(__name__)


class SQLiteSessionStore(SessionStore):
    """
    Persistent SQLite-backed session store.
    
    Guarantees:
    - Session records survive process and application restarts.
    - Indexed lookups by session_id, project_id, current_agent_id, and status.
    - Thread-safe access via re-entrant locking.
    """

    def __init__(self, database_path: Union[str, Path] = "sessions.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._closed = False

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise SessionError(f"Failed to create directory for session database: {e}") from e

        try:
            self._conn = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )
            self._conn.row_factory = sqlite3.Row
            if self.database_path != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                except sqlite3.Error:
                    pass
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._create_schema()
        except sqlite3.Error as e:
            raise SessionError(f"Failed to initialize SQLite session store: {e}") from e

    def _create_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    current_agent_id TEXT,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(current_agent_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);")

    def save(self, session: SessionRecord) -> None:
        """Persist or update a SessionRecord."""
        with self._lock:
            if self._closed:
                raise SessionError("Session store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        project_id,
                        title,
                        current_agent_id,
                        status,
                        data,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        project_id=excluded.project_id,
                        title=excluded.title,
                        current_agent_id=excluded.current_agent_id,
                        status=excluded.status,
                        data=excluded.data,
                        updated_at=excluded.updated_at;
                    """,
                    (
                        session.session_id,
                        session.project_id,
                        session.title,
                        session.current_agent_id,
                        session.status.value,
                        session.to_json(),
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                    ),
                )
            except sqlite3.Error as e:
                raise SessionError(f"Failed to persist session '{session.session_id}': {e}") from e

    def get(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by its session_id."""
        with self._lock:
            if self._closed:
                raise SessionError("Session store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute("SELECT data FROM sessions WHERE session_id = ?;", (session_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return SessionRecord.from_json(row["data"])
            except sqlite3.Error as e:
                raise SessionError(f"Failed to load session '{session_id}': {e}") from e

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
    ) -> List[SessionRecord]:
        """List and filter session records."""
        with self._lock:
            if self._closed:
                raise SessionError("Session store is closed")
            try:
                query = "SELECT data FROM sessions WHERE 1=1"
                params: List[Any] = []
                if project_id:
                    query += " AND project_id = ?"
                    params.append(project_id)
                if agent_id:
                    query += " AND (current_agent_id = ? OR data LIKE ?)"
                    params.append(agent_id)
                    params.append(f'%"{agent_id}"%')
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                query += " ORDER BY created_at ASC;"

                cur = self._conn.cursor()
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [SessionRecord.from_json(r["data"]) for r in rows]
            except sqlite3.Error as e:
                raise SessionError(f"Failed to query sessions: {e}") from e

    def delete(self, session_id: str) -> bool:
        """Delete a session record."""
        with self._lock:
            if self._closed:
                raise SessionError("Session store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
                return cur.rowcount > 0
            except sqlite3.Error as e:
                raise SessionError(f"Failed to delete session '{session_id}': {e}") from e

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            if not self._closed:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._closed = True

