"""
Persistent SQLite storage implementation for Agent Registry records.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, List, Optional, Union

from ai_brain.agents.exceptions import AgentRegistrationError
from ai_brain.agents.models import AgentRecord, AgentStatus, AgentType
from ai_brain.agents.registry.base import AgentRegistryStore

logger = logging.getLogger(__name__)


class SQLiteAgentRegistryStore(AgentRegistryStore):
    """
    Persistent SQLite-backed agent registry store.
    
    Guarantees:
    - Agent identities survive application and process restarts.
    - Strict primary key uniqueness on agent_id.
    - Indexed lookups by agent_id, status, agent_type, and provider.
    - Thread-safe access via internal locking.
    """

    def __init__(self, database_path: Union[str, Path] = "agents.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._closed = False

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise AgentRegistrationError(f"Failed to create directory for agent registry database: {e}") from e

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
            raise AgentRegistrationError(f"Failed to initialize SQLite agent registry store: {e}") from e

    def _create_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_provider ON agents(provider);")

    def save(self, record: AgentRecord) -> None:
        """Persist or update an AgentRecord."""
        with self._lock:
            if self._closed:
                raise AgentRegistrationError("Agent registry store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO agents (
                        agent_id,
                        name,
                        provider,
                        agent_type,
                        status,
                        data,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id) DO UPDATE SET
                        name=excluded.name,
                        provider=excluded.provider,
                        agent_type=excluded.agent_type,
                        status=excluded.status,
                        data=excluded.data,
                        updated_at=excluded.updated_at;
                    """,
                    (
                        record.agent_id,
                        record.name,
                        record.provider,
                        record.agent_type.value,
                        record.status.value,
                        record.to_json(),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
            except sqlite3.Error as e:
                raise AgentRegistrationError(f"Failed to persist agent '{record.agent_id}': {e}") from e

    def get(self, agent_id: str) -> Optional[AgentRecord]:
        """Retrieve an AgentRecord by its agent_id."""
        with self._lock:
            if self._closed:
                raise AgentRegistrationError("Agent registry store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute("SELECT data FROM agents WHERE agent_id = ?;", (agent_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return AgentRecord.from_json(row["data"])
            except sqlite3.Error as e:
                raise AgentRegistrationError(f"Failed to load agent '{agent_id}': {e}") from e

    def list(
        self,
        *,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        provider: Optional[str] = None,
    ) -> List[AgentRecord]:
        """List and filter registered AgentRecords."""
        with self._lock:
            if self._closed:
                raise AgentRegistrationError("Agent registry store is closed")
            try:
                query = "SELECT data FROM agents WHERE 1=1"
                params: List[Any] = []
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                if agent_type:
                    query += " AND agent_type = ?"
                    params.append(agent_type.value)
                if provider:
                    query += " AND provider = ?"
                    params.append(provider)
                query += " ORDER BY created_at ASC;"

                cur = self._conn.cursor()
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [AgentRecord.from_json(r["data"]) for r in rows]
            except sqlite3.Error as e:
                raise AgentRegistrationError(f"Failed to query agents: {e}") from e

    def delete(self, agent_id: str) -> bool:
        """Remove an AgentRecord from persistent storage."""
        with self._lock:
            if self._closed:
                raise AgentRegistrationError("Agent registry store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute("DELETE FROM agents WHERE agent_id = ?;", (agent_id,))
                return cur.rowcount > 0
            except sqlite3.Error as e:
                raise AgentRegistrationError(f"Failed to delete agent '{agent_id}': {e}") from e

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            if not self._closed:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._closed = True

