"""
Persistent SQLite storage implementation for Brain Loop interactions.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, List, Optional, Union

from ai_brain.loop.exceptions import BrainLoopError
from ai_brain.loop.models import InteractionRecord, InteractionStatus
from ai_brain.loop.storage.base import InteractionStore

logger = logging.getLogger(__name__)


class SQLiteInteractionStore(InteractionStore):
    """
    Persistent SQLite-backed interaction store.
    
    Guarantees:
    - Interaction records and lifecycle state survive application/process restarts.
    - Preserves project, agent, session, interaction ID, lifecycle status, and stage progress.
    - Preserves all linkages to input, context, response, and completion events.
    - Indexed lookups by interaction_id, idempotency_key, project_id, session_id, and status.
    - Thread-safe operations via re-entrant locking.
    """

    def __init__(self, database_path: Union[str, Path] = "interactions.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = True

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise BrainLoopError(f"Failed to create directory for interaction database: {e}") from e

        try:
            self._conn = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode; transactions managed explicitly
            )
            self._conn.row_factory = sqlite3.Row
            if self.database_path != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                except sqlite3.Error:
                    pass
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._create_schema()
            self._closed = False
        except sqlite3.Error as e:
            raise BrainLoopError(f"Failed to initialize SQLite interaction store: {e}") from e

    def _create_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    idempotency_key TEXT,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    stages_completed TEXT NOT NULL,
                    input_event_id TEXT,
                    context_event_id TEXT,
                    response_event_id TEXT,
                    completion_event_id TEXT,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_project ON interactions(project_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_idempotency ON interactions(idempotency_key) WHERE idempotency_key IS NOT NULL;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status);")

    def save(self, record: InteractionRecord) -> None:
        """Persist or update an InteractionRecord in the SQLite database."""
        with self._lock:
            if self._closed:
                raise BrainLoopError("Interaction store is closed")
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO interactions (
                        interaction_id,
                        idempotency_key,
                        project_id,
                        agent_id,
                        session_id,
                        status,
                        current_stage,
                        stages_completed,
                        input_event_id,
                        context_event_id,
                        response_event_id,
                        completion_event_id,
                        data,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(interaction_id) DO UPDATE SET
                        idempotency_key=excluded.idempotency_key,
                        project_id=excluded.project_id,
                        agent_id=excluded.agent_id,
                        session_id=excluded.session_id,
                        status=excluded.status,
                        current_stage=excluded.current_stage,
                        stages_completed=excluded.stages_completed,
                        input_event_id=excluded.input_event_id,
                        context_event_id=excluded.context_event_id,
                        response_event_id=excluded.response_event_id,
                        completion_event_id=excluded.completion_event_id,
                        data=excluded.data,
                        updated_at=excluded.updated_at;
                    """,
                    (
                        record.interaction_id,
                        record.idempotency_key,
                        record.project_id,
                        record.agent_id,
                        record.session_id,
                        record.status.value,
                        record.current_stage,
                        json.dumps(list(record.stages_completed)),
                        record.input_event_id,
                        record.context_event_id,
                        record.response_event_id,
                        record.completion_event_id,
                        record.to_json(),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
            except sqlite3.Error as e:
                raise BrainLoopError(f"Failed to persist interaction record '{record.interaction_id}': {e}") from e

    def get(self, interaction_id: str, project_id: Optional[str] = None) -> Optional[InteractionRecord]:
        """Retrieve an interaction record by its interaction_id."""
        with self._lock:
            if self._closed:
                raise BrainLoopError("Interaction store is closed")
            try:
                cur = self._conn.cursor()
                if project_id:
                    cur.execute(
                        "SELECT data FROM interactions WHERE interaction_id = ? AND project_id = ?;",
                        (interaction_id, project_id),
                    )
                else:
                    cur.execute("SELECT data FROM interactions WHERE interaction_id = ?;", (interaction_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return InteractionRecord.from_json(row["data"])
            except sqlite3.Error as e:
                raise BrainLoopError(f"Failed to load interaction '{interaction_id}': {e}") from e

    def get_by_idempotency_key(
        self, idempotency_key: str, project_id: Optional[str] = None
    ) -> Optional[InteractionRecord]:
        """Retrieve an interaction record by its idempotency_key."""
        with self._lock:
            if self._closed:
                raise BrainLoopError("Interaction store is closed")
            try:
                cur = self._conn.cursor()
                if project_id:
                    cur.execute(
                        "SELECT data FROM interactions WHERE idempotency_key = ? AND project_id = ?;",
                        (idempotency_key, project_id),
                    )
                else:
                    cur.execute("SELECT data FROM interactions WHERE idempotency_key = ?;", (idempotency_key,))
                row = cur.fetchone()
                if row is None:
                    return None
                return InteractionRecord.from_json(row["data"])
            except sqlite3.Error as e:
                raise BrainLoopError(f"Failed to query interaction by idempotency_key '{idempotency_key}': {e}") from e

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[Union[str, InteractionStatus]] = None,
    ) -> List[InteractionRecord]:
        """List interaction records matching the specified filters."""
        with self._lock:
            if self._closed:
                raise BrainLoopError("Interaction store is closed")
            try:
                query = "SELECT data FROM interactions WHERE 1=1"
                params: List[Any] = []
                if project_id:
                    query += " AND project_id = ?"
                    params.append(project_id)
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                if status:
                    status_str = status.value if isinstance(status, InteractionStatus) else str(status)
                    query += " AND status = ?"
                    params.append(status_str)
                query += " ORDER BY created_at ASC;"

                cur = self._conn.cursor()
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [InteractionRecord.from_json(r["data"]) for r in rows]
            except sqlite3.Error as e:
                raise BrainLoopError(f"Failed to query interactions: {e}") from e

    def close(self) -> None:
        """Close SQLite database connection (idempotent)."""
        if getattr(self, "_closed", False):
            return
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if getattr(self, "_closed", False):
                    return
                conn = getattr(self, "_conn", None)
                if conn is not None:
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                self._closed = True
        else:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._closed = True

    def __enter__(self) -> "SQLiteInteractionStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

