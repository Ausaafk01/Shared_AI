"""
Persistent SQLite implementation of the AI Brain Event Storage.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional, Union

from ..exceptions import (
    DuplicateEventError,
    StorageCorruptionError,
    StorageError,
)
from ..models import Event
from ..types import EventType, normalize_event_type
from .base import EventStore

logger = logging.getLogger(__name__)


class SQLiteEventStore(EventStore):
    """
    Lightweight, persistent local event storage powered by SQLite.
    
    Guarantees:
    - Events persist across application restarts.
    - Append-only semantics enforced via database triggers disallowing UPDATE and DELETE.
    - Strict primary-key uniqueness on event_id.
    - High-performance indexed lookups by project, agent, session, type, and timestamp.
    - Thread-safe access via internal locking.
    """

    def __init__(
        self,
        database_path: Union[str, Path] = "events.db",
        *,
        ignore_corrupt: bool = False,
    ) -> None:
        """
        Initialize the SQLite event store.
        
        Args:
            database_path: File path for the SQLite database, or ':memory:' for ephemeral.
            ignore_corrupt: If True, skip corrupt records instead of raising StorageCorruptionError.
        """
        self.database_path = str(database_path)
        self.ignore_corrupt = ignore_corrupt
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = True

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise StorageError(f"Failed to create directory for database: {e}") from e

        try:
            self._conn = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode; we manage transactions explicitly
            )
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()
            self._closed = False
        except sqlite3.Error as e:
            raise StorageError(f"Failed to connect or initialize SQLite database: {e}") from e

    def _initialize_schema(self) -> None:
        """Initialize schema, indexes, and append-only constraints."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                # Performance & safety pragmas
                if self.database_path != ":memory:":
                    cur.execute("PRAGMA journal_mode = WAL;")
                cur.execute("PRAGMA synchronous = NORMAL;")
                cur.execute("PRAGMA foreign_keys = ON;")

                # Main events table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_id TEXT,
                    session_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    tool_used TEXT,
                    files_involved_json TEXT,
                    result_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                # Indexes for fast querying
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id, timestamp);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id, timestamp);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, timestamp);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, timestamp);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp, id);")

                # Database-level triggers guaranteeing append-only immutability
                cur.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_event_update
                BEFORE UPDATE ON events
                BEGIN
                    SELECT RAISE(ABORT, 'Event records are immutable: UPDATE operations are strictly prohibited.');
                END;
                """)

                cur.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_event_delete
                BEFORE DELETE ON events
                BEGIN
                    SELECT RAISE(ABORT, 'Event records are append-only: DELETE operations are strictly prohibited.');
                END;
                """)
            finally:
                cur.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("Event store has been closed")

    def store_event(self, event: Event) -> None:
        """
        Store an event in the database.
        
        Raises:
            DuplicateEventError: If an event with the same ID already exists.
            StorageError: If the persistence operation fails.
        """
        self._ensure_open()
        if not isinstance(event, Event):
            raise StorageError(f"Expected Event object, got {type(event).__name__}")

        now_utc = datetime.now(timezone.utc).isoformat()
        
        insert_sql = """
        INSERT INTO events (
            event_id,
            event_type,
            timestamp,
            project_id,
            agent_id,
            session_id,
            actor,
            input_json,
            output_json,
            tool_used,
            files_involved_json,
            result_json,
            metadata_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        params = (
            event.event_id,
            event.event_type,
            event.timestamp.isoformat(),
            event.project_id,
            event.agent_id,
            event.session_id,
            event.actor,
            json.dumps(event.to_dict()["input"]) if event.input is not None else None,
            json.dumps(event.to_dict()["output"]) if event.output is not None else None,
            event.tool_used,
            json.dumps(list(event.files_involved)),
            json.dumps(event.to_dict()["result"]) if event.result is not None else None,
            json.dumps(event.to_dict()["metadata"]),
            now_utc,
        )

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(insert_sql, params)
                cur.execute("COMMIT;")
            except sqlite3.IntegrityError as e:
                cur.execute("ROLLBACK;")
                err_msg = str(e).lower()
                if "unique constraint" in err_msg or "events.event_id" in err_msg:
                    raise DuplicateEventError(
                        f"Event with ID '{event.event_id}' already exists in the store."
                    ) from e
                raise StorageError(f"Database integrity error: {e}") from e
            except sqlite3.Error as e:
                cur.execute("ROLLBACK;")
                raise StorageError(f"Failed to store event: {e}") from e
            finally:
                cur.close()

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Deserialize a SQLite database row into an immutable Event."""
        try:
            input_val = json.loads(row["input_json"]) if row["input_json"] is not None else None
            output_val = json.loads(row["output_json"]) if row["output_json"] is not None else None
            result_val = json.loads(row["result_json"]) if row["result_json"] is not None else None
            files_val = json.loads(row["files_involved_json"]) if row["files_involved_json"] is not None else []
            metadata_val = json.loads(row["metadata_json"]) if row["metadata_json"] is not None else {}

            event_dict = {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "timestamp": row["timestamp"],
                "project_id": row["project_id"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "actor": row["actor"],
                "input": input_val,
                "output": output_val,
                "tool_used": row["tool_used"],
                "files_involved": files_val,
                "result": result_val,
                "metadata": metadata_val,
            }
            return Event.from_dict(event_dict)
        except Exception as e:
            event_id = row["event_id"] if "event_id" in row.keys() else "unknown"
            raise StorageCorruptionError(
                f"Corrupt event record for ID '{event_id}': {e}"
            ) from e

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Retrieve an event by its unique ID, or None if not found."""
        self._ensure_open()
        if not event_id:
            return None

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT * FROM events WHERE event_id = ? LIMIT 1;", (str(event_id).strip(),))
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_event(row)
            except StorageCorruptionError:
                raise
            except sqlite3.Error as e:
                raise StorageError(f"Database error retrieving event '{event_id}': {e}") from e
            finally:
                cur.close()

    def get_events_by_project(
        self, project_id: str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events belonging to a project ordered chronologically."""
        return self.filter_events(project_id=project_id, ascending=ascending)

    def get_events_by_agent(
        self, agent_id: str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events involving a specific agent ordered chronologically."""
        return self.filter_events(agent_id=agent_id, ascending=ascending)

    def get_events_by_session(
        self, session_id: str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events in a session ordered chronologically."""
        return self.filter_events(session_id=session_id, ascending=ascending)

    def get_events_by_type(
        self, event_type: EventType | str, ascending: bool = True
    ) -> list[Event]:
        """Retrieve all events of a specific type ordered chronologically."""
        normalized = normalize_event_type(event_type)
        return self.filter_events(event_type=normalized, ascending=ascending)

    def get_all_events(self, ascending: bool = True) -> list[Event]:
        """Retrieve all stored events ordered chronologically."""
        return self.filter_events(ascending=ascending)

    def filter_events(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[EventType | str] = None,
        actor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        ascending: bool = True,
    ) -> list[Event]:
        """Query events with flexible filtering, ordered chronologically."""
        self._ensure_open()

        clauses: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id.strip())

        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id.strip())

        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id.strip())

        if event_type is not None:
            norm_type = normalize_event_type(event_type)
            clauses.append("event_type = ?")
            params.append(norm_type)

        if actor is not None:
            clauses.append("actor = ?")
            params.append(actor.strip())

        if start_time is not None:
            utc_start = (
                start_time.replace(tzinfo=timezone.utc)
                if start_time.tzinfo is None
                else start_time.astimezone(timezone.utc)
            )
            clauses.append("timestamp >= ?")
            params.append(utc_start.isoformat())

        if end_time is not None:
            utc_end = (
                end_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None
                else end_time.astimezone(timezone.utc)
            )
            clauses.append("timestamp <= ?")
            params.append(utc_end.isoformat())

        order_dir = "ASC" if ascending else "DESC"
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query_sql = f"SELECT * FROM events {where_sql} ORDER BY timestamp {order_dir}, id {order_dir}"

        if limit is not None:
            query_sql += " LIMIT ?"
            params.append(max(0, int(limit)))
            if offset > 0:
                query_sql += " OFFSET ?"
                params.append(int(offset))
        elif offset > 0:
            query_sql += " LIMIT -1 OFFSET ?"
            params.append(int(offset))

        events: list[Event] = []
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(query_sql, params)
                rows = cur.fetchall()
                for row in rows:
                    try:
                        events.append(self._row_to_event(row))
                    except StorageCorruptionError as e:
                        if self.ignore_corrupt:
                            logger.warning("Skipping corrupt record: %s", e)
                            continue
                        raise
            except sqlite3.Error as e:
                raise StorageError(f"Database query error: {e}") from e
            finally:
                cur.close()

        return events

    def count(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[EventType | str] = None,
    ) -> int:
        """Count the number of stored events matching the filter."""
        self._ensure_open()
        clauses: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id.strip())

        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id.strip())

        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id.strip())

        if event_type is not None:
            norm_type = normalize_event_type(event_type)
            clauses.append("event_type = ?")
            params.append(norm_type)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT COUNT(*) FROM events {where_sql}"

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row else 0
            except sqlite3.Error as e:
                raise StorageError(f"Database count query failed: {e}") from e
            finally:
                cur.close()

    def close(self) -> None:
        """Close the database connection and release resources (idempotent)."""
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

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "SQLiteEventStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

