"""
Persistent SQLite vector storage backend for AI Brain.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, List, Optional, Sequence, Union

from ai_brain.vectors.exceptions import (
    DuplicateVectorError,
    VectorNotFoundError,
    VectorStorageError,
    VectorValidationError,
)
from ai_brain.vectors.models import VectorRecord
from ai_brain.vectors.storage.base import VectorStore

logger = logging.getLogger(__name__)


class SQLiteVectorStore(VectorStore):
    """
    Persistent SQLite storage engine for vector records.
    
    Guarantees:
    - Persistence across application and process restarts.
    - Project isolation at the database index and query level.
    - Duplicate prevention for identical memory version and embedding model.
    - Thread-safe operations via internal reentrant locking.
    """

    def __init__(self, database_path: Union[str, Path] = "vectors.db") -> None:
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
                    raise VectorStorageError(f"Failed to create directory for vector database: {e}") from e

        try:
            self._conn = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._init_db()
            self._closed = False
        except sqlite3.Error as e:
            raise VectorStorageError(f"Failed to initialize SQLite vector storage: {e}") from e

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            cur.execute("PRAGMA foreign_keys = ON;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_records (
                    record_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, memory_id, version, model_id)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vec_proj ON vector_records(project_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vec_mem ON vector_records(project_id, memory_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vec_status ON vector_records(project_id, status);")
            self._conn.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStorageError("Cannot perform operations on closed vector store")

    def _row_to_record(self, row: sqlite3.Row) -> VectorRecord:
        """Convert a database row into an immutable VectorRecord."""
        try:
            emb_list = json.loads(row["embedding"])
            meta_dict = json.loads(row["metadata"])
            return VectorRecord.create(
                record_id=row["record_id"],
                memory_id=row["memory_id"],
                project_id=row["project_id"],
                version=row["version"],
                model_id=row["model_id"],
                embedding=emb_list,
                content=row["content"],
                status=row["status"],
                metadata=meta_dict,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        except Exception as e:
            raise VectorStorageError(f"Failed to deserialize vector record '{row['record_id']}': {e}") from e

    def add_vector(self, record: VectorRecord, *, fail_if_exists: bool = False) -> VectorRecord:
        self._ensure_open()
        if not isinstance(record, VectorRecord):
            raise VectorValidationError(f"Expected VectorRecord, got {type(record).__name__}")

        with self._lock:
            cur = self._conn.cursor()
            # Check for existing record by unique constraint (project_id, memory_id, version, model_id)
            cur.execute(
                """
                SELECT record_id FROM vector_records
                WHERE project_id = ? AND memory_id = ? AND version = ? AND model_id = ?
                """,
                (record.project_id, record.memory_id, record.version, record.model_id),
            )
            existing_row = cur.fetchone()
            if existing_row:
                if fail_if_exists:
                    raise DuplicateVectorError(
                        f"Vector record already exists for memory '{record.memory_id}' (version {record.version}, model '{record.model_id}')"
                    )
                return self.get_vector(existing_row["record_id"])  # type: ignore

            try:
                cur.execute(
                    """
                    INSERT INTO vector_records (
                        record_id, memory_id, project_id, version, model_id,
                        dimensions, embedding, content, status, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.memory_id,
                        record.project_id,
                        record.version,
                        record.model_id,
                        record.dimensions,
                        json.dumps(list(record.embedding)),
                        record.content,
                        record.status,
                        json.dumps(dict(record.metadata)),
                        record.created_at.isoformat(),
                    ),
                )
                self._conn.commit()
                return record
            except sqlite3.IntegrityError as e:
                if fail_if_exists:
                    raise DuplicateVectorError(f"Duplicate vector insertion: {e}") from e
                # Fallback retrieve existing
                cur.execute(
                    "SELECT record_id FROM vector_records WHERE record_id = ?", (record.record_id,)
                )
                r = cur.fetchone()
                if r:
                    return self.get_vector(r["record_id"])  # type: ignore
                raise VectorStorageError(f"Integrity error during vector insertion: {e}") from e

    def get_vector(self, record_id: str) -> Optional[VectorRecord]:
        self._ensure_open()
        if not record_id:
            return None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM vector_records WHERE record_id = ?", (record_id,))
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def get_vector_by_memory(
        self,
        project_id: str,
        memory_id: str,
        version: int,
        model_id: str,
    ) -> Optional[VectorRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT * FROM vector_records
                WHERE project_id = ? AND memory_id = ? AND version = ? AND model_id = ?
                """,
                (project_id, memory_id, version, model_id),
            )
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def update_vector(self, record: VectorRecord) -> VectorRecord:
        self._ensure_open()
        if not isinstance(record, VectorRecord):
            raise VectorValidationError(f"Expected VectorRecord, got {type(record).__name__}")

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE vector_records
                SET status = ?, metadata = ?, content = ?
                WHERE record_id = ?
                """,
                (
                    record.status,
                    json.dumps(dict(record.metadata)),
                    record.content,
                    record.record_id,
                ),
            )
            if cur.rowcount == 0:
                raise VectorNotFoundError(f"Vector record '{record.record_id}' not found for update")
            self._conn.commit()
            return record

    def delete_vector(self, record_id: str) -> bool:
        self._ensure_open()
        if not record_id:
            return False
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM vector_records WHERE record_id = ?", (record_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_vectors_by_memory(self, project_id: str, memory_id: str) -> int:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM vector_records WHERE project_id = ? AND memory_id = ?",
                (project_id, memory_id),
            )
            self._conn.commit()
            return cur.rowcount

    def get_vectors_by_project(
        self,
        project_id: str,
        *,
        model_id: Optional[str] = None,
        allowed_statuses: Optional[Sequence[str]] = None,
    ) -> List[VectorRecord]:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            query = "SELECT * FROM vector_records WHERE project_id = ?"
            params: List[Any] = [project_id]

            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)

            if allowed_statuses:
                placeholders = ",".join("?" for _ in allowed_statuses)
                query += f" AND status IN ({placeholders})"
                params.extend([s.upper() for s in allowed_statuses])

            query += " ORDER BY created_at ASC"
            cur.execute(query, params)
            return [self._row_to_record(row) for row in cur.fetchall()]

    def count(self, project_id: Optional[str] = None) -> int:
        self._ensure_open()
        with self._lock:
            cur = self._conn.cursor()
            if project_id:
                cur.execute("SELECT COUNT(*) FROM vector_records WHERE project_id = ?", (project_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM vector_records")
            return cur.fetchone()[0]

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

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

