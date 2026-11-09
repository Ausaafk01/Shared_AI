"""
Persistent SQLite implementation of the AI Brain Memory Storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union

from ai_brain.memories.exceptions import MemoryValidationError
from ai_brain.memories.models import Memory, _freeze_container, _unfreeze_container
from ai_brain.memories.storage.base import MemoryStore
from ai_brain.memories.types import MemoryStatus, normalize_memory_status

logger = logging.getLogger(__name__)


class SQLiteMemoryStore(MemoryStore):
    """
    Lightweight, persistent local memory storage powered by SQLite.
    
    Guarantees:
    - Thread-safe operations via internal re-entrant locking.
    - Version chain tracking and OCC validation per root_memory_id.
    - Automatic superseding of older active versions upon storage of new active versions.
    - Compatibility with both file-backed and ephemeral ':memory:' databases.
    - Preserves all metadata, provenance, relationships, and timestamps.
    """

    def __init__(
        self,
        database_path: Union[str, Path] = ":memory:",
        default_project_id: Optional[str] = None,
    ) -> None:
        self.database_path = str(database_path)
        self.default_project_id = default_project_id
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = True

        if self.database_path != ":memory:":
            db_dir = Path(self.database_path).parent
            if db_dir and not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise RuntimeError(f"Failed to create directory for memory database: {e}") from e

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
        except Exception as e:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to initialize SQLiteMemoryStore: {e}") from e

    def _create_schema(self) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    root_memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    tier TEXT NOT NULL DEFAULT 'SHORT_TERM',
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    dispute_status TEXT NOT NULL DEFAULT 'NONE',
                    supersedes_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_proj_root ON memories(project_id, root_memory_id);
                CREATE INDEX IF NOT EXISTS idx_memories_proj_status ON memories(project_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_proj_tier ON memories(project_id, tier);
                """
            )

    def _resolve_project(self, project_id: Optional[str]) -> str:
        proj = project_id or self.default_project_id
        if not proj:
            raise MemoryValidationError(
                "project_id must be provided or default_project_id set on SQLiteMemoryStore."
            )
        return proj

    def store(self, memory: Memory) -> None:
        """Persist a memory instance with versioning and automatic superseding."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        proj_id = memory.project_id
        root_id = memory.root_memory_id or memory.memory_id
        created_iso = (
            memory.created_at.replace(tzinfo=timezone.utc).isoformat()
            if memory.created_at.tzinfo is None
            else memory.created_at.astimezone(timezone.utc).isoformat()
        )
        updated_iso = (
            memory.updated_at.replace(tzinfo=timezone.utc).isoformat()
            if memory.updated_at.tzinfo is None
            else memory.updated_at.astimezone(timezone.utc).isoformat()
        )

        meta_payload: Dict[str, Any] = {
            "root_memory_id": root_id,
            "parent_memory_id": memory.parent_memory_id,
            "source_event_ids": list(memory.source_event_ids),
            "source_agent_ids": list(memory.source_agent_ids),
            "importance": memory.importance,
            "confidence": memory.confidence,
            "relationships": _unfreeze_container(memory.relationships),
            "memory_type": memory.memory_type,
            "custom_metadata": _unfreeze_container(memory.metadata),
        }

        with self._lock:
            # Check existing version conflicts
            cursor = self._conn.execute(
                "SELECT memory_id, version FROM memories WHERE project_id = ? AND root_memory_id = ? ORDER BY version DESC",
                (proj_id, root_id),
            )
            for row in cursor.fetchall():
                if row["memory_id"] != memory.memory_id and row["version"] >= memory.version:
                    logger.warning(
                        "Concurrent version conflict on root %s: existing version %s >= new version %s",
                        root_id,
                        row["version"],
                        memory.version,
                    )

            # Insert or replace
            self._conn.execute(
                """
                INSERT INTO memories (
                    memory_id, project_id, root_memory_id, version, status,
                    tier, content, metadata, dispute_status, supersedes_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    root_memory_id = excluded.root_memory_id,
                    version = excluded.version,
                    status = excluded.status,
                    tier = excluded.tier,
                    content = excluded.content,
                    metadata = excluded.metadata,
                    dispute_status = excluded.dispute_status,
                    supersedes_id = excluded.supersedes_id,
                    updated_at = excluded.updated_at;
                """,
                (
                    memory.memory_id,
                    proj_id,
                    root_id,
                    memory.version,
                    memory.status,
                    memory.memory_type.upper(),
                    memory.content,
                    json.dumps(meta_payload),
                    "NONE",
                    memory.parent_memory_id,
                    created_iso,
                    updated_iso,
                ),
            )

            # If ACTIVE, supersede older versions
            if memory.status == MemoryStatus.ACTIVE.value:
                self._conn.execute(
                    """
                    UPDATE memories
                    SET status = 'SUPERSEDED', updated_at = ?
                    WHERE project_id = ?
                      AND root_memory_id = ?
                      AND memory_id != ?
                      AND version < ?
                      AND status = 'ACTIVE';
                    """,
                    (updated_iso, proj_id, root_id, memory.memory_id, memory.version),
                )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        meta_raw = row["metadata"]
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw or {})

        root_id = row["root_memory_id"] or meta.get("root_memory_id") or row["memory_id"]
        parent_id = row["supersedes_id"] or meta.get("parent_memory_id")
        source_event_ids = tuple(meta.get("source_event_ids", ()))
        source_agent_ids = tuple(meta.get("source_agent_ids", ()))
        importance = float(meta.get("importance", 0.5))
        confidence = float(meta.get("confidence", 1.0))
        relationships = meta.get("relationships", {})
        memory_type = meta.get("memory_type", row["tier"].lower())
        custom_metadata = meta.get("custom_metadata", {})

        created_dt = datetime.fromisoformat(row["created_at"])
        updated_dt = datetime.fromisoformat(row["updated_at"])

        return Memory(
            memory_id=row["memory_id"],
            memory_type=memory_type,
            content=row["content"],
            project_id=row["project_id"],
            created_at=created_dt,
            updated_at=updated_dt,
            source_event_ids=source_event_ids,
            source_agent_ids=source_agent_ids,
            importance=importance,
            confidence=confidence,
            status=row["status"],
            relationships=relationships,
            metadata=custom_metadata,
            root_memory_id=root_id,
            parent_memory_id=parent_id,
            version=row["version"],
        )

    def get(self, memory_id: str, project_id: Optional[str] = None) -> Optional[Memory]:
        """Retrieve a memory snapshot by memory_id."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        with self._lock:
            if project_id:
                cursor = self._conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ? AND project_id = ?",
                    (memory_id, project_id),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?",
                    (memory_id,),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    def get_by_root_id(self, root_memory_id: str, project_id: Optional[str] = None) -> List[Memory]:
        """Retrieve all versions in a conceptual memory lineage, ordered by version asc."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        with self._lock:
            if project_id:
                cursor = self._conn.execute(
                    "SELECT * FROM memories WHERE project_id = ? AND (root_memory_id = ? OR memory_id = ?) ORDER BY version ASC",
                    (project_id, root_memory_id, root_memory_id),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM memories WHERE (root_memory_id = ? OR memory_id = ?) ORDER BY version ASC",
                    (root_memory_id, root_memory_id),
                )
            return [self._row_to_memory(r) for r in cursor.fetchall()]

    def get_current(self, root_memory_id: str, project_id: Optional[str] = None) -> Optional[Memory]:
        """Retrieve current active or latest version for root_memory_id."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        with self._lock:
            # Try ACTIVE first
            if project_id:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE project_id = ? AND (root_memory_id = ? OR memory_id = ?) AND status = 'ACTIVE'
                    ORDER BY version DESC LIMIT 1
                    """,
                    (project_id, root_memory_id, root_memory_id),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE (root_memory_id = ? OR memory_id = ?) AND status = 'ACTIVE'
                    ORDER BY version DESC LIMIT 1
                    """,
                    (root_memory_id, root_memory_id),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)

            # Fallback to highest version
            if project_id:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE project_id = ? AND (root_memory_id = ? OR memory_id = ?)
                    ORDER BY version DESC LIMIT 1
                    """,
                    (project_id, root_memory_id, root_memory_id),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE (root_memory_id = ? OR memory_id = ?)
                    ORDER BY version DESC LIMIT 1
                    """,
                    (root_memory_id, root_memory_id),
                )
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)

        return None

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """List and filter memories by project, status, and/or tier."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        sql = "SELECT * FROM memories WHERE project_id = ?"
        params: List[Any] = [proj]

        if status:
            sql += " AND status = ?"
            params.append(status.upper())
        if tier:
            sql += " AND tier = ?"
            params.append(tier.upper())

        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            cursor = self._conn.execute(sql, params)
            return [self._row_to_memory(r) for r in cursor.fetchall()]

    def update_status(self, memory_id: str, new_status: str, project_id: Optional[str] = None) -> bool:
        """Update lifecycle status of a memory."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        norm_status = normalize_memory_status(new_status)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if project_id:
                cursor = self._conn.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ? AND project_id = ?",
                    (norm_status, now_iso, memory_id, project_id),
                )
            else:
                cursor = self._conn.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                    (norm_status, now_iso, memory_id),
                )
            return cursor.rowcount > 0

    def count(self, *, project_id: Optional[str] = None, status: Optional[str] = None) -> int:
        """Count memories matching filter criteria."""
        if self._closed or self._conn is None:
            raise RuntimeError("SQLiteMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        sql = "SELECT COUNT(*) FROM memories WHERE project_id = ?"
        params: List[Any] = [proj]
        if status:
            sql += " AND status = ?"
            params.append(status.upper())

        with self._lock:
            cursor = self._conn.execute(sql, params)
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        with self._lock:
            if not self._closed and self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                self._closed = True
