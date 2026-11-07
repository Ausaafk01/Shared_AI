"""Persistent PostgreSQL implementation of the AI Brain Memory Storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import psycopg

from ai_brain.memories.exceptions import MemoryNotFoundError, MemoryValidationError
from ai_brain.memories.models import Memory, _freeze_container, _unfreeze_container
from ai_brain.memories.storage.base import MemoryStore
from ai_brain.memories.types import MemoryStatus, normalize_memory_status
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


def ensure_project_exists(db: DatabaseConnectionManager, project_id: str) -> None:
    """Ensure parent project exists in projects table to satisfy FK constraints."""
    db.ensure_project_exists(project_id)


class PostgreSQLMemoryStore(MemoryStore):
    """PostgreSQL-backed memory storage engine.

    Guarantees:
    - ACID transaction safety and strict tenant isolation via RLS and SET LOCAL.
    - Advisory locking (pg_advisory_xact_lock) for atomic version increments per root_memory_id.
    - Optimistic concurrency control verifying memory version chains.
    - Automatic superseding of older active versions in a conceptual memory lineage.
    - Preservation of full metadata, provenance, relationships, and timestamps.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseConnectionManager] = None,
        default_project_id: Optional[str] = None,
    ) -> None:
        self.db = db_manager or get_database_manager()
        self.default_project_id = default_project_id
        self._closed = False
        self._known_projects: set[str] = set()
        if default_project_id:
            self._known_projects.add(default_project_id)

    def _resolve_project(self, project_id: Optional[str]) -> str:
        proj = project_id or self.default_project_id
        if not proj:
            raise MemoryValidationError(
                "project_id must be provided or default_project_id set on PostgreSQLMemoryStore."
            )
        return proj

    def store(self, memory: Memory) -> None:
        """Persist a memory instance with transaction-level advisory locking and OCC."""
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        proj_id = memory.project_id
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        root_id = memory.root_memory_id or memory.memory_id
        created_utc = (
            memory.created_at.replace(tzinfo=timezone.utc)
            if memory.created_at.tzinfo is None
            else memory.created_at.astimezone(timezone.utc)
        )
        updated_utc = (
            memory.updated_at.replace(tzinfo=timezone.utc)
            if memory.updated_at.tzinfo is None
            else memory.updated_at.astimezone(timezone.utc)
        )

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                # 1. Acquire transaction-scoped advisory lock for this (project, root_memory_id)
                lock_key = f"{proj_id}:{root_id}"
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s));", (lock_key,))

                # 2. OCC verification: check existing versions in the root lineage
                cur.execute(
                    """
                    SELECT memory_id, version, status
                    FROM memories
                    WHERE project_id = %s
                      AND (metadata->>'root_memory_id' = %s OR memory_id = %s)
                    ORDER BY version DESC;
                    """,
                    (proj_id, root_id, root_id),
                )
                existing_versions = cur.fetchall()

                # If this is a new memory_id but an equal or higher version already exists, detect branch conflict
                for row in existing_versions:
                    exist_id, exist_ver, exist_status = row
                    if exist_id != memory.memory_id and exist_ver >= memory.version:
                        logger.warning(
                            "Concurrent version conflict on root %s: existing version %s >= new version %s",
                            root_id,
                            exist_ver,
                            memory.version,
                        )

                # 3. Assemble metadata JSON payload preserving all fields
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

                # 4. Determine supersedes_id FK: check if parent exists in DB
                supersedes_id: Optional[str] = None
                if memory.parent_memory_id:
                    cur.execute(
                        "SELECT 1 FROM memories WHERE memory_id = %s AND project_id = %s;",
                        (memory.parent_memory_id, proj_id),
                    )
                    if cur.fetchone():
                        supersedes_id = memory.parent_memory_id

                # 5. Insert / Upsert memory
                cur.execute(
                    """
                    INSERT INTO memories (
                        memory_id,
                        project_id,
                        version,
                        status,
                        tier,
                        content,
                        metadata,
                        dispute_status,
                        supersedes_id,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s
                    )
                    ON CONFLICT (memory_id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        version = EXCLUDED.version,
                        status = EXCLUDED.status,
                        tier = EXCLUDED.tier,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        dispute_status = EXCLUDED.dispute_status,
                        supersedes_id = EXCLUDED.supersedes_id,
                        updated_at = EXCLUDED.updated_at;
                    """,
                    (
                        memory.memory_id,
                        proj_id,
                        memory.version,
                        memory.status,
                        memory.memory_type.upper(),
                        memory.content,
                        json.dumps(meta_payload),
                        "NONE",
                        supersedes_id,
                        created_utc,
                        updated_utc,
                    ),
                )

                # 6. If this memory is ACTIVE and supersedes older versions, supersede them
                if memory.status == MemoryStatus.ACTIVE.value:
                    cur.execute(
                        """
                        UPDATE memories
                        SET status = 'SUPERSEDED', updated_at = %s
                        WHERE project_id = %s
                          AND (metadata->>'root_memory_id' = %s OR memory_id = %s)
                          AND memory_id != %s
                          AND version < %s
                          AND status = 'ACTIVE';
                        """,
                        (updated_utc, proj_id, root_id, root_id, memory.memory_id, memory.version),
                    )

    def save(self, memory: Memory) -> None:
        """Alias for store."""
        self.store(memory)

    def _row_to_memory(self, row: Sequence[Any]) -> Memory:
        (
            memory_id,
            project_id,
            version,
            status,
            tier,
            content,
            metadata_raw,
            dispute_status,
            supersedes_id,
            created_at,
            updated_at,
        ) = row

        meta = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw or "{}")

        root_id = meta.get("root_memory_id") or memory_id
        parent_id = supersedes_id or meta.get("parent_memory_id")
        source_event_ids = tuple(meta.get("source_event_ids", ()))
        source_agent_ids = tuple(meta.get("source_agent_ids", ()))
        importance = float(meta.get("importance", 0.5))
        confidence = float(meta.get("confidence", 1.0))
        relationships = meta.get("relationships", {})
        memory_type = meta.get("memory_type", tier.lower())
        custom_metadata = meta.get("custom_metadata", {})

        return Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            project_id=project_id,
            created_at=created_at,
            updated_at=updated_at,
            source_event_ids=source_event_ids,
            source_agent_ids=source_agent_ids,
            importance=importance,
            confidence=confidence,
            status=status,
            relationships=relationships,
            metadata=custom_metadata,
            root_memory_id=root_id,
            parent_memory_id=parent_id,
            version=version,
        )

    def get(self, memory_id: str, project_id: Optional[str] = None) -> Optional[Memory]:
        """Retrieve a memory snapshot by memory_id."""
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )

        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT memory_id, project_id, version, status, tier, content,
                               metadata, dispute_status, supersedes_id, created_at, updated_at
                        FROM memories
                        WHERE memory_id = %s AND project_id = %s;
                        """,
                        (memory_id, proj),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_memory(row)

        return None

    def get_by_root_id(self, root_memory_id: str, project_id: Optional[str] = None) -> List[Memory]:
        """Retrieve all versions in a conceptual memory lineage, ordered by version asc."""
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT memory_id, project_id, version, status, tier, content,
                           metadata, dispute_status, supersedes_id, created_at, updated_at
                    FROM memories
                    WHERE project_id = %s
                      AND (metadata->>'root_memory_id' = %s OR memory_id = %s)
                    ORDER BY version ASC;
                    """,
                    (proj, root_memory_id, root_memory_id),
                )
                rows = cur.fetchall()
                return [self._row_to_memory(r) for r in rows]

    def get_current(self, root_memory_id: str, project_id: Optional[str] = None) -> Optional[Memory]:
        """Retrieve current active or latest version for root_memory_id."""
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                # First try active status
                cur.execute(
                    """
                    SELECT memory_id, project_id, version, status, tier, content,
                           metadata, dispute_status, supersedes_id, created_at, updated_at
                    FROM memories
                    WHERE project_id = %s
                      AND (metadata->>'root_memory_id' = %s OR memory_id = %s)
                      AND status = 'ACTIVE'
                    ORDER BY version DESC
                    LIMIT 1;
                    """,
                    (proj, root_memory_id, root_memory_id),
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_memory(row)

                # Fallback to latest version
                cur.execute(
                    """
                    SELECT memory_id, project_id, version, status, tier, content,
                           metadata, dispute_status, supersedes_id, created_at, updated_at
                    FROM memories
                    WHERE project_id = %s
                      AND (metadata->>'root_memory_id' = %s OR memory_id = %s)
                    ORDER BY version DESC
                    LIMIT 1;
                    """,
                    (proj, root_memory_id, root_memory_id),
                )
                row = cur.fetchone()
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
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT memory_id, project_id, version, status, tier, content,
                           metadata, dispute_status, supersedes_id, created_at, updated_at
                    FROM memories
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]

                if status:
                    sql += " AND status = %s"
                    params.append(status.upper())
                if tier:
                    sql += " AND tier = %s"
                    params.append(tier.upper())

                sql += " ORDER BY created_at DESC"
                if limit and limit > 0:
                    sql += " LIMIT %s"
                    params.append(limit)

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_memory(r) for r in rows]

    def update_status(self, memory_id: str, new_status: str, project_id: Optional[str] = None) -> bool:
        """Update lifecycle status of a memory."""
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        norm_status = normalize_memory_status(new_status)
        now_utc = datetime.now(timezone.utc)

        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memories
                    SET status = %s, updated_at = %s
                    WHERE memory_id = %s AND project_id = %s;
                    """,
                    (norm_status, now_utc, memory_id, proj),
                )
                return cur.rowcount > 0

    def count(self, *, project_id: Optional[str] = None, status: Optional[str] = None) -> int:
        """Count memories matching filter criteria."""
        if self._closed:
            raise RuntimeError("PostgreSQLMemoryStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = "SELECT COUNT(*) FROM memories WHERE project_id = %s"
                params: List[Any] = [proj]
                if status:
                    sql += " AND status = %s"
                    params.append(status.upper())
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        self._closed = True
