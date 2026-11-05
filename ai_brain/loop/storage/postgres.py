"""Persistent PostgreSQL implementation of Brain Loop Interaction Storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import psycopg

from ai_brain.loop.exceptions import BrainLoopError
from ai_brain.loop.models import InteractionRecord, InteractionStatus
from ai_brain.loop.storage.base import InteractionStore
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


def ensure_project_exists(db: DatabaseConnectionManager, project_id: str) -> None:
    """Ensure parent project exists in projects table to satisfy FK constraints."""
    if not project_id:
        return
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (project_id, name)
                VALUES (%s, %s)
                ON CONFLICT (project_id) DO NOTHING;
                """,
                (project_id, project_id),
            )
        conn.commit()


class PostgreSQLInteractionStore(InteractionStore):
    """PostgreSQL-backed interaction store for Brain Loop.

    Guarantees:
    - ACID transaction safety and strict tenant isolation via RLS and SET LOCAL.
    - Idempotency key lookups for deduplicating agent interaction loops.
    - Full round-trip preservation of input, context, response, and completion results.
    - Graceful foreign key handling for session_id and agent_id references.
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
            raise BrainLoopError(
                "project_id must be provided or default_project_id set on PostgreSQLInteractionStore."
            )
        return proj

    def save(self, record: InteractionRecord) -> None:
        """Persist or update an InteractionRecord in PostgreSQL."""
        if self._closed:
            raise BrainLoopError("PostgreSQLInteractionStore is closed.")

        proj_id = record.project_id
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        created_utc = (
            record.created_at.replace(tzinfo=timezone.utc)
            if record.created_at.tzinfo is None
            else record.created_at.astimezone(timezone.utc)
        )

        input_text = record.input.content if hasattr(record.input, "content") else str(record.input)
        output_text = record.response.content if (record.response and hasattr(record.response, "content")) else None

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                # Check optional FKs to avoid referential integrity violation if agent/session don't exist yet
                valid_session_id: Optional[str] = None
                if record.session_id:
                    cur.execute(
                        "SELECT 1 FROM sessions WHERE session_id = %s AND project_id = %s;",
                        (record.session_id, proj_id),
                    )
                    if cur.fetchone():
                        valid_session_id = record.session_id

                valid_agent_id: Optional[str] = None
                if record.agent_id:
                    cur.execute(
                        "SELECT 1 FROM agents WHERE agent_id = %s AND project_id = %s;",
                        (record.agent_id, proj_id),
                    )
                    if cur.fetchone():
                        valid_agent_id = record.agent_id

                metrics_payload = {
                    "record": record.to_dict(),
                }

                cur.execute(
                    """
                    INSERT INTO interactions (
                        interaction_id,
                        project_id,
                        session_id,
                        agent_id,
                        idempotency_key,
                        status,
                        input_text,
                        output_text,
                        decision_records,
                        metrics,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, '[]'::jsonb, %s::jsonb, %s
                    )
                    ON CONFLICT (interaction_id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        session_id = EXCLUDED.session_id,
                        agent_id = EXCLUDED.agent_id,
                        idempotency_key = EXCLUDED.idempotency_key,
                        status = EXCLUDED.status,
                        input_text = EXCLUDED.input_text,
                        output_text = EXCLUDED.output_text,
                        metrics = EXCLUDED.metrics;
                    """,
                    (
                        record.interaction_id,
                        proj_id,
                        valid_session_id,
                        valid_agent_id,
                        record.idempotency_key,
                        record.status.value if isinstance(record.status, InteractionStatus) else str(record.status),
                        input_text,
                        output_text,
                        json.dumps(metrics_payload),
                        created_utc,
                    ),
                )

    def _row_to_record(self, row: Sequence[Any]) -> InteractionRecord:
        (
            interaction_id,
            project_id,
            session_id,
            agent_id,
            idempotency_key,
            status,
            input_text,
            output_text,
            decision_records,
            metrics_raw,
            created_at,
        ) = row

        metrics = metrics_raw if isinstance(metrics_raw, dict) else json.loads(metrics_raw or "{}")
        record_dict = metrics.get("record")
        if record_dict:
            return InteractionRecord.from_dict(record_dict)

        raise BrainLoopError(f"Corrupted interaction record payload for '{interaction_id}'")

    def get(self, interaction_id: str, project_id: Optional[str] = None) -> Optional[InteractionRecord]:
        """Retrieve an interaction record by interaction_id."""
        if self._closed:
            raise BrainLoopError("PostgreSQLInteractionStore is closed.")

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
                        SELECT interaction_id, project_id, session_id, agent_id,
                               idempotency_key, status, input_text, output_text,
                               decision_records, metrics, created_at
                        FROM interactions
                        WHERE interaction_id = %s AND project_id = %s;
                        """,
                        (interaction_id, proj),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_record(row)

        return None

    def get_by_idempotency_key(
        self, idempotency_key: str, project_id: Optional[str] = None
    ) -> Optional[InteractionRecord]:
        """Retrieve an interaction record by its idempotency_key."""
        if self._closed:
            raise BrainLoopError("PostgreSQLInteractionStore is closed.")

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
                        SELECT interaction_id, project_id, session_id, agent_id,
                               idempotency_key, status, input_text, output_text,
                               decision_records, metrics, created_at
                        FROM interactions
                        WHERE idempotency_key = %s AND project_id = %s;
                        """,
                        (idempotency_key, proj),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_record(row)

        return None

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[Union[str, InteractionStatus]] = None,
    ) -> List[InteractionRecord]:
        """List and filter interaction records."""
        if self._closed:
            raise BrainLoopError("PostgreSQLInteractionStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT interaction_id, project_id, session_id, agent_id,
                           idempotency_key, status, input_text, output_text,
                           decision_records, metrics, created_at
                    FROM interactions
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]

                if session_id:
                    sql += " AND (session_id = %s OR metrics->'record'->>'session_id' = %s)"
                    params.extend([session_id, session_id])
                if status:
                    stat_val = status.value if isinstance(status, InteractionStatus) else str(status)
                    sql += " AND status = %s"
                    params.append(stat_val)

                sql += " ORDER BY created_at DESC;"

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_record(r) for r in rows]

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        self._closed = True
