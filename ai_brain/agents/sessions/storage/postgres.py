"""Persistent PostgreSQL implementation of Session Storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, List, Optional, Sequence


from ai_brain.agents.exceptions import SessionError
from ai_brain.agents.sessions.models import SessionRecord, SessionStatus
from ai_brain.agents.sessions.storage.base import SessionStore
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


def ensure_project_exists(db: DatabaseConnectionManager, project_id: str) -> None:
    """Ensure parent project exists in projects table to satisfy FK constraints."""
    db.ensure_project_exists(project_id)


class PostgreSQLSessionStore(SessionStore):
    """PostgreSQL-backed storage for multi-agent interactive sessions.

    Guarantees:
    - ACID transaction safety and strict tenant isolation via RLS and SET LOCAL.
    - Full round-trip preservation of session metadata, participants, history, and status.
    - Graceful foreign key handling for primary_agent_id references.
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
            raise SessionError(
                "project_id must be provided or default_project_id set on PostgreSQLSessionStore."
            )
        return proj

    def save(self, session: SessionRecord) -> None:
        """Persist or update a SessionRecord in PostgreSQL."""
        if self._closed:
            raise SessionError("PostgreSQLSessionStore is closed.")

        proj_id = session.project_id
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        created_utc = (
            session.created_at.replace(tzinfo=timezone.utc)
            if session.created_at.tzinfo is None
            else session.created_at.astimezone(timezone.utc)
        )
        updated_utc = (
            session.updated_at.replace(tzinfo=timezone.utc)
            if session.updated_at.tzinfo is None
            else session.updated_at.astimezone(timezone.utc)
        )

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                valid_agent_id: Optional[str] = None
                if session.current_agent_id:
                    cur.execute(
                        "SELECT 1 FROM agents WHERE agent_id = %s AND project_id = %s;",
                        (session.current_agent_id, proj_id),
                    )
                    if cur.fetchone():
                        valid_agent_id = session.current_agent_id

                meta_payload = {
                    "session": session.to_dict(),
                }

                stat_val = (
                    session.status.value
                    if isinstance(session.status, SessionStatus)
                    else str(session.status).upper()
                )

                cur.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        project_id,
                        primary_agent_id,
                        status,
                        summary,
                        metadata,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb, %s, %s
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        primary_agent_id = EXCLUDED.primary_agent_id,
                        status = EXCLUDED.status,
                        summary = EXCLUDED.summary,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at;
                    """,
                    (
                        session.session_id,
                        proj_id,
                        valid_agent_id,
                        stat_val,
                        session.title,
                        json.dumps(meta_payload),
                        created_utc,
                        updated_utc,
                    ),
                )

    def _row_to_session(self, row: Sequence[Any]) -> SessionRecord:
        (
            session_id,
            project_id,
            primary_agent_id,
            status,
            summary,
            metadata_raw,
            created_at,
            updated_at,
        ) = row

        meta = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw or "{}")
        session_dict = meta.get("session")
        if session_dict:
            return SessionRecord.from_dict(session_dict)

        raise SessionError(f"Corrupted session record payload for '{session_id}'")

    def get(self, session_id: str, project_id: Optional[str] = None) -> Optional[SessionRecord]:
        """Retrieve a session by session_id."""
        if self._closed:
            raise SessionError("PostgreSQLSessionStore is closed.")

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
                        SELECT session_id, project_id, primary_agent_id, status,
                               summary, metadata, created_at, updated_at
                        FROM sessions
                        WHERE session_id = %s AND project_id = %s;
                        """,
                        (session_id, proj),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_session(row)

        return None

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
    ) -> List[SessionRecord]:
        """List and filter session records."""
        if self._closed:
            raise SessionError("PostgreSQLSessionStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT session_id, project_id, primary_agent_id, status,
                           summary, metadata, created_at, updated_at
                    FROM sessions
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]

                if agent_id:
                    sql += " AND (primary_agent_id = %s OR metadata->'session'->>'current_agent_id' = %s)"
                    params.extend([agent_id, agent_id])
                if status:
                    stat_val = status.value if isinstance(status, SessionStatus) else str(status).upper()
                    sql += " AND status = %s"
                    params.append(stat_val)

                sql += " ORDER BY created_at DESC;"

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_session(r) for r in rows]

    def delete(self, session_id: str, project_id: Optional[str] = None) -> bool:
        """Delete a session record."""
        if self._closed:
            raise SessionError("PostgreSQLSessionStore is closed.")

        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )

        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM sessions WHERE session_id = %s AND project_id = %s;",
                        (session_id, proj),
                    )
                    if cur.rowcount > 0:
                        return True

        return False

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        self._closed = True
