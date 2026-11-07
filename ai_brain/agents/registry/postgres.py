"""Persistent PostgreSQL implementation of Agent Registry Storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import psycopg

from ai_brain.agents.exceptions import AgentRegistrationError
from ai_brain.agents.models import AgentRecord, AgentStatus, AgentType
from ai_brain.agents.registry.base import AgentRegistryStore
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


def ensure_project_exists(db: DatabaseConnectionManager, project_id: str) -> None:
    """Ensure parent project exists in projects table to satisfy FK constraints."""
    db.ensure_project_exists(project_id)


class PostgreSQLAgentRegistryStore(AgentRegistryStore):
    """PostgreSQL-backed storage engine for registered agent identities.

    Guarantees:
    - ACID transaction safety and strict tenant isolation via RLS and SET LOCAL.
    - Full round-trip preservation of agent capabilities, configurations, and provider metadata.
    - Clean indexed lookups and filtering by agent status, type, and provider.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseConnectionManager] = None,
        default_project_id: str = "default",
    ) -> None:
        self.db = db_manager or get_database_manager()
        self.default_project_id = default_project_id
        self._closed = False
        self._known_projects: set[str] = {default_project_id}

    def _resolve_project(self, project_id: Optional[str]) -> str:
        proj = project_id or self.default_project_id
        if not proj:
            raise AgentRegistrationError(
                "project_id must be provided or default_project_id set on PostgreSQLAgentRegistryStore."
            )
        return proj

    def save(self, record: AgentRecord, project_id: Optional[str] = None) -> None:
        """Persist or update an AgentRecord in PostgreSQL."""
        if self._closed:
            raise AgentRegistrationError("PostgreSQLAgentRegistryStore is closed.")

        proj_id = self._resolve_project(project_id)
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        created_utc = (
            record.created_at.replace(tzinfo=timezone.utc)
            if record.created_at.tzinfo is None
            else record.created_at.astimezone(timezone.utc)
        )
        updated_utc = (
            record.updated_at.replace(tzinfo=timezone.utc)
            if record.updated_at.tzinfo is None
            else record.updated_at.astimezone(timezone.utc)
        )

        cap_payload = {
            "record": record.to_dict(),
            "capabilities": list(record.capabilities),
        }

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agents (
                        agent_id,
                        project_id,
                        name,
                        role,
                        capabilities,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s::jsonb, %s, %s, %s
                    )
                    ON CONFLICT (agent_id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        name = EXCLUDED.name,
                        role = EXCLUDED.role,
                        capabilities = EXCLUDED.capabilities,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at;
                    """,
                    (
                        record.agent_id,
                        proj_id,
                        record.name,
                        record.agent_type.value,
                        json.dumps(cap_payload),
                        record.status.value,
                        created_utc,
                        updated_utc,
                    ),
                )

    def _row_to_record(self, row: Sequence[Any]) -> AgentRecord:
        (
            agent_id,
            project_id,
            name,
            role,
            capabilities_raw,
            status,
            created_at,
            updated_at,
        ) = row

        caps = capabilities_raw if isinstance(capabilities_raw, dict) else json.loads(capabilities_raw or "{}")
        if isinstance(caps, dict) and "record" in caps:
            return AgentRecord.from_dict(caps["record"])
        elif isinstance(caps, list):
            # Fallback reconstruction
            return AgentRecord(
                agent_id=agent_id,
                name=name,
                provider="local",
                agent_type=AgentType(role) if role in AgentType._value2member_map_ else AgentType.GENERAL,
                capabilities=tuple(caps),
                status=AgentStatus(status) if status in AgentStatus._value2member_map_ else AgentStatus.ACTIVE,
                created_at=created_at,
                updated_at=updated_at,
            )

        raise AgentRegistrationError(f"Corrupted agent record payload for '{agent_id}'")

    def get(self, agent_id: str, project_id: Optional[str] = None) -> Optional[AgentRecord]:
        """Retrieve an AgentRecord by its agent_id."""
        if self._closed:
            raise AgentRegistrationError("PostgreSQLAgentRegistryStore is closed.")

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
                        SELECT agent_id, project_id, name, role, capabilities, status, created_at, updated_at
                        FROM agents
                        WHERE agent_id = %s AND project_id = %s;
                        """,
                        (agent_id, proj),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_record(row)

        return None

    def list(
        self,
        *,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        provider: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[AgentRecord]:
        """List and filter registered AgentRecords."""
        if self._closed:
            raise AgentRegistrationError("PostgreSQLAgentRegistryStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT agent_id, project_id, name, role, capabilities, status, created_at, updated_at
                    FROM agents
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]

                if status:
                    sql += " AND status = %s"
                    params.append(status.value)
                if agent_type:
                    sql += " AND role = %s"
                    params.append(agent_type.value)
                if provider:
                    sql += " AND capabilities->'record'->>'provider' = %s"
                    params.append(provider)

                sql += " ORDER BY created_at ASC;"

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_record(r) for r in rows]

    def delete(self, agent_id: str, project_id: Optional[str] = None) -> bool:
        """Remove an AgentRecord from persistent storage."""
        if self._closed:
            raise AgentRegistrationError("PostgreSQLAgentRegistryStore is closed.")

        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )

        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM agents WHERE agent_id = %s AND project_id = %s;",
                        (agent_id, proj),
                    )
                    if cur.rowcount > 0:
                        return True

        return False

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        self._closed = True
