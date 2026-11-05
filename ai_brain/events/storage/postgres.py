"""Persistent PostgreSQL implementation of the AI Brain Event Storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Union

import psycopg
from psycopg.errors import UniqueViolation

from ai_brain.events.exceptions import DuplicateEventError, StorageError
from ai_brain.events.models import Event
from ai_brain.events.storage.base import EventStore
from ai_brain.events.types import EventType, normalize_event_type
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


class PostgreSQLEventStore(EventStore):
    """PostgreSQL-backed event storage engine.

    Guarantees:
    - Events are stored in partitioned `events` table (partitioned by RANGE on timestamp).
    - Global uniqueness and routing enforced via `event_lookup` table.
    - Append-only immutability enforced via engine-level rules and triggers.
    - Tenant isolation enforced via PostgreSQL Row-Level Security and `SET LOCAL`.
    - Fully thread-safe via connection pooling.
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

    def store_event(self, event: Event) -> None:
        """Persist an event to storage in an append-only manner."""
        if self._closed:
            raise StorageError("EventStore is closed.")

        self._known_projects.add(event.project_id)

        ts_utc = (
            event.timestamp.replace(tzinfo=timezone.utc)
            if event.timestamp.tzinfo is None
            else event.timestamp.astimezone(timezone.utc)
        )
        created = getattr(event, "created_at", None) or ts_utc
        created_utc = (
            created.replace(tzinfo=timezone.utc)
            if created.tzinfo is None
            else created.astimezone(timezone.utc)
        )

        # First ensure parent project exists to satisfy FK constraint
        with self.db.connection() as admin_conn:
            with admin_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (project_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (project_id) DO NOTHING;
                    """,
                    (event.project_id, event.project_id),
                )
            admin_conn.commit()

        # Insert within tenant context
        try:
            with self.db.tenant_context(event.project_id) as conn:
                with conn.cursor() as cur:
                    # 1. Insert into event_lookup (enforces global event_id uniqueness)
                    try:
                        cur.execute(
                            """
                            INSERT INTO event_lookup (event_id, project_id, timestamp, event_type, created_at)
                            VALUES (%s, %s, %s, %s, %s);
                            """,
                            (event.event_id, event.project_id, ts_utc, str(event.event_type), created_utc),
                        )
                    except UniqueViolation as e:
                        raise DuplicateEventError(f"Event with ID '{event.event_id}' already exists.") from e

                    # 2. Insert into partitioned events table
                    payload_dict = event.to_dict() if hasattr(event, "to_dict") else {}
                    payload_json = json.dumps(payload_dict)
                    idemp_key = getattr(event, "idempotency_key", None)
                    cur.execute(
                        """
                        INSERT INTO events (
                            event_id, project_id, timestamp, event_type, agent_id, session_id,
                            idempotency_key, payload, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s);
                        """,
                        (
                            event.event_id,
                            event.project_id,
                            ts_utc,
                            str(event.event_type),
                            event.agent_id,
                            event.session_id,
                            idemp_key,
                            payload_json,
                            created_utc,
                        ),
                    )
        except DuplicateEventError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to store event '{event.event_id}': {e}") from e

    def _row_to_event(self, row: tuple[Any, ...]) -> Event:
        """Convert a database tuple into a validated Event model."""
        (
            event_id,
            project_id,
            timestamp,
            event_type_str,
            agent_id,
            session_id,
            idempotency_key,
            payload_data,
            created_at,
        ) = row

        payload = payload_data if isinstance(payload_data, dict) else json.loads(payload_data or "{}")

        return Event(
            event_id=event_id,
            event_type=normalize_event_type(event_type_str),
            timestamp=timestamp,
            project_id=project_id,
            session_id=session_id or payload.get("session_id", "default_session"),
            actor=payload.get("actor", "system"),
            agent_id=agent_id or payload.get("agent_id"),
            input=payload.get("input"),
            output=payload.get("output"),
            tool_used=payload.get("tool_used"),
            files_involved=tuple(payload.get("files_involved", ())),
            result=payload.get("result"),
            metadata=payload.get("metadata", {}),
        )

    def get_event_by_id(self, event_id: str, project_id: Optional[str] = None) -> Optional[Event]:
        """Retrieve an event by its unique ID across partitions."""
        if self._closed:
            raise StorageError("EventStore is closed.")

        candidate_projects = [project_id] if project_id else (
            [self.default_project_id] if self.default_project_id else list(self._known_projects)
        )

        for proj in candidate_projects:
            if not proj:
                continue
            with self.db.tenant_context(proj) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT project_id, timestamp FROM event_lookup WHERE event_id = %s AND project_id = %s;",
                        (event_id, proj),
                    )
                    lookup = cur.fetchone()
                    if not lookup:
                        continue
                    p_id, ts = lookup
                    cur.execute(
                        """
                        SELECT event_id, project_id, timestamp, event_type, agent_id, session_id,
                               idempotency_key, payload, created_at
                        FROM events
                        WHERE event_id = %s AND timestamp = %s AND project_id = %s;
                        """,
                        (event_id, ts, p_id),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_event(row)

        return None

    def get_events_by_project(self, project_id: str, ascending: bool = True) -> list[Event]:
        """Retrieve all events for a project ordered chronologically."""
        return self.filter_events(project_id=project_id, ascending=ascending)

    def get_events_by_agent(self, agent_id: str, ascending: bool = True) -> list[Event]:
        """Retrieve all events involving a specific agent."""
        return self.filter_events(agent_id=agent_id, ascending=ascending)

    def get_events_by_session(self, session_id: str, ascending: bool = True) -> list[Event]:
        """Retrieve all events in a session."""
        return self.filter_events(session_id=session_id, ascending=ascending)

    def get_events_by_type(self, event_type: EventType | str, ascending: bool = True) -> list[Event]:
        """Retrieve all events of a specific type."""
        return self.filter_events(event_type=event_type, ascending=ascending)

    def get_all_events(self, ascending: bool = True) -> list[Event]:
        """Retrieve all events across all projects ordered chronologically."""
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
        """Query events matching multiple criteria ordered chronologically."""
        if self._closed:
            raise StorageError("EventStore is closed.")

        conditions: List[str] = []
        params: List[Any] = []

        if project_id is not None:
            conditions.append("project_id = %s")
            params.append(project_id)
        if agent_id is not None:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
        if event_type is not None:
            conditions.append("event_type = %s")
            params.append(str(normalize_event_type(event_type)))
        if actor is not None:
            conditions.append("payload->>'actor' = %s")
            params.append(actor)
        if start_time is not None:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= %s")
            params.append(end_time)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_dir = "ASC" if ascending else "DESC"
        order_clause = f"ORDER BY timestamp {order_dir}, event_id {order_dir}"

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {int(limit)} OFFSET {int(offset)}"
        elif offset > 0:
            limit_clause = f"OFFSET {int(offset)}"

        sql = f"""
            SELECT event_id, project_id, timestamp, event_type, agent_id, session_id,
                   idempotency_key, payload, created_at
            FROM events
            {where_clause}
            {order_clause}
            {limit_clause};
        """

        # If project_id is specified, use tenant context
        ctx = self.db.tenant_context(project_id) if project_id else self.db.connection()
        with ctx as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
                return [self._row_to_event(row) for row in rows]

    def count(
        self,
        *,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[EventType | str] = None,
    ) -> int:
        """Count the number of stored events matching the given criteria."""
        if self._closed:
            raise StorageError("EventStore is closed.")

        conditions: List[str] = []
        params: List[Any] = []

        if project_id is not None:
            conditions.append("project_id = %s")
            params.append(project_id)
        if agent_id is not None:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
        if event_type is not None:
            conditions.append("event_type = %s")
            params.append(str(normalize_event_type(event_type)))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT COUNT(*) FROM events {where_clause};"

        ctx = self.db.tenant_context(project_id) if project_id else self.db.connection()
        with ctx as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return cur.fetchone()[0]

    def close(self) -> None:
        """Close storage resources."""
        self._closed = True
