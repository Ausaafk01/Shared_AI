"""Persistent PostgreSQL implementation of ML Feedback Storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, List, Optional, Sequence


from ai_brain.ml.feedback.models import FeedbackRecord, FeedbackSignal, FeedbackType
from ai_brain.ml.feedback.store import FeedbackStore
from ai_brain.storage.database import DatabaseConnectionManager, get_database_manager

logger = logging.getLogger(__name__)


def ensure_project_exists(db: DatabaseConnectionManager, project_id: str) -> None:
    """Ensure parent project exists in projects table to satisfy FK constraints."""
    db.ensure_project_exists(project_id)


class PostgreSQLFeedbackStore(FeedbackStore):
    """PostgreSQL-backed feedback storage engine for online ML refinement.

    Guarantees:
    - ACID transaction safety and strict tenant isolation via RLS and SET LOCAL.
    - Full round-trip preservation of predictions, corrections, feature snapshots, and signals.
    - Indexed lookups and filtering by project, feedback aspect/type, and model name.
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
            raise ValueError(
                "project_id must be provided or default_project_id set on PostgreSQLFeedbackStore."
            )
        return proj

    def record_feedback(self, record: FeedbackRecord) -> None:
        """Persist a FeedbackRecord in PostgreSQL."""
        if self._closed:
            raise RuntimeError("PostgreSQLFeedbackStore is closed.")

        proj_id = record.project_id
        self._known_projects.add(proj_id)
        ensure_project_exists(self.db, proj_id)

        created_utc = (
            record.created_at.replace(tzinfo=timezone.utc)
            if record.created_at.tzinfo is None
            else record.created_at.astimezone(timezone.utc)
        )

        score = 1.0 if record.signal == FeedbackSignal.POSITIVE else (
            0.0 if record.signal == FeedbackSignal.NEGATIVE else 0.5
        )

        meta_payload = {
            "record": record.to_dict(),
        }

        with self.db.tenant_context(proj_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback_records (
                        feedback_id,
                        project_id,
                        interaction_id,
                        score,
                        aspect,
                        notes,
                        metadata,
                        created_at
                    ) VALUES (
                        %s, %s, NULL, %s, %s, %s, %s::jsonb, %s
                    )
                    ON CONFLICT (feedback_id) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        score = EXCLUDED.score,
                        aspect = EXCLUDED.aspect,
                        notes = EXCLUDED.notes,
                        metadata = EXCLUDED.metadata;
                    """,
                    (
                        record.feedback_id,
                        proj_id,
                        score,
                        record.feedback_type.value,
                        record.notes or "",
                        json.dumps(meta_payload),
                        created_utc,
                    ),
                )

    def _row_to_record(self, row: Sequence[Any]) -> FeedbackRecord:
        feedback_id, project_id, interaction_id, score, aspect, notes, metadata_raw, created_at = row
        meta = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw or "{}")
        record_dict = meta.get("record")
        if record_dict:
            return FeedbackRecord.from_dict(record_dict)

        raise ValueError(f"Corrupted feedback record payload for '{feedback_id}'")

    def list_feedback(
        self,
        *,
        project_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        """List and filter feedback records."""
        if self._closed:
            raise RuntimeError("PostgreSQLFeedbackStore is closed.")

        proj = self._resolve_project(project_id)
        with self.db.tenant_context(proj) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT feedback_id, project_id, interaction_id, score,
                           aspect, notes, metadata, created_at
                    FROM feedback_records
                    WHERE project_id = %s
                """
                params: List[Any] = [proj]

                if feedback_type:
                    sql += " AND aspect = %s"
                    params.append(feedback_type.value)
                if model_name:
                    sql += " AND metadata->'record'->>'model_name' = %s"
                    params.append(model_name)

                sql += " ORDER BY created_at DESC"
                if limit and limit > 0:
                    sql += " LIMIT %s"
                    params.append(limit)

                cur.execute(sql, params)
                rows = cur.fetchall()
                return [self._row_to_record(r) for r in rows]

    def close(self) -> None:
        """Close storage resources (idempotent)."""
        self._closed = True

    def __enter__(self) -> "PostgreSQLFeedbackStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
