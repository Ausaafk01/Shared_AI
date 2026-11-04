"""
Durable storage for ML feedback records.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from ai_brain.ml.feedback.models import FeedbackRecord, FeedbackSignal, FeedbackType


class FeedbackStore(ABC):
    """Abstract store for feedback records."""

    @abstractmethod
    def record_feedback(self, record: FeedbackRecord) -> None:
        pass

    @abstractmethod
    def list_feedback(
        self,
        *,
        project_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SQLiteFeedbackStore(FeedbackStore):
    """ACID SQLite store for ML feedback with project isolation."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    feedback_type TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    agent_id TEXT,
                    session_id TEXT,
                    memory_ids_json TEXT NOT NULL,
                    predicted_value_json TEXT,
                    corrected_value_json TEXT,
                    features_snapshot_json TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fb_proj_type ON feedback_records(project_id, feedback_type)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fb_model ON feedback_records(model_name, model_version)"
            )

    def record_feedback(self, record: FeedbackRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO feedback_records (
                    feedback_id, feedback_type, signal, project_id, model_name, model_version,
                    created_at, agent_id, session_id, memory_ids_json, predicted_value_json,
                    corrected_value_json, features_snapshot_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.feedback_id,
                    record.feedback_type.value,
                    record.signal.value,
                    record.project_id,
                    record.model_name,
                    record.model_version,
                    record.created_at.isoformat(),
                    record.agent_id,
                    record.session_id,
                    json.dumps(list(record.memory_ids)),
                    json.dumps(record.predicted_value),
                    json.dumps(record.corrected_value),
                    json.dumps(dict(record.features_snapshot)),
                    record.notes,
                ),
            )

    def list_feedback(
        self,
        *,
        project_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        query = "SELECT * FROM feedback_records WHERE 1=1"
        params: List[Any] = []
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.execute(query, params)
        rows = cur.fetchall()
        results = []
        for r in rows:
            data = {
                "feedback_id": r["feedback_id"],
                "feedback_type": r["feedback_type"],
                "signal": r["signal"],
                "project_id": r["project_id"],
                "model_name": r["model_name"],
                "model_version": r["model_version"],
                "created_at": r["created_at"],
                "agent_id": r["agent_id"],
                "session_id": r["session_id"],
                "memory_ids": tuple(json.loads(r["memory_ids_json"])),
                "predicted_value": json.loads(r["predicted_value_json"]) if r["predicted_value_json"] else None,
                "corrected_value": json.loads(r["corrected_value_json"]) if r["corrected_value_json"] else None,
                "features_snapshot": json.loads(r["features_snapshot_json"]),
                "notes": r["notes"] or "",
            }
            results.append(FeedbackRecord.from_dict(data))
        return results

    def close(self) -> None:
        self._conn.close()
