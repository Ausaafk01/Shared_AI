"""
Feedback record schemas for AI Brain ML layer.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import uuid


class FeedbackType(StrEnum):
    RETRIEVAL_RELEVANCE = "RETRIEVAL_RELEVANCE"
    DUPLICATE_DETECTION = "DUPLICATE_DETECTION"
    CONFLICT_DETECTION = "CONFLICT_DETECTION"
    IMPORTANCE_PREDICTION = "IMPORTANCE_PREDICTION"
    STALENESS_PREDICTION = "STALENESS_PREDICTION"


class FeedbackSignal(StrEnum):
    POSITIVE = "POSITIVE"      # Model prediction was accurate / useful
    NEGATIVE = "NEGATIVE"      # Model prediction was inaccurate / misleading
    CORRECTION = "CORRECTION"  # Explicit ground-truth correction supplied


@dataclass(frozen=True)
class FeedbackRecord:
    """Immutable record capturing feedback signal on an ML prediction with full provenance."""
    feedback_id: str
    feedback_type: FeedbackType
    signal: FeedbackSignal
    project_id: str
    model_name: str
    model_version: str
    created_at: datetime
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    memory_ids: Tuple[str, ...] = field(default_factory=tuple)
    predicted_value: Optional[Union[float, str, bool]] = None
    corrected_value: Optional[Union[float, str, bool]] = None
    features_snapshot: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "signal": self.signal.value,
            "project_id": self.project_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "memory_ids": list(self.memory_ids),
            "predicted_value": self.predicted_value,
            "corrected_value": self.corrected_value,
            "features_snapshot": dict(self.features_snapshot),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeedbackRecord":
        return cls(
            feedback_id=str(data["feedback_id"]),
            feedback_type=FeedbackType(data["feedback_type"]),
            signal=FeedbackSignal(data["signal"]),
            project_id=str(data["project_id"]),
            model_name=str(data["model_name"]),
            model_version=str(data["model_version"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            memory_ids=tuple(data.get("memory_ids", ())),
            predicted_value=data.get("predicted_value"),
            corrected_value=data.get("corrected_value"),
            features_snapshot=MappingProxyType(dict(data.get("features_snapshot", {}))),
            notes=str(data.get("notes", "")),
        )
