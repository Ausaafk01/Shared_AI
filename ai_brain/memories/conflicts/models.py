"""
Immutable MemoryConflict model for AI Brain Conflict Management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from ai_brain.memories.exceptions import ConflictValidationError
from .types import (
    ConflictStatus,
    ConflictType,
    normalize_conflict_status,
    normalize_conflict_type,
    validate_status_transition,
)


def _json_serializer(obj: Any) -> Any:
    """Helper serializer for JSON conversion."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (ConflictType, ConflictStatus)):
        return obj.value
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (MappingProxyType, Mapping)):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


from ai_brain.common.freeze import _freeze_container, _unfreeze_container


@dataclass(frozen=True, kw_only=True)
class MemoryConflict:
    """
    Immutable representation of an identified conflict between Memories.
    
    All attributes are read-only once instantiated. Attempting to modify
    or delete attributes raises a FrozenInstanceError (AttributeError).
    """
    conflict_id: str
    project_id: str
    memory_ids: tuple[str, ...]
    conflict_type: str
    description: str
    detected_at: datetime
    status: str = ConflictStatus.DETECTED.value
    resolution: Optional[Union[str, Mapping[str, Any]]] = None
    evidence: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate all fields and enforce immutability."""
        # 1. Validate conflict_id
        if not self.conflict_id or not isinstance(self.conflict_id, str) or not self.conflict_id.strip():
            raise ConflictValidationError("conflict_id must be a non-empty string")
        object.__setattr__(self, "conflict_id", self.conflict_id.strip())

        # 2. Validate project_id
        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ConflictValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        # 3. Validate memory_ids
        if not self.memory_ids or not isinstance(self.memory_ids, (list, tuple, set)):
            raise ConflictValidationError("memory_ids must be a sequence of at least 2 memory IDs")
        cleaned_ids = []
        for m_id in self.memory_ids:
            if not isinstance(m_id, str) or not m_id.strip():
                raise ConflictValidationError(f"Invalid memory_id in conflict: {m_id!r}")
            cleaned_ids.append(m_id.strip())
        if len(cleaned_ids) < 2:
            raise ConflictValidationError("A conflict must involve at least 2 distinct memories")
        object.__setattr__(self, "memory_ids", tuple(cleaned_ids))

        # 4. Validate conflict_type
        norm_type = normalize_conflict_type(self.conflict_type)
        object.__setattr__(self, "conflict_type", norm_type)

        # 5. Validate description
        if not self.description or not isinstance(self.description, str) or not self.description.strip():
            raise ConflictValidationError("description must be a non-empty string")
        object.__setattr__(self, "description", self.description.strip())

        # 6. Validate detected_at
        if not isinstance(self.detected_at, datetime):
            raise ConflictValidationError(
                f"detected_at must be a datetime object, got {type(self.detected_at).__name__}"
            )
        detected_utc = (
            self.detected_at.replace(tzinfo=timezone.utc)
            if self.detected_at.tzinfo is None
            else self.detected_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "detected_at", detected_utc)

        # 7. Validate status
        norm_status = normalize_conflict_status(self.status)
        object.__setattr__(self, "status", norm_status)

        # 8. Validate confidence [0.0, 1.0]
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ConflictValidationError(f"confidence must be a number between 0.0 and 1.0, got {self.confidence!r}")
        conf_float = float(self.confidence)
        if not (0.0 <= conf_float <= 1.0):
            raise ConflictValidationError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        object.__setattr__(self, "confidence", round(conf_float, 4))

        # 9. Freeze evidence
        if self.evidence is None:
            object.__setattr__(self, "evidence", MappingProxyType({}))
        elif isinstance(self.evidence, (dict, MappingProxyType)):
            object.__setattr__(self, "evidence", _freeze_container(_unfreeze_container(self.evidence)))
        else:
            raise ConflictValidationError("evidence must be a dictionary or mapping")

        # 10. Freeze metadata
        if self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        elif isinstance(self.metadata, (dict, MappingProxyType)):
            object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))
        else:
            raise ConflictValidationError("metadata must be a dictionary or mapping")

        # 11. Freeze resolution if dict
        if isinstance(self.resolution, (dict, MappingProxyType)):
            object.__setattr__(self, "resolution", _freeze_container(_unfreeze_container(self.resolution)))

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        memory_ids: Sequence[str],
        conflict_type: str | ConflictType,
        description: str,
        conflict_id: Optional[str] = None,
        detected_at: Optional[datetime] = None,
        status: str | ConflictStatus = ConflictStatus.DETECTED,
        resolution: Optional[Union[str, Mapping[str, Any]]] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        confidence: float = 1.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MemoryConflict":
        """Factory method to construct an immutable validated MemoryConflict."""
        return cls(
            conflict_id=conflict_id or str(uuid.uuid4()),
            project_id=project_id,
            memory_ids=tuple(memory_ids),
            conflict_type=conflict_type,
            description=description,
            detected_at=detected_at or datetime.now(timezone.utc),
            status=status,
            resolution=resolution,
            evidence=evidence or {},
            confidence=confidence,
            metadata=metadata or {},
        )

    def with_status(
        self,
        new_status: str | ConflictStatus,
        *,
        resolution: Optional[Union[str, Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MemoryConflict":
        """
        Transition conflict to a new status with validation against allowed state transitions.
        Returns a new immutable MemoryConflict instance.
        """
        validate_status_transition(self.status, new_status)
        merged_meta = dict(self.metadata)
        if metadata:
            merged_meta.update(metadata)

        return MemoryConflict(
            conflict_id=self.conflict_id,
            project_id=self.project_id,
            memory_ids=self.memory_ids,
            conflict_type=self.conflict_type,
            description=self.description,
            detected_at=self.detected_at,
            status=new_status,
            resolution=resolution if resolution is not None else self.resolution,
            evidence=self.evidence,
            confidence=self.confidence,
            metadata=merged_meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert conflict into standard dictionary representation."""
        return {
            "conflict_id": self.conflict_id,
            "project_id": self.project_id,
            "memory_ids": list(self.memory_ids),
            "conflict_type": self.conflict_type,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "status": self.status,
            "resolution": _unfreeze_container(self.resolution) if isinstance(self.resolution, (dict, MappingProxyType)) else self.resolution,
            "evidence": _unfreeze_container(self.evidence),
            "confidence": self.confidence,
            "metadata": _unfreeze_container(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConflict":
        """Reconstruct a MemoryConflict from a dictionary."""
        if not isinstance(data, dict):
            raise ConflictValidationError(f"Expected dict, got {type(data).__name__}")
        raw = dict(data)
        det = raw.get("detected_at")
        if isinstance(det, str):
            try:
                norm_d = det.replace("Z", "+00:00")
                raw["detected_at"] = datetime.fromisoformat(norm_d)
            except Exception as e:
                raise ConflictValidationError(f"Invalid detected_at format '{det}': {e}") from e

        if "memory_ids" in raw and raw["memory_ids"] is not None:
            raw["memory_ids"] = tuple(raw["memory_ids"])

        try:
            return cls(**raw)
        except TypeError as e:
            raise ConflictValidationError(f"Invalid fields in conflict dict: {e}") from e

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize conflict to JSON string."""
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryConflict":
        """Deserialize conflict from JSON string."""
        if not isinstance(json_str, (str, bytes)):
            raise ConflictValidationError(f"Expected JSON string, got {type(json_str).__name__}")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ConflictValidationError(f"Malformed JSON: {e}") from e
        return cls.from_dict(data)
