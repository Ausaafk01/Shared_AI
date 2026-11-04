"""
Immutable data models for the Vector Memory subsystem.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence
import uuid

from ai_brain.vectors.exceptions import DimensionMismatchError, VectorValidationError


def _freeze_container(data: Any) -> Any:
    """Recursively freeze dicts and lists for immutability."""
    if isinstance(data, (dict, MappingProxyType)):
        return MappingProxyType({k: _freeze_container(v) for k, v in data.items()})
    elif isinstance(data, (list, tuple)):
        return tuple(_freeze_container(v) for v in data)
    elif isinstance(data, (set, frozenset)):
        return frozenset(_freeze_container(v) for v in data)
    return data


def _unfreeze_container(data: Any) -> Any:
    """Recursively convert frozen containers back into standard dicts and lists."""
    if isinstance(data, (MappingProxyType, dict)):
        return {k: _unfreeze_container(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list, frozenset, set)):
        return [_unfreeze_container(v) for v in data]
    return data


def _json_serializer(obj: Any) -> Any:
    """JSON serializer helper for datetime and proxy objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (MappingProxyType, Mapping)):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass(frozen=True, kw_only=True)
class VectorRecord:
    """
    Immutable representation of an indexed vector embedding in storage.
    """
    record_id: str
    memory_id: str
    project_id: str
    version: int
    model_id: str
    dimensions: int
    embedding: tuple[float, ...]
    content: str
    status: str = "ACTIVE"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate vector record attributes and enforce deep immutability."""
        # 1. Validate record_id
        if not self.record_id or not isinstance(self.record_id, str) or not self.record_id.strip():
            raise VectorValidationError("record_id must be a non-empty string")
        object.__setattr__(self, "record_id", self.record_id.strip())

        # 2. Validate memory_id
        if not self.memory_id or not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise VectorValidationError("memory_id must be a non-empty string")
        object.__setattr__(self, "memory_id", self.memory_id.strip())

        # 3. Validate project_id
        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise VectorValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        # 4. Validate version
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise VectorValidationError("version must be a positive integer >= 1")

        # 5. Validate model_id
        if not self.model_id or not isinstance(self.model_id, str) or not self.model_id.strip():
            raise VectorValidationError("model_id must be a non-empty string")
        object.__setattr__(self, "model_id", self.model_id.strip())

        # 6. Validate dimensions
        if not isinstance(self.dimensions, int) or self.dimensions <= 0:
            raise VectorValidationError("dimensions must be a positive integer > 0")

        # 7. Validate embedding vector
        if not isinstance(self.embedding, (list, tuple)):
            raise VectorValidationError("embedding must be a sequence of floats")
        if len(self.embedding) != self.dimensions:
            raise DimensionMismatchError(
                f"Embedding dimensions mismatch: record specifies {self.dimensions}, got {len(self.embedding)}"
            )
        frozen_vec = tuple(float(x) for x in self.embedding)
        object.__setattr__(self, "embedding", frozen_vec)

        # 8. Validate content
        if not self.content or not isinstance(self.content, str) or not self.content.strip():
            raise VectorValidationError("content must be a non-empty string")
        object.__setattr__(self, "content", self.content.strip())

        # 9. Validate status
        if not self.status or not isinstance(self.status, str) or not self.status.strip():
            raise VectorValidationError("status must be a non-empty string")
        object.__setattr__(self, "status", self.status.strip().upper())

        # 10. Validate created_at
        if not isinstance(self.created_at, datetime):
            raise VectorValidationError("created_at must be a datetime object")
        c_utc = self.created_at.replace(tzinfo=timezone.utc) if self.created_at.tzinfo is None else self.created_at.astimezone(timezone.utc)
        object.__setattr__(self, "created_at", c_utc)

        # 11. Freeze metadata
        if self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        elif isinstance(self.metadata, (dict, MappingProxyType)):
            object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))
        else:
            raise VectorValidationError("metadata must be a dictionary or mapping")

    @classmethod
    def create(
        cls,
        *,
        record_id: Optional[str] = None,
        memory_id: str,
        project_id: str,
        version: int = 1,
        model_id: str,
        embedding: Sequence[float],
        content: str,
        status: str = "ACTIVE",
        metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> "VectorRecord":
        """Factory constructor for VectorRecord."""
        rid = record_id.strip() if record_id and isinstance(record_id, str) else str(uuid.uuid4())
        return cls(
            record_id=rid,
            memory_id=memory_id,
            project_id=project_id,
            version=version,
            model_id=model_id,
            dimensions=len(embedding),
            embedding=tuple(embedding),
            content=content,
            status=status,
            metadata=metadata or {},
            created_at=created_at or datetime.now(timezone.utc),
        )

    def with_updates(
        self,
        *,
        status: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "VectorRecord":
        """Return an immutable copy with updated status or metadata."""
        merged_meta = dict(_unfreeze_container(self.metadata))
        if metadata:
            merged_meta.update(_unfreeze_container(metadata))

        return VectorRecord(
            record_id=self.record_id,
            memory_id=self.memory_id,
            project_id=self.project_id,
            version=self.version,
            model_id=self.model_id,
            dimensions=self.dimensions,
            embedding=self.embedding,
            content=self.content,
            status=status if status is not None else self.status,
            metadata=merged_meta,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "record_id": self.record_id,
            "memory_id": self.memory_id,
            "project_id": self.project_id,
            "version": self.version,
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            "embedding": list(self.embedding),
            "content": self.content,
            "status": self.status,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorRecord":
        """Reconstruct VectorRecord from dictionary."""
        if not isinstance(data, (dict, Mapping)):
            raise VectorValidationError("Record data must be a dictionary")
        c_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc)
        return cls.create(
            record_id=data.get("record_id"),
            memory_id=data["memory_id"],
            project_id=data["project_id"],
            version=int(data.get("version", 1)),
            model_id=data["model_id"],
            embedding=data["embedding"],
            content=data["content"],
            status=data.get("status", "ACTIVE"),
            metadata=data.get("metadata", {}),
            created_at=c_at,
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=_json_serializer)

    @classmethod
    def from_json(cls, json_str: str) -> "VectorRecord":
        """Reconstruct from JSON string."""
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise VectorValidationError(f"Invalid JSON format for VectorRecord: {e}") from e
        return cls.from_dict(data)


@dataclass(frozen=True, kw_only=True)
class VectorMatch:
    """
    Representation of a retrieved memory match from a semantic similarity query.
    """
    memory_id: str
    score: float
    project_id: str
    version: int
    status: str
    content: str
    model_id: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", round(float(self.score), 4))
        if isinstance(self.metadata, (dict, MappingProxyType)):
            object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))


@dataclass(frozen=True, kw_only=True)
class VectorSearchResult:
    """
    Encapsulates the complete result of a vector similarity search query.
    """
    query: str
    matches: tuple[VectorMatch, ...]
    project_id: str
    model_id: str
    total_candidates: int = 0

    @property
    def top_match(self) -> Optional[VectorMatch]:
        """Convenience property for highest scoring match."""
        return self.matches[0] if self.matches else None

