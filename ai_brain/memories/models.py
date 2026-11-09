"""
Memory model for the AI Brain Memory System.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from .exceptions import MemoryValidationError
from .types import (
    MemoryStatus,
    MemoryType,
    normalize_memory_status,
    normalize_memory_type,
)


def _json_serializer(obj: Any) -> Any:
    """Helper serializer for JSON conversion of non-standard objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (MemoryType, MemoryStatus)):
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
class Memory:
    """
    Immutable representation of established knowledge in the AI Brain.
    
    A Memory represents useful facts, decisions, preferences, or context
    derived from events and maintained across multiple agents.
    
    All attributes are read-only once created. Attempting to assign or delete
    attributes will raise a FrozenInstanceError (AttributeError).
    """
    memory_id: str
    memory_type: str
    content: str
    project_id: str
    created_at: datetime
    updated_at: datetime
    source_event_ids: tuple[str, ...] = field(default_factory=tuple)
    source_agent_ids: tuple[str, ...] = field(default_factory=tuple)
    importance: float = 0.5
    confidence: float = 1.0
    status: str = MemoryStatus.ACTIVE.value
    relationships: Union[Mapping[str, Any], tuple[Any, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    root_memory_id: Optional[str] = None
    parent_memory_id: Optional[str] = None
    version: int = 1

    def __post_init__(self) -> None:
        """Validate all fields and enforce deep immutability."""
        # 1. Validate memory_id
        if not self.memory_id or not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise MemoryValidationError("memory_id must be a non-empty string")
        object.__setattr__(self, "memory_id", self.memory_id.strip())

        # 2. Validate and normalize memory_type
        try:
            normalized_type = normalize_memory_type(self.memory_type)
            object.__setattr__(self, "memory_type", normalized_type)
        except ValueError as e:
            raise MemoryValidationError(str(e)) from e

        # 3. Validate content
        if not self.content or not isinstance(self.content, str) or not self.content.strip():
            raise MemoryValidationError("content must be a non-empty string")
        object.__setattr__(self, "content", self.content.strip())

        # 4. Validate project_id
        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise MemoryValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        # 5. Validate created_at
        if not isinstance(self.created_at, datetime):
            raise MemoryValidationError(
                f"created_at must be a datetime object, got {type(self.created_at).__name__}"
            )
        created_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", created_utc)

        # 6. Validate updated_at
        if not isinstance(self.updated_at, datetime):
            raise MemoryValidationError(
                f"updated_at must be a datetime object, got {type(self.updated_at).__name__}"
            )
        updated_utc = (
            self.updated_at.replace(tzinfo=timezone.utc)
            if self.updated_at.tzinfo is None
            else self.updated_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "updated_at", updated_utc)

        # 7. Check timestamp consistency
        if updated_utc < created_utc:
            raise MemoryValidationError(
                f"updated_at ({updated_utc.isoformat()}) cannot be earlier than created_at ({created_utc.isoformat()})"
            )

        # 8. Validate importance range [0.0, 1.0]
        if not isinstance(self.importance, (int, float)) or isinstance(self.importance, bool):
            raise MemoryValidationError(
                f"importance must be a float or int between 0.0 and 1.0, got {type(self.importance).__name__}"
            )
        importance_float = float(self.importance)
        if not (0.0 <= importance_float <= 1.0):
            raise MemoryValidationError(
                f"importance must be between 0.0 and 1.0, got {self.importance}"
            )
        object.__setattr__(self, "importance", round(importance_float, 4))

        # 9. Validate confidence range [0.0, 1.0]
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise MemoryValidationError(
                f"confidence must be a float or int between 0.0 and 1.0, got {type(self.confidence).__name__}"
            )
        confidence_float = float(self.confidence)
        if not (0.0 <= confidence_float <= 1.0):
            raise MemoryValidationError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        object.__setattr__(self, "confidence", round(confidence_float, 4))

        # 10. Validate and normalize status
        try:
            normalized_status = normalize_memory_status(self.status)
            object.__setattr__(self, "status", normalized_status)
        except ValueError as e:
            raise MemoryValidationError(str(e)) from e

        # 11. Normalize source_event_ids
        if self.source_event_ids is None:
            object.__setattr__(self, "source_event_ids", ())
        elif isinstance(self.source_event_ids, (list, tuple, set)):
            event_ids = []
            for item in self.source_event_ids:
                if not isinstance(item, str) or not item.strip():
                    raise MemoryValidationError(
                        f"All source_event_ids must be non-empty strings, got {item!r}"
                    )
                event_ids.append(item.strip())
            object.__setattr__(self, "source_event_ids", tuple(event_ids))
        else:
            raise MemoryValidationError("source_event_ids must be a sequence of strings")

        # 12. Normalize source_agent_ids
        if self.source_agent_ids is None:
            object.__setattr__(self, "source_agent_ids", ())
        elif isinstance(self.source_agent_ids, (list, tuple, set)):
            agent_ids = []
            for item in self.source_agent_ids:
                if not isinstance(item, str) or not item.strip():
                    raise MemoryValidationError(
                        f"All source_agent_ids must be non-empty strings, got {item!r}"
                    )
                agent_ids.append(item.strip())
            object.__setattr__(self, "source_agent_ids", tuple(agent_ids))
        else:
            raise MemoryValidationError("source_agent_ids must be a sequence of strings")

        # 13. Deep freeze relationships
        if self.relationships is None:
            object.__setattr__(self, "relationships", MappingProxyType({}))
        elif isinstance(self.relationships, (dict, MappingProxyType)):
            frozen_rel = _freeze_container(deepcopy(dict(self.relationships)))
            object.__setattr__(self, "relationships", frozen_rel)
        elif isinstance(self.relationships, (list, tuple)):
            frozen_rel = _freeze_container(deepcopy(list(self.relationships)))
            object.__setattr__(self, "relationships", frozen_rel)
        else:
            raise MemoryValidationError(
                "relationships must be a dictionary or sequence of relationships"
            )

        # 14. Deep freeze metadata
        if self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        elif isinstance(self.metadata, (dict, MappingProxyType)):
            frozen_meta = _freeze_container(_unfreeze_container(self.metadata))
            object.__setattr__(self, "metadata", frozen_meta)
        else:
            raise MemoryValidationError("metadata must be a dictionary or mapping")

        # 15. Validate versioning fields
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise MemoryValidationError(
                f"version must be a positive integer >= 1, got {self.version!r}"
            )

        if self.root_memory_id is None or (isinstance(self.root_memory_id, str) and not self.root_memory_id.strip()):
            object.__setattr__(self, "root_memory_id", self.memory_id)
        elif isinstance(self.root_memory_id, str):
            object.__setattr__(self, "root_memory_id", self.root_memory_id.strip())
        else:
            raise MemoryValidationError("root_memory_id must be a string")

        if self.parent_memory_id is not None:
            if not isinstance(self.parent_memory_id, str) or not self.parent_memory_id.strip():
                raise MemoryValidationError("parent_memory_id must be a non-empty string if provided")
            object.__setattr__(self, "parent_memory_id", self.parent_memory_id.strip())

    @classmethod
    def create(
        cls,
        *,
        memory_type: str | MemoryType,
        content: str,
        project_id: str,
        memory_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        source_event_ids: Optional[Sequence[str]] = None,
        source_agent_ids: Optional[Sequence[str]] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        status: str | MemoryStatus = MemoryStatus.ACTIVE,
        relationships: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        root_memory_id: Optional[str] = None,
        parent_memory_id: Optional[str] = None,
        version: int = 1,
    ) -> "Memory":
        """
        Factory method to create a validated immutable Memory instance.
        
        Automatically generates a UUID v4 memory_id if not supplied,
        and defaults timestamps to current UTC time.
        """
        now = datetime.now(timezone.utc)
        effective_created = created_at or now
        effective_updated = updated_at or effective_created

        return cls(
            memory_id=memory_id or str(uuid.uuid4()),
            memory_type=memory_type,
            content=content,
            project_id=project_id,
            created_at=effective_created,
            updated_at=effective_updated,
            source_event_ids=tuple(source_event_ids or ()),
            source_agent_ids=tuple(source_agent_ids or ()),
            importance=importance,
            confidence=confidence,
            status=status,
            relationships=relationships if relationships is not None else {},
            metadata=metadata or {},
            root_memory_id=root_memory_id,
            parent_memory_id=parent_memory_id,
            version=version,
        )

    def with_updates(
        self,
        *,
        content: Optional[str] = None,
        status: Optional[str | MemoryStatus] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        updated_at: Optional[datetime] = None,
        source_event_ids: Optional[Sequence[str]] = None,
        source_agent_ids: Optional[Sequence[str]] = None,
        relationships: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        parent_memory_id: Optional[str] = None,
        version: Optional[int] = None,
        memory_id: Optional[str] = None,
        root_memory_id: Optional[str] = None,
    ) -> "Memory":
        """
        Create a new immutable Memory derived from this instance with specific field overrides.
        """
        new_source_events = (
            tuple(source_event_ids) if source_event_ids is not None else self.source_event_ids
        )
        new_source_agents = (
            tuple(source_agent_ids) if source_agent_ids is not None else self.source_agent_ids
        )

        effective_updated = updated_at or max(self.updated_at, datetime.now(timezone.utc))
        if self.created_at > effective_updated:
            effective_updated = self.created_at

        return Memory(
            memory_id=memory_id or str(uuid.uuid4()),
            memory_type=self.memory_type,
            content=content if content is not None else self.content,
            project_id=self.project_id,
            created_at=self.created_at,
            updated_at=effective_updated,
            source_event_ids=new_source_events,
            source_agent_ids=new_source_agents,
            importance=importance if importance is not None else self.importance,
            confidence=confidence if confidence is not None else self.confidence,
            status=status if status is not None else self.status,
            relationships=relationships if relationships is not None else dict(self.relationships),
            metadata=metadata if metadata is not None else dict(self.metadata),
            root_memory_id=root_memory_id if root_memory_id is not None else self.root_memory_id,
            parent_memory_id=parent_memory_id if parent_memory_id is not None else self.parent_memory_id,
            version=version if version is not None else self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Memory into a standard Python dictionary."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_event_ids": list(self.source_event_ids),
            "source_agent_ids": list(self.source_agent_ids),
            "importance": self.importance,
            "confidence": self.confidence,
            "status": self.status,
            "relationships": _unfreeze_container(self.relationships),
            "metadata": _unfreeze_container(self.metadata),
            "root_memory_id": self.root_memory_id,
            "parent_memory_id": self.parent_memory_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """
        Reconstruct a Memory instance from a dictionary.
        
        Parses ISO format timestamps and validates data structure.
        """
        if not isinstance(data, dict):
            raise MemoryValidationError(
                f"Expected dict for Memory deserialization, got {type(data).__name__}"
            )

        raw_data = dict(data)

        # Parse created_at
        created = raw_data.get("created_at")
        if isinstance(created, str):
            try:
                normalized_c = created.replace("Z", "+00:00")
                raw_data["created_at"] = datetime.fromisoformat(normalized_c)
            except Exception as e:
                raise MemoryValidationError(f"Invalid created_at format '{created}': {e}") from e

        # Parse updated_at
        updated = raw_data.get("updated_at")
        if isinstance(updated, str):
            try:
                normalized_u = updated.replace("Z", "+00:00")
                raw_data["updated_at"] = datetime.fromisoformat(normalized_u)
            except Exception as e:
                raise MemoryValidationError(f"Invalid updated_at format '{updated}': {e}") from e

        # Normalize sequences
        if "source_event_ids" in raw_data and raw_data["source_event_ids"] is not None:
            raw_data["source_event_ids"] = tuple(raw_data["source_event_ids"])
        if "source_agent_ids" in raw_data and raw_data["source_agent_ids"] is not None:
            raw_data["source_agent_ids"] = tuple(raw_data["source_agent_ids"])

        # Default versioning fields if not present (backward compatibility)
        if "root_memory_id" not in raw_data or raw_data["root_memory_id"] is None:
            raw_data["root_memory_id"] = raw_data.get("memory_id")
        if "version" not in raw_data or raw_data["version"] is None:
            raw_data["version"] = 1
        if "parent_memory_id" not in raw_data:
            raw_data["parent_memory_id"] = None

        try:
            return cls(**raw_data)
        except TypeError as e:
            raise MemoryValidationError(f"Invalid fields in memory dictionary: {e}") from e

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize memory to a JSON string."""
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Memory":
        """Deserialize a Memory from a JSON string."""
        if not isinstance(json_str, (str, bytes)):
            raise MemoryValidationError(f"Expected JSON string, got {type(json_str).__name__}")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise MemoryValidationError(f"Malformed JSON: {e}") from e
        return cls.from_dict(data)

