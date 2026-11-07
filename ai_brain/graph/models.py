"""
Immutable GraphNode and GraphRelationship models for AI Brain Knowledge Graph.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union
import uuid

from ai_brain.graph.exceptions import GraphValidationError
from ai_brain.graph.types import (
    NodeType,
    RelationshipType,
    normalize_node_type,
    normalize_relationship_type,
)


def _json_serializer(obj: Any) -> Any:
    """Helper serializer for JSON conversion."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (NodeType, RelationshipType)):
        return obj.value
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (MappingProxyType, Mapping)):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


from ai_brain.common.freeze import _freeze_container, _unfreeze_container


def _normalize_utc_datetime(dt: Any, field_name: str) -> datetime:
    """Validate and normalize a datetime to UTC."""
    if not isinstance(dt, datetime):
        raise GraphValidationError(f"{field_name} must be a datetime object, got {type(dt).__name__}")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class GraphNode:
    """
    Immutable representation of an entity, concept, memory, event, or project in the Knowledge Graph.
    
    All attributes are read-only once created. Attempting to assign or delete
    attributes will raise a FrozenInstanceError (AttributeError).
    """
    node_id: str
    node_type: str
    project_id: str
    label: str
    source_ref: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate all node attributes and enforce immutability."""
        # 1. Validate node_id
        if not self.node_id or not isinstance(self.node_id, str) or not self.node_id.strip():
            raise GraphValidationError("node_id must be a non-empty string")
        object.__setattr__(self, "node_id", self.node_id.strip())

        # 2. Validate node_type
        try:
            norm_type = normalize_node_type(self.node_type)
            object.__setattr__(self, "node_type", norm_type)
        except ValueError as e:
            raise GraphValidationError(str(e)) from e

        # 3. Validate project_id
        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise GraphValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        # 4. Validate label
        if not self.label or not isinstance(self.label, str) or not self.label.strip():
            raise GraphValidationError("label must be a non-empty string")
        object.__setattr__(self, "label", self.label.strip())

        # 5. Validate source_ref
        if self.source_ref is not None:
            if not isinstance(self.source_ref, str) or not self.source_ref.strip():
                raise GraphValidationError("source_ref must be a non-empty string if provided")
            object.__setattr__(self, "source_ref", self.source_ref.strip())

        # 6. Validate timestamps
        c_utc = _normalize_utc_datetime(self.created_at, "created_at")
        u_utc = _normalize_utc_datetime(self.updated_at, "updated_at")
        if u_utc < c_utc:
            raise GraphValidationError(
                f"updated_at ({u_utc.isoformat()}) cannot be earlier than created_at ({c_utc.isoformat()})"
            )
        object.__setattr__(self, "created_at", c_utc)
        object.__setattr__(self, "updated_at", u_utc)

        # 7. Freeze metadata
        if self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        elif isinstance(self.metadata, (dict, MappingProxyType)):
            object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))
        else:
            raise GraphValidationError("metadata must be a dictionary or mapping")

    @classmethod
    def create(
        cls,
        *,
        node_id: Optional[str] = None,
        node_type: str | NodeType,
        project_id: str,
        label: str,
        source_ref: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> "GraphNode":
        """Factory method to construct a validated, immutable GraphNode."""
        nid = node_id.strip() if node_id is not None else str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        c_at = created_at or now
        u_at = updated_at or c_at

        return cls(
            node_id=nid,
            node_type=node_type if isinstance(node_type, str) else node_type.value,
            project_id=project_id,
            label=label,
            source_ref=source_ref,
            metadata=metadata or {},
            created_at=c_at,
            updated_at=u_at,
        )

    def with_updates(
        self,
        *,
        label: Optional[str] = None,
        source_ref: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        updated_at: Optional[datetime] = None,
    ) -> "GraphNode":
        """Produce an updated immutable copy of this node, updating updated_at."""
        merged_meta = dict(_unfreeze_container(self.metadata))
        if metadata:
            merged_meta.update(_unfreeze_container(metadata))

        now = datetime.now(timezone.utc)
        effective_updated = updated_at or max(self.updated_at, now)
        if effective_updated < self.created_at:
            effective_updated = self.created_at

        return GraphNode(
            node_id=self.node_id,
            node_type=self.node_type,
            project_id=self.project_id,
            label=label if label is not None else self.label,
            source_ref=source_ref if source_ref is not None else self.source_ref,
            metadata=merged_meta,
            created_at=self.created_at,
            updated_at=effective_updated,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the GraphNode to a JSON-compatible dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "project_id": self.project_id,
            "label": self.label,
            "source_ref": self.source_ref,
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphNode":
        """Reconstruct a GraphNode from a dictionary."""
        if not isinstance(data, (dict, Mapping)):
            raise GraphValidationError("Node data must be a dictionary")
        c_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc)
        u_at = datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else c_at

        return cls.create(
            node_id=data.get("node_id"),
            node_type=data.get("node_type", ""),
            project_id=data.get("project_id", ""),
            label=data.get("label", ""),
            source_ref=data.get("source_ref"),
            metadata=data.get("metadata", {}),
            created_at=c_at,
            updated_at=u_at,
        )

    def to_json(self) -> str:
        """Serialize the GraphNode to a JSON string."""
        return json.dumps(self.to_dict(), default=_json_serializer)

    @classmethod
    def from_json(cls, json_str: str) -> "GraphNode":
        """Deserialize a GraphNode from a JSON string."""
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise GraphValidationError(f"Invalid JSON format for GraphNode: {e}") from e
        return cls.from_dict(data)


@dataclass(frozen=True, kw_only=True)
class GraphRelationship:
    """
    Immutable representation of a directed relationship (edge) between two nodes.
    
    All attributes are read-only once instantiated. Attempting to assign or delete
    attributes will raise a FrozenInstanceError (AttributeError).
    """
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: str
    project_id: str
    weight: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate relationship attributes and enforce immutability."""
        # 1. Validate relationship_id
        if not self.relationship_id or not isinstance(self.relationship_id, str) or not self.relationship_id.strip():
            raise GraphValidationError("relationship_id must be a non-empty string")
        object.__setattr__(self, "relationship_id", self.relationship_id.strip())

        # 2. Validate source_id & target_id
        if not self.source_id or not isinstance(self.source_id, str) or not self.source_id.strip():
            raise GraphValidationError("source_id must be a non-empty string")
        if not self.target_id or not isinstance(self.target_id, str) or not self.target_id.strip():
            raise GraphValidationError("target_id must be a non-empty string")
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "target_id", self.target_id.strip())

        # 3. Prevent self-references
        if self.source_id.strip() == self.target_id.strip():
            raise GraphValidationError(
                f"Self-referencing relationships are not permitted: source '{self.source_id}' == target '{self.target_id}'"
            )

        # 4. Validate relationship_type
        try:
            norm_type = normalize_relationship_type(self.relationship_type)
            object.__setattr__(self, "relationship_type", norm_type)
        except ValueError as e:
            raise GraphValidationError(str(e)) from e

        # 5. Validate project_id
        if not self.project_id or not isinstance(self.project_id, str) or not self.project_id.strip():
            raise GraphValidationError("project_id must be a non-empty string")
        object.__setattr__(self, "project_id", self.project_id.strip())

        # 6. Validate weight
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise GraphValidationError(f"weight must be a non-negative number, got {self.weight!r}")
        w_float = float(self.weight)
        if w_float < 0.0:
            raise GraphValidationError(f"weight cannot be negative, got {self.weight}")
        object.__setattr__(self, "weight", round(w_float, 4))

        # 7. Validate created_at
        c_utc = _normalize_utc_datetime(self.created_at, "created_at")
        object.__setattr__(self, "created_at", c_utc)

        # 8. Freeze provenance & metadata
        if self.provenance is None:
            object.__setattr__(self, "provenance", MappingProxyType({}))
        elif isinstance(self.provenance, (dict, MappingProxyType)):
            object.__setattr__(self, "provenance", _freeze_container(_unfreeze_container(self.provenance)))
        else:
            raise GraphValidationError("provenance must be a dictionary or mapping")

        if self.metadata is None:
            object.__setattr__(self, "metadata", MappingProxyType({}))
        elif isinstance(self.metadata, (dict, MappingProxyType)):
            object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))
        else:
            raise GraphValidationError("metadata must be a dictionary or mapping")

    @property
    def source_node_id(self) -> str:
        """Alias for source_id."""
        return self.source_id

    @property
    def target_node_id(self) -> str:
        """Alias for target_id."""
        return self.target_id

    @classmethod
    def create(
        cls,
        *,
        relationship_id: Optional[str] = None,
        source_id: Optional[str] = None,
        source_node_id: Optional[str] = None,
        target_id: Optional[str] = None,
        target_node_id: Optional[str] = None,
        relationship_type: str | RelationshipType,
        project_id: str,
        weight: float = 1.0,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> "GraphRelationship":
        """Factory method to construct a validated, immutable GraphRelationship."""
        rid = relationship_id.strip() if relationship_id is not None else str(uuid.uuid4())
        src = source_id or source_node_id or ""
        tgt = target_id or target_node_id or ""
        now = datetime.now(timezone.utc)
        c_at = created_at or now

        return cls(
            relationship_id=rid,
            source_id=src,
            target_id=tgt,
            relationship_type=relationship_type if isinstance(relationship_type, str) else relationship_type.value,
            project_id=project_id,
            weight=weight,
            provenance=provenance or {},
            metadata=metadata or {},
            created_at=c_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the GraphRelationship to a JSON-compatible dictionary."""
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "project_id": self.project_id,
            "weight": self.weight,
            "provenance": _unfreeze_container(self.provenance),
            "metadata": _unfreeze_container(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphRelationship":
        """Reconstruct a GraphRelationship from a dictionary."""
        if not isinstance(data, (dict, Mapping)):
            raise GraphValidationError("Relationship data must be a dictionary")
        c_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc)

        return cls.create(
            relationship_id=data.get("relationship_id"),
            source_id=data.get("source_id") or data.get("source_node_id"),
            target_id=data.get("target_id") or data.get("target_node_id"),
            relationship_type=data.get("relationship_type", ""),
            project_id=data.get("project_id", ""),
            weight=float(data.get("weight", 1.0)),
            provenance=data.get("provenance", {}),
            metadata=data.get("metadata", {}),
            created_at=c_at,
        )

    def to_json(self) -> str:
        """Serialize the GraphRelationship to a JSON string."""
        return json.dumps(self.to_dict(), default=_json_serializer)

    @classmethod
    def from_json(cls, json_str: str) -> "GraphRelationship":
        """Deserialize a GraphRelationship from a JSON string."""
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise GraphValidationError(f"Invalid JSON format for GraphRelationship: {e}") from e
        return cls.from_dict(data)
