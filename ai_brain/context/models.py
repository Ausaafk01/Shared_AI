"""
Immutable data models for the Context Builder subsystem.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional
import uuid

from ai_brain.context.exceptions import ContextBuilderError


from ai_brain.common.freeze import _freeze_container, _unfreeze_container


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for datetimes, UUIDs, tuples, and proxies."""
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
class PriorityBreakdown:
    """
    Transparent breakdown of priority score components for a context item.
    """
    relevance_score: float
    importance_score: float
    confidence_score: float
    type_bonus: float
    recency_factor: float
    status_multiplier: float
    final_priority: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance_score": round(self.relevance_score, 4),
            "importance_score": round(self.importance_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "type_bonus": round(self.type_bonus, 4),
            "recency_factor": round(self.recency_factor, 4),
            "status_multiplier": round(self.status_multiplier, 4),
            "final_priority": round(self.final_priority, 4),
        }


@dataclass(frozen=True, kw_only=True)
class ContextItem:
    """
    Immutable representation of an item selected for inclusion in the agent context.
    """
    memory_id: str
    root_memory_id: Optional[str] = None
    version: int = 1
    project_id: str
    content: str
    memory_type: str
    status: str
    priority_score: float
    priority_breakdown: PriorityBreakdown
    relevance_score: float
    token_count: int
    is_disputed: bool = False
    source_agent_ids: tuple[str, ...] = field(default_factory=tuple)
    source_event_ids: tuple[str, ...] = field(default_factory=tuple)
    graph_relationships: tuple[str, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_agent_ids", tuple(self.source_agent_ids))
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        object.__setattr__(self, "graph_relationships", tuple(self.graph_relationships))
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

        if not isinstance(self.created_at, datetime):
            raise ContextBuilderError("created_at must be a datetime object")
        c_utc = (
            self.created_at.replace(tzinfo=timezone.utc)
            if self.created_at.tzinfo is None
            else self.created_at.astimezone(timezone.utc)
        )
        object.__setattr__(self, "created_at", c_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "root_memory_id": self.root_memory_id,
            "version": self.version,
            "project_id": self.project_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "status": self.status,
            "priority_score": round(self.priority_score, 4),
            "priority_breakdown": self.priority_breakdown.to_dict(),
            "relevance_score": round(self.relevance_score, 4),
            "token_count": self.token_count,
            "is_disputed": self.is_disputed,
            "source_agent_ids": list(self.source_agent_ids),
            "source_event_ids": list(self.source_event_ids),
            "graph_relationships": list(self.graph_relationships),
            "created_at": self.created_at.isoformat(),
            "metadata": _unfreeze_container(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class ExcludedItem:
    """
    Immutable representation of an item excluded from context with reasoning.
    """
    memory_id: str
    reason: str
    priority_score: float = 0.0
    token_count: int = 0
    is_disputed: bool = False
    status: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze_container(_unfreeze_container(self.details)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "reason": self.reason,
            "priority_score": round(self.priority_score, 4),
            "token_count": self.token_count,
            "is_disputed": self.is_disputed,
            "status": self.status,
            "details": _unfreeze_container(self.details),
        }


@dataclass(frozen=True, kw_only=True)
class ContextSection:
    """
    Immutable named group of context items (e.g., Decisions, Facts, Tasks).
    """
    name: str
    items: tuple[ContextItem, ...] = field(default_factory=tuple)
    section_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "items": [it.to_dict() for it in self.items],
            "section_tokens": self.section_tokens,
        }


@dataclass(frozen=True, kw_only=True)
class ContextPackage:
    """
    Immutable complete context package ready for agent ingestion.
    """
    project_id: str
    query: str
    sections: tuple[ContextSection, ...] = field(default_factory=tuple)
    items: tuple[ContextItem, ...] = field(default_factory=tuple)
    token_budget: int
    tokens_used: int
    remaining_tokens: int
    excluded_items: tuple[ExcludedItem, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    rendered_markdown: str = ""
    rendered_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "excluded_items", tuple(self.excluded_items))
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

    @property
    def has_disputes(self) -> bool:
        """True if any included context item is disputed."""
        return any(it.is_disputed or it.status == "DISPUTED" for it in self.items)

    @property
    def has_omitted_disputes(self) -> bool:
        """True if any excluded item is disputed."""
        return any(ex.is_disputed or ex.status == "DISPUTED" for ex in self.excluded_items)

    def get_excluded_disputes(self) -> tuple[ExcludedItem, ...]:
        """Return all excluded disputed items."""
        return tuple(ex for ex in self.excluded_items if ex.is_disputed or ex.status == "DISPUTED")

    def get_section(self, name: str) -> Optional[ContextSection]:
        """Look up section by name (case-insensitive)."""
        name_lower = name.strip().lower()
        for sec in self.sections:
            if sec.name.strip().lower() == name_lower:
                return sec
        return None

    def get_active_tasks(self) -> tuple[ContextItem, ...]:
        """Return all active task items in the package."""
        return tuple(it for it in self.items if it.memory_type == "TASK" and it.status == "ACTIVE")

    def get_active_decisions(self) -> tuple[ContextItem, ...]:
        """Return all active decision items in the package."""
        return tuple(it for it in self.items if it.memory_type == "DECISION" and it.status == "ACTIVE")

    def to_markdown(self) -> str:
        """Return prompt-ready markdown string."""
        return self.rendered_markdown

    def to_text(self) -> str:
        """Return plain text representation."""
        return self.rendered_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "query": self.query,
            "sections": [sec.to_dict() for sec in self.sections],
            "items": [it.to_dict() for it in self.items],
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "remaining_tokens": self.remaining_tokens,
            "excluded_items": [ex.to_dict() for ex in self.excluded_items],
            "metadata": _unfreeze_container(self.metadata),
            "rendered_markdown": self.rendered_markdown,
            "rendered_text": self.rendered_text,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextPackage":
        items = tuple(
            ContextItem(
                memory_id=d["memory_id"],
                root_memory_id=d.get("root_memory_id"),
                version=int(d.get("version", 1)),
                project_id=d["project_id"],
                content=d["content"],
                memory_type=d["memory_type"],
                status=d["status"],
                priority_score=float(d["priority_score"]),
                priority_breakdown=PriorityBreakdown(**d["priority_breakdown"]),
                relevance_score=float(d["relevance_score"]),
                token_count=int(d["token_count"]),
                is_disputed=bool(d.get("is_disputed", False)),
                source_agent_ids=tuple(d.get("source_agent_ids", ())),
                source_event_ids=tuple(d.get("source_event_ids", ())),
                graph_relationships=tuple(d.get("graph_relationships", ())),
                created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(timezone.utc),
                metadata=d.get("metadata", {}),
            )
            for d in data.get("items", [])
        )

        item_map = {it.memory_id: it for it in items}

        sections = tuple(
            ContextSection(
                name=sec["name"],
                items=tuple(item_map[it_d["memory_id"]] for it_d in sec.get("items", []) if it_d["memory_id"] in item_map),
                section_tokens=int(sec.get("section_tokens", 0)),
            )
            for sec in data.get("sections", [])
        )

        excluded = tuple(
            ExcludedItem(
                memory_id=ex["memory_id"],
                reason=ex["reason"],
                priority_score=float(ex.get("priority_score", 0.0)),
                token_count=int(ex.get("token_count", 0)),
                is_disputed=bool(ex.get("is_disputed", False)),
                status=ex.get("status"),
                details=ex.get("details", {}),
            )
            for ex in data.get("excluded_items", [])
        )

        return cls(
            project_id=data["project_id"],
            query=data["query"],
            sections=sections,
            items=items,
            token_budget=int(data["token_budget"]),
            tokens_used=int(data["tokens_used"]),
            remaining_tokens=int(data["remaining_tokens"]),
            excluded_items=excluded,
            metadata=data.get("metadata", {}),
            rendered_markdown=data.get("rendered_markdown", ""),
            rendered_text=data.get("rendered_text", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ContextPackage":
        return cls.from_dict(json.loads(json_str))

