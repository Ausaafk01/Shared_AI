"""
Immutable data models for the Retrieval Engine subsystem.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union
import uuid

from ai_brain.retrieval.exceptions import QueryValidationError, RetrievalExecutionError
from ai_brain.retrieval.types import RetrievalMethod, normalize_retrieval_method


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
    """JSON serializer helper for datetime, enums, and mapping proxies."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, RetrievalMethod):
        return obj.value
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (MappingProxyType, Mapping)):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass(frozen=True, kw_only=True)
class QueryFilters:
    """
    Optional filter criteria applied during memory retrieval.
    """
    memory_types: tuple[str, ...] = field(default_factory=tuple)
    allowed_statuses: tuple[str, ...] = field(default=("ACTIVE", "DISPUTED"))
    source_agent_ids: tuple[str, ...] = field(default_factory=tuple)
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    seed_memory_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate and normalize filter values."""
        # Normalize memory types
        types = tuple(t.strip().upper() for t in self.memory_types if t and isinstance(t, str))
        object.__setattr__(self, "memory_types", types)

        # Normalize statuses (default to ACTIVE and DISPUTED if empty)
        statuses = tuple(s.strip().upper() for s in self.allowed_statuses if s and isinstance(s, str))
        if not statuses:
            statuses = ("ACTIVE", "DISPUTED")
        object.__setattr__(self, "allowed_statuses", statuses)

        # Normalize agent IDs
        agents = tuple(a.strip() for a in self.source_agent_ids if a and isinstance(a, str))
        object.__setattr__(self, "source_agent_ids", agents)

        # Normalize seeds
        seeds = tuple(s.strip() for s in self.seed_memory_ids if s and isinstance(s, str))
        object.__setattr__(self, "seed_memory_ids", seeds)

        # Normalize datetimes to UTC
        if self.time_range_start is not None:
            if not isinstance(self.time_range_start, datetime):
                raise QueryValidationError("time_range_start must be a datetime object")
            t_start = (
                self.time_range_start.replace(tzinfo=timezone.utc)
                if self.time_range_start.tzinfo is None
                else self.time_range_start.astimezone(timezone.utc)
            )
            object.__setattr__(self, "time_range_start", t_start)

        if self.time_range_end is not None:
            if not isinstance(self.time_range_end, datetime):
                raise QueryValidationError("time_range_end must be a datetime object")
            t_end = (
                self.time_range_end.replace(tzinfo=timezone.utc)
                if self.time_range_end.tzinfo is None
                else self.time_range_end.astimezone(timezone.utc)
            )
            object.__setattr__(self, "time_range_end", t_end)

        if self.time_range_start and self.time_range_end and self.time_range_start > self.time_range_end:
            raise QueryValidationError("time_range_start cannot be later than time_range_end")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_types": list(self.memory_types),
            "allowed_statuses": list(self.allowed_statuses),
            "source_agent_ids": list(self.source_agent_ids),
            "time_range_start": self.time_range_start.isoformat() if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
            "seed_memory_ids": list(self.seed_memory_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueryFilters":
        t_start = None
        if data.get("time_range_start"):
            t_start = datetime.fromisoformat(data["time_range_start"])
        t_end = None
        if data.get("time_range_end"):
            t_end = datetime.fromisoformat(data["time_range_end"])

        return cls(
            memory_types=tuple(data.get("memory_types", ())),
            allowed_statuses=tuple(data.get("allowed_statuses", ("ACTIVE", "DISPUTED"))),
            source_agent_ids=tuple(data.get("source_agent_ids", ())),
            time_range_start=t_start,
            time_range_end=t_end,
            seed_memory_ids=tuple(data.get("seed_memory_ids", ())),
        )


@dataclass(frozen=True, kw_only=True)
class ProcessedQuery:
    """
    Immutable representation of an analyzed and normalized retrieval query.
    """
    original_query: str
    normalized_query: str
    keywords: tuple[str, ...]
    entities: tuple[str, ...]
    project_id: str
    filters: QueryFilters
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "keywords", tuple(self.keywords))
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))


@dataclass(frozen=True, kw_only=True)
class ScoreBreakdown:
    """
    Transparent breakdown of relevance score components for a retrieved memory.
    """
    base_score: float
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    importance_factor: float = 1.0
    confidence_factor: float = 1.0
    recency_factor: float = 1.0
    status_multiplier: float = 1.0
    provenance_bonus: float = 0.0
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_score": round(self.base_score, 4),
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
            "graph_score": round(self.graph_score, 4) if self.graph_score is not None else None,
            "importance_factor": round(self.importance_factor, 4),
            "confidence_factor": round(self.confidence_factor, 4),
            "recency_factor": round(self.recency_factor, 4),
            "status_multiplier": round(self.status_multiplier, 4),
            "provenance_bonus": round(self.provenance_bonus, 4),
            "final_score": round(self.final_score, 4),
        }


@dataclass(frozen=True, kw_only=True)
class RetrievalItem:
    """
    Immutable representation of a single retrieved memory with explanation metadata.
    """
    memory_id: str
    root_memory_id: Optional[str] = None
    version: int = 1
    project_id: str
    content: str
    memory_type: str
    status: str
    score: float
    score_breakdown: ScoreBreakdown
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    graph_metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    sources: tuple[str, ...] = field(default_factory=tuple)
    source_event_ids: tuple[str, ...] = field(default_factory=tuple)
    source_agent_ids: tuple[str, ...] = field(default_factory=tuple)
    is_disputed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        object.__setattr__(self, "source_agent_ids", tuple(self.source_agent_ids))
        object.__setattr__(self, "graph_metadata", _freeze_container(_unfreeze_container(self.graph_metadata)))
        object.__setattr__(self, "metadata", _freeze_container(_unfreeze_container(self.metadata)))

        if not isinstance(self.created_at, datetime):
            raise RetrievalExecutionError("created_at must be a datetime object")
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
            "score": round(self.score, 4),
            "score_breakdown": self.score_breakdown.to_dict(),
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
            "graph_score": round(self.graph_score, 4) if self.graph_score is not None else None,
            "graph_metadata": _unfreeze_container(self.graph_metadata),
            "sources": list(self.sources),
            "source_event_ids": list(self.source_event_ids),
            "source_agent_ids": list(self.source_agent_ids),
            "is_disputed": self.is_disputed,
            "created_at": self.created_at.isoformat(),
            "metadata": _unfreeze_container(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalItem":
        sb_data = data.get("score_breakdown", {})
        score_breakdown = ScoreBreakdown(
            base_score=float(sb_data.get("base_score", data.get("score", 0.0))),
            vector_score=sb_data.get("vector_score"),
            graph_score=sb_data.get("graph_score"),
            importance_factor=float(sb_data.get("importance_factor", 1.0)),
            confidence_factor=float(sb_data.get("confidence_factor", 1.0)),
            recency_factor=float(sb_data.get("recency_factor", 1.0)),
            status_multiplier=float(sb_data.get("status_multiplier", 1.0)),
            provenance_bonus=float(sb_data.get("provenance_bonus", 0.0)),
            final_score=float(sb_data.get("final_score", data.get("score", 0.0))),
        )
        c_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc)

        return cls(
            memory_id=data["memory_id"],
            root_memory_id=data.get("root_memory_id"),
            version=int(data.get("version", 1)),
            project_id=data["project_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            status=data["status"],
            score=float(data["score"]),
            score_breakdown=score_breakdown,
            vector_score=data.get("vector_score"),
            graph_score=data.get("graph_score"),
            graph_metadata=data.get("graph_metadata", {}),
            sources=tuple(data.get("sources", ())),
            source_event_ids=tuple(data.get("source_event_ids", ())),
            source_agent_ids=tuple(data.get("source_agent_ids", ())),
            is_disputed=bool(data.get("is_disputed", False)),
            created_at=c_at,
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "RetrievalItem":
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True, kw_only=True)
class RetrievalResponse:
    """
    Immutable structured response returned by the Retrieval Engine.
    """
    query: str
    normalized_query: str
    project_id: str
    method: str
    results: tuple[RetrievalItem, ...] = field(default_factory=tuple)
    total_count: int = 0
    execution_time_ms: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.total_count == 0 and len(self.results) > 0:
            object.__setattr__(self, "total_count", len(self.results))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "project_id": self.project_id,
            "method": self.method,
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalResponse":
        results = tuple(RetrievalItem.from_dict(r) for r in data.get("results", []))
        return cls(
            query=data["query"],
            normalized_query=data["normalized_query"],
            project_id=data["project_id"],
            method=data["method"],
            results=results,
            total_count=int(data.get("total_count", len(results))),
            execution_time_ms=float(data.get("execution_time_ms", 0.0)),
            warnings=tuple(data.get("warnings", ())),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "RetrievalResponse":
        return cls.from_dict(json.loads(json_str))

