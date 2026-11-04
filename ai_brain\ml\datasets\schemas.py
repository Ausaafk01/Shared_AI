"""
Schemas for AI Brain machine learning datasets with explicit provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


class ExampleSource(StrEnum):
    REAL = "REAL"                    # Directly harvested from authentic interactions/events
    MANUALLY_LABELLED = "MANUALLY_LABELLED"  # Human or domain-expert annotated
    FEEDBACK_DERIVED = "FEEDBACK_DERIVED"    # Generated from recorded agent/user feedback
    SYNTHETIC = "SYNTHETIC"          # Synthetically generated for cold-start / bootstrapping


@dataclass(frozen=True)
class ExampleProvenance:
    """Provenance tracking origin, creation time, source entity IDs, and collection method."""
    source_type: ExampleSource
    created_at: datetime
    project_id: Optional[str] = None
    source_memory_ids: Tuple[str, ...] = field(default_factory=tuple)
    source_event_ids: Tuple[str, ...] = field(default_factory=tuple)
    collector: str = "system"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "created_at": self.created_at.isoformat(),
            "project_id": self.project_id,
            "source_memory_ids": list(self.source_memory_ids),
            "source_event_ids": list(self.source_event_ids),
            "collector": self.collector,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExampleProvenance":
        return cls(
            source_type=ExampleSource(data["source_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            project_id=data.get("project_id"),
            source_memory_ids=tuple(data.get("source_memory_ids", ())),
            source_event_ids=tuple(data.get("source_event_ids", ())),
            collector=str(data.get("collector", "system")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class DatasetExample:
    """Single labeled training/evaluation example."""
    example_id: str
    features: Mapping[str, float]
    label: Union[float, str, int, bool]
    provenance: ExampleProvenance
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "features": dict(self.features),
            "label": self.label,
            "provenance": self.provenance.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetExample":
        return cls(
            example_id=str(data["example_id"]),
            features=MappingProxyType(dict(data["features"])),
            label=data["label"],
            provenance=ExampleProvenance.from_dict(data["provenance"]),
            metadata=MappingProxyType(dict(data.get("metadata", {}))),
        )


@dataclass(frozen=True)
class DatasetSplit:
    """Immutable split containing train, validation, and optional test examples."""
    name: str
    train_examples: Tuple[DatasetExample, ...]
    validation_examples: Tuple[DatasetExample, ...]
    test_examples: Tuple[DatasetExample, ...] = field(default_factory=tuple)
    feature_names: Tuple[str, ...] = field(default_factory=tuple)
    target_name: str = "label"

    @property
    def total_size(self) -> int:
        return len(self.train_examples) + len(self.validation_examples) + len(self.test_examples)

    @property
    def is_synthetic_only(self) -> bool:
        """Indicates if the dataset consists solely of synthetic examples."""
        all_ex = self.train_examples + self.validation_examples + self.test_examples
        return bool(all_ex) and all(e.provenance.source_type == ExampleSource.SYNTHETIC for e in all_ex)
