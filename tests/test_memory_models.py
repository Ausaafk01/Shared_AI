"""
Automated tests for the Memory model, validation, immutability, and serialization.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone, timedelta
import unittest

from ai_brain.memories.exceptions import MemoryValidationError
from ai_brain.memories.models import Memory
from ai_brain.memories.types import (
    MemoryStatus,
    MemoryType,
    is_valid_memory_status,
    is_valid_memory_type,
    register_memory_type,
    unregister_memory_type,
)


class TestMemoryModel(unittest.TestCase):
    """Tests covering Memory creation, validation, immutability, and serialization."""

    def test_memory_creation_minimal(self) -> None:
        """Test creating a memory with only minimal required fields."""
        mem = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its primary database.",
            project_id="proj-db",
        )
        self.assertIsNotNone(mem.memory_id)
        self.assertEqual(len(mem.memory_id), 36)  # UUID4
        self.assertEqual(mem.memory_type, "FACT")
        self.assertEqual(mem.content, "The project uses PostgreSQL as its primary database.")
        self.assertEqual(mem.project_id, "proj-db")
        self.assertEqual(mem.importance, 0.5)
        self.assertEqual(mem.confidence, 1.0)
        self.assertEqual(mem.status, "ACTIVE")
        self.assertEqual(mem.source_event_ids, ())
        self.assertEqual(mem.source_agent_ids, ())
        self.assertEqual(dict(mem.relationships), {})
        self.assertEqual(dict(mem.metadata), {})
        self.assertIsNotNone(mem.created_at)
        self.assertIsNotNone(mem.updated_at)
        self.assertEqual(mem.created_at.tzinfo, timezone.utc)
        self.assertEqual(mem.updated_at.tzinfo, timezone.utc)
        self.assertGreaterEqual(mem.updated_at, mem.created_at)

    def test_memory_creation_all_fields(self) -> None:
        """Test creating a memory with all optional fields provided."""
        t0 = datetime(2026, 11, 2, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 2, 10, 30, 0, tzinfo=timezone.utc)
        mem = Memory.create(
            memory_id="mem-custom-123",
            memory_type=MemoryType.DECISION,
            content="Redis was removed because it caused stale cache problems.",
            project_id="proj-arch",
            created_at=t0,
            updated_at=t1,
            source_event_ids=["evt-1", "evt-2"],
            source_agent_ids=["claude", "gemini"],
            importance=0.9,
            confidence=0.95,
            status=MemoryStatus.ACTIVE,
            relationships={"supersedes": ["mem-old-1"]},
            metadata={"domain": "caching", "verified": True},
        )
        self.assertEqual(mem.memory_id, "mem-custom-123")
        self.assertEqual(mem.memory_type, "DECISION")
        self.assertEqual(mem.content, "Redis was removed because it caused stale cache problems.")
        self.assertEqual(mem.project_id, "proj-arch")
        self.assertEqual(mem.created_at, t0)
        self.assertEqual(mem.updated_at, t1)
        self.assertEqual(mem.source_event_ids, ("evt-1", "evt-2"))
        self.assertEqual(mem.source_agent_ids, ("claude", "gemini"))
        self.assertEqual(mem.importance, 0.9)
        self.assertEqual(mem.confidence, 0.95)
        self.assertEqual(mem.status, "ACTIVE")
        self.assertEqual(dict(mem.relationships), {"supersedes": ("mem-old-1",)})
        self.assertEqual(mem.metadata["domain"], "caching")

    def test_required_fields_validation(self) -> None:
        """Test that missing or empty required fields raise MemoryValidationError."""
        # Empty memory_id
        with self.assertRaises(MemoryValidationError):
            Memory(
                memory_id="   ",
                memory_type="FACT",
                content="Valid content",
                project_id="proj-1",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

        # Empty content
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.FACT,
                content="",
                project_id="proj-1",
            )

        # Whitespace content
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.FACT,
                content="    ",
                project_id="proj-1",
            )

        # Empty project_id
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.FACT,
                content="Postgres in use",
                project_id="",
            )

    def test_memory_types_support_and_validation(self) -> None:
        """Test all standard memory types and validation of invalid types."""
        standard_types = [
            MemoryType.FACT,
            MemoryType.DECISION,
            MemoryType.TASK,
            MemoryType.ERROR,
            MemoryType.ATTEMPT,
            MemoryType.CODE_KNOWLEDGE,
            MemoryType.PREFERENCE,
            MemoryType.CONTEXT,
        ]

        for m_type in standard_types:
            mem = Memory.create(
                memory_type=m_type,
                content=f"Testing type {m_type}",
                project_id="proj-types",
            )
            self.assertEqual(mem.memory_type, m_type.value)

        # Lowercase string normalization
        mem_lower = Memory.create(
            memory_type="preference",
            content="User prefers concise code.",
            project_id="proj-types",
        )
        self.assertEqual(mem_lower.memory_type, "PREFERENCE")

        # Invalid memory type
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type="NON_EXISTENT_TYPE",
                content="Some knowledge",
                project_id="proj-types",
            )

    def test_memory_type_extensibility(self) -> None:
        """Test registering and using custom extensible memory types."""
        self.assertFalse(is_valid_memory_type("DOMAIN_HEURISTIC"))

        register_memory_type("DOMAIN_HEURISTIC")
        try:
            self.assertTrue(is_valid_memory_type("DOMAIN_HEURISTIC"))

            mem = Memory.create(
                memory_type="DOMAIN_HEURISTIC",
                content="Cache misses are handled by falling back to cold storage.",
                project_id="proj-custom",
            )
            self.assertEqual(mem.memory_type, "DOMAIN_HEURISTIC")
        finally:
            unregister_memory_type("DOMAIN_HEURISTIC")

        # Once unregistered, new creations of that type fail
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type="DOMAIN_HEURISTIC",
                content="Another heuristic",
                project_id="proj-custom",
            )

    def test_memory_status_validation(self) -> None:
        """Test memory lifecycle status values and validation."""
        statuses = [
            MemoryStatus.ACTIVE,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.DISPUTED,
            MemoryStatus.ARCHIVED,
        ]

        for status in statuses:
            mem = Memory.create(
                memory_type=MemoryType.FACT,
                content="Testing status",
                project_id="proj-status",
                status=status,
            )
            self.assertEqual(mem.status, status.value)

        # Lowercase normalization
        mem_norm = Memory.create(
            memory_type=MemoryType.FACT,
            content="Testing normalization",
            project_id="proj-status",
            status="superseded",
        )
        self.assertEqual(mem_norm.status, "SUPERSEDED")

        # Invalid status
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.FACT,
                content="Invalid status test",
                project_id="proj-status",
                status="DELETED_OR_INVALID",
            )

    def test_importance_and_confidence_boundaries(self) -> None:
        """Test boundary conditions and range enforcement for importance and confidence [0.0, 1.0]."""
        # Exact boundaries: 0.0 and 1.0
        mem_min = Memory.create(
            memory_type=MemoryType.FACT,
            content="Min bounds",
            project_id="proj-bounds",
            importance=0.0,
            confidence=0.0,
        )
        self.assertEqual(mem_min.importance, 0.0)
        self.assertEqual(mem_min.confidence, 0.0)

        mem_max = Memory.create(
            memory_type=MemoryType.FACT,
            content="Max bounds",
            project_id="proj-bounds",
            importance=1.0,
            confidence=1.0,
        )
        self.assertEqual(mem_max.importance, 1.0)
        self.assertEqual(mem_max.confidence, 1.0)

        # Integer 0 and 1 should be converted to float
        mem_int = Memory.create(
            memory_type=MemoryType.FACT,
            content="Integer bounds",
            project_id="proj-bounds",
            importance=1,
            confidence=0,
        )
        self.assertEqual(mem_int.importance, 1.0)
        self.assertEqual(mem_int.confidence, 0.0)

        # Importance out of range
        with self.assertRaises(MemoryValidationError):
            Memory.create(memory_type=MemoryType.FACT, content="c", project_id="p", importance=-0.1)

        with self.assertRaises(MemoryValidationError):
            Memory.create(memory_type=MemoryType.FACT, content="c", project_id="p", importance=1.05)

        # Confidence out of range
        with self.assertRaises(MemoryValidationError):
            Memory.create(memory_type=MemoryType.FACT, content="c", project_id="p", confidence=-0.01)

        with self.assertRaises(MemoryValidationError):
            Memory.create(memory_type=MemoryType.FACT, content="c", project_id="p", confidence=2.0)

        # Boolean inputs rejected
        with self.assertRaises(MemoryValidationError):
            Memory.create(memory_type=MemoryType.FACT, content="c", project_id="p", importance=True)  # type: ignore

    def test_timestamp_handling_and_consistency(self) -> None:
        """Test timestamp timezone handling and consistency between created_at and updated_at."""
        # Naive datetime normalized to UTC
        naive_dt = datetime(2026, 11, 2, 14, 0, 0)
        mem_naive = Memory.create(
            memory_type=MemoryType.FACT,
            content="Naive dt",
            project_id="proj-ts",
            created_at=naive_dt,
            updated_at=naive_dt,
        )
        self.assertEqual(mem_naive.created_at.tzinfo, timezone.utc)
        self.assertEqual(mem_naive.updated_at.tzinfo, timezone.utc)

        # Non-UTC timezone converted to UTC
        tz_plus5 = timezone(timedelta(hours=5))
        aware_dt = datetime(2026, 11, 2, 19, 0, 0, tzinfo=tz_plus5)
        mem_aware = Memory.create(
            memory_type=MemoryType.FACT,
            content="Aware dt",
            project_id="proj-ts",
            created_at=aware_dt,
            updated_at=aware_dt,
        )
        self.assertEqual(mem_aware.created_at.hour, 14)
        self.assertEqual(mem_aware.created_at.tzinfo, timezone.utc)

        # Consistency: updated_at earlier than created_at must fail
        t_early = datetime(2026, 11, 2, 10, 0, 0, tzinfo=timezone.utc)
        t_late = datetime(2026, 11, 2, 11, 0, 0, tzinfo=timezone.utc)
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.FACT,
                content="Inconsistent timestamps",
                project_id="proj-ts",
                created_at=t_late,
                updated_at=t_early,
            )

    def test_source_event_references_validation(self) -> None:
        """Test source_event_ids and source_agent_ids validation."""
        mem = Memory.create(
            memory_type=MemoryType.TASK,
            content="The authentication bug remains unresolved.",
            project_id="proj-bug",
            source_event_ids=["evt-auth-1", "evt-auth-2"],
            source_agent_ids=["codex"],
        )
        self.assertEqual(mem.source_event_ids, ("evt-auth-1", "evt-auth-2"))
        self.assertEqual(mem.source_agent_ids, ("codex",))

        # Empty string inside source_event_ids rejected
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.TASK,
                content="Bad event id",
                project_id="p",
                source_event_ids=["evt-1", ""],
            )

        # Non-string inside source_agent_ids rejected
        with self.assertRaises(MemoryValidationError):
            Memory.create(
                memory_type=MemoryType.TASK,
                content="Bad agent id",
                project_id="p",
                source_agent_ids=[123],  # type: ignore
            )

    def test_immutability(self) -> None:
        """Test that memories cannot be directly mutated after creation."""
        mem = Memory.create(
            memory_type=MemoryType.FACT,
            content="Initial knowledge",
            project_id="proj-freeze",
            metadata={"source": "scan", "nested": {"k": "v"}},
            relationships={"related": ["mem-2"]},
            source_event_ids=["evt-1"],
        )

        # Direct attribute assignment fails
        with self.assertRaises(FrozenInstanceError):
            mem.content = "Mutated knowledge"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            mem.status = "SUPERSEDED"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            del mem.project_id  # type: ignore

        # Container modification fails
        with self.assertRaises(TypeError):
            mem.metadata["new_key"] = "val"  # type: ignore

        with self.assertRaises(TypeError):
            mem.relationships["related"] = "something"  # type: ignore

        with self.assertRaises(AttributeError):
            mem.source_event_ids.append("evt-2")  # type: ignore

        # Modifying outside references doesn't affect internal state
        external_meta = {"key": "original"}
        external_events = ["e1", "e2"]
        mem_ext = Memory.create(
            memory_type=MemoryType.FACT,
            content="Defensive copy test",
            project_id="proj-freeze",
            metadata=external_meta,
            source_event_ids=external_events,
        )
        external_meta["key"] = "tampered"
        external_events.append("e3")
        self.assertEqual(mem_ext.metadata["key"], "original")
        self.assertEqual(mem_ext.source_event_ids, ("e1", "e2"))

    def test_serialization_to_dict_and_from_dict(self) -> None:
        """Test dictionary serialization and roundtrip deserialization."""
        mem = Memory.create(
            memory_id="mem-roundtrip",
            memory_type=MemoryType.DECISION,
            content="Authentication is handled through JWT.",
            project_id="proj-auth",
            source_event_ids=["evt-10"],
            source_agent_ids=["agent-antigravity"],
            importance=0.85,
            confidence=0.98,
            status=MemoryStatus.ACTIVE,
            relationships={"applies_to": ["module-auth"]},
            metadata={"spec": "RFC-7519"},
        )

        mem_dict = mem.to_dict()
        self.assertIsInstance(mem_dict, dict)
        self.assertEqual(mem_dict["memory_id"], "mem-roundtrip")
        self.assertEqual(mem_dict["memory_type"], "DECISION")
        self.assertEqual(mem_dict["source_event_ids"], ["evt-10"])
        self.assertEqual(mem_dict["relationships"], {"applies_to": ["module-auth"]})

        # Reconstruct
        reconstructed = Memory.from_dict(mem_dict)
        self.assertEqual(reconstructed.memory_id, mem.memory_id)
        self.assertEqual(reconstructed.memory_type, mem.memory_type)
        self.assertEqual(reconstructed.content, mem.content)
        self.assertEqual(reconstructed.project_id, mem.project_id)
        self.assertEqual(reconstructed.created_at, mem.created_at)
        self.assertEqual(reconstructed.updated_at, mem.updated_at)
        self.assertEqual(reconstructed.source_event_ids, mem.source_event_ids)
        self.assertEqual(reconstructed.source_agent_ids, mem.source_agent_ids)
        self.assertEqual(reconstructed.importance, mem.importance)
        self.assertEqual(reconstructed.confidence, mem.confidence)
        self.assertEqual(reconstructed.status, mem.status)
        self.assertEqual(dict(reconstructed.relationships), dict(mem.relationships))
        self.assertEqual(dict(reconstructed.metadata), dict(mem.metadata))

    def test_serialization_to_json_and_from_json(self) -> None:
        """Test JSON string serialization and deserialization."""
        mem = Memory.create(
            memory_type=MemoryType.PREFERENCE,
            content="The user prefers concise code.",
            project_id="proj-pref",
            metadata={"verbosity": "low"},
        )

        json_str = mem.to_json()
        self.assertIsInstance(json_str, str)

        parsed = Memory.from_json(json_str)
        self.assertEqual(parsed.memory_id, mem.memory_id)
        self.assertEqual(parsed.content, mem.content)
        self.assertEqual(parsed.memory_type, "PREFERENCE")
        self.assertEqual(parsed.metadata["verbosity"], "low")

    def test_invalid_deserialization_data(self) -> None:
        """Test deserialization with invalid data raises MemoryValidationError."""
        with self.assertRaises(MemoryValidationError):
            Memory.from_dict("not-a-dict")  # type: ignore

        with self.assertRaises(MemoryValidationError):
            Memory.from_json("{invalid-json-content")

        with self.assertRaises(MemoryValidationError):
            Memory.from_dict({
                "memory_id": "m1",
                "memory_type": "FACT",
                "content": "Valid",
                "project_id": "p1",
                "created_at": "bad-timestamp",
                "updated_at": "2026-11-02T10:00:00Z",
            })


if __name__ == "__main__":
    unittest.main()

