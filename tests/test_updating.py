"""
Automated tests for AI Brain Memory Updating and version history component.
"""

from datetime import datetime, timezone, timedelta
import unittest

from ai_brain.memories.exceptions import (
    InvalidUpdateOperationError,
    VersionConflictError,
)
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.memories.updating import (
    MemoryUpdater,
    MemoryVersionHistory,
    UpdateAction,
)


class TestMemoryUpdating(unittest.TestCase):
    """Tests covering memory updates, version transitions, duplicate handling, and conflict preservation."""

    def setUp(self) -> None:
        self.updater = MemoryUpdater()
        self.now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)

    def test_create_new_memory_version_superseding(self) -> None:
        """
        Rule B: Test updating an existing memory with newer knowledge:
        - Creates a new ACTIVE version (version 2).
        - Supersedes the previous version (version 1).
        - Links new version to previous via parent_memory_id and relationships.
        - Preserves the original memory without in-place mutation.
        """
        old_memory = Memory.create(
            memory_id="mem-v1-id",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-db",
            source_event_ids=["evt-1"],
            source_agent_ids=["agent-a"],
            created_at=self.now,
            updated_at=self.now,
        )
        self.assertEqual(old_memory.version, 1)
        self.assertEqual(old_memory.root_memory_id, "mem-v1-id")
        self.assertIsNone(old_memory.parent_memory_id)

        new_info = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-db",
            source_event_ids=["evt-2"],
            source_agent_ids=["agent-b"],
            created_at=self.now + timedelta(minutes=5),
            updated_at=self.now + timedelta(minutes=5),
        )

        result = self.updater.update(old_memory, new_info)

        self.assertTrue(result.created)
        self.assertTrue(result.superseded)
        self.assertFalse(result.duplicate)
        self.assertFalse(result.conflict)
        self.assertEqual(result.action, UpdateAction.SUPERSEDED)

        # Check new current memory (version 2)
        new_memory = result.current_memory
        self.assertIsNotNone(new_memory)
        self.assertEqual(new_memory.version, 2)
        self.assertEqual(new_memory.root_memory_id, old_memory.root_memory_id)
        self.assertEqual(new_memory.parent_memory_id, old_memory.memory_id)
        self.assertEqual(new_memory.status, MemoryStatus.ACTIVE.value)
        self.assertEqual(new_memory.content, "The project uses PostgreSQL as its database.")
        self.assertEqual(new_memory.relationships.get("supersedes"), old_memory.memory_id)

        # Check superseded previous memory representation
        prev_memory = result.previous_memory
        self.assertIsNotNone(prev_memory)
        self.assertEqual(prev_memory.version, 1)
        self.assertEqual(prev_memory.status, MemoryStatus.SUPERSEDED.value)
        self.assertEqual(prev_memory.relationships.get("superseded_by"), new_memory.memory_id)

        # Verify original old_memory object was NOT mutated
        self.assertEqual(old_memory.status, MemoryStatus.ACTIVE.value)

    def test_version_numbering_and_chain_across_multiple_updates(self) -> None:
        """Test multiple sequential updates maintaining an ordered, immutable version chain."""
        # Version 1
        m1 = Memory.create(
            memory_id="mem-seq-1",
            memory_type=MemoryType.DECISION,
            content="Initial decision: Use SQLite for local storage.",
            project_id="proj-chain",
            created_at=self.now,
        )
        self.updater.tracker.track(m1)

        # Update to Version 2
        m2_info = Memory.create(
            memory_type=MemoryType.DECISION,
            content="Decided to use PostgreSQL for relational storage.",
            project_id="proj-chain",
            created_at=self.now + timedelta(hours=1),
        )
        res1 = self.updater.update(m1, m2_info, strategy="supersede")
        m2 = res1.current_memory
        self.assertIsNotNone(m2)
        self.assertEqual(m2.version, 2)
        self.assertEqual(m2.parent_memory_id, "mem-seq-1")
        self.assertEqual(m2.root_memory_id, m1.root_memory_id)

        # Update to Version 3
        m3_info = Memory.create(
            memory_type=MemoryType.DECISION,
            content="Decided to use PostgreSQL with connection pooling.",
            project_id="proj-chain",
            created_at=self.now + timedelta(hours=2),
        )
        res2 = self.updater.update(m2, m3_info, strategy="supersede")
        m3 = res2.current_memory
        self.assertIsNotNone(m3)
        self.assertEqual(m3.version, 3)
        self.assertEqual(m3.parent_memory_id, m2.memory_id)
        self.assertEqual(m3.root_memory_id, m1.root_memory_id)

        # Verify version history retrieval
        history = self.updater.get_history(m1.root_memory_id)
        self.assertEqual(len(history), 3)
        self.assertEqual([m.version for m in history], [1, 2, 3])
        self.assertEqual(self.updater.get_current(m1.root_memory_id).version, 3)

    def test_duplicate_information_handling(self) -> None:
        """Rule C: Duplicate incoming information creates no new memory and no version increase."""
        existing = Memory.create(
            memory_id="mem-dup-exist",
            memory_type=MemoryType.FACT,
            content="Authentication is handled through JWT.",
            project_id="proj-dup",
        )
        incoming = Memory.create(
            memory_type=MemoryType.FACT,
            content="Authentication is handled through JWT.",
            project_id="proj-dup",
        )

        result = self.updater.update(existing, incoming)
        self.assertFalse(result.created)
        self.assertFalse(result.superseded)
        self.assertTrue(result.duplicate)
        self.assertFalse(result.conflict)
        self.assertEqual(result.action, UpdateAction.DUPLICATE_NOOP)
        self.assertEqual(result.current_memory.memory_id, existing.memory_id)
        self.assertEqual(result.current_memory.version, 1)

    def test_unchanged_information_handling(self) -> None:
        """Rule D: Semantically unchanged content avoids unnecessary version creation."""
        existing = Memory.create(
            memory_id="mem-unchanged",
            memory_type=MemoryType.PREFERENCE,
            content="The user prefers concise code.",
            project_id="proj-d",
        )
        # Identical text with leading/trailing whitespace
        incoming = Memory.create(
            memory_type=MemoryType.PREFERENCE,
            content="  The user prefers concise code.  ",
            project_id="proj-d",
        )

        result = self.updater.update(existing, incoming)
        self.assertFalse(result.created)
        self.assertTrue(result.duplicate)
        self.assertEqual(result.action, UpdateAction.DUPLICATE_NOOP)

    def test_contradictory_disputed_information(self) -> None:
        """
        Rule E: Contradictory information marks both memories as DISPUTED without silently choosing one.
        """
        existing = Memory.create(
            memory_id="mem-jwt",
            memory_type=MemoryType.FACT,
            content="Authentication is handled through JWT.",
            project_id="proj-auth",
            confidence=0.9,
        )
        incoming = Memory.create(
            memory_id="mem-session",
            memory_type=MemoryType.FACT,
            content="Authentication is handled through Session Cookies.",
            project_id="proj-auth",
            confidence=0.9,
        )

        result = self.updater.update(existing, incoming, strategy="dispute")
        self.assertTrue(result.created)
        self.assertFalse(result.superseded)
        self.assertFalse(result.duplicate)
        self.assertTrue(result.conflict)
        self.assertEqual(result.action, UpdateAction.DISPUTED)

        # Both memories are marked as DISPUTED
        prev_disp = result.previous_memory
        curr_disp = result.current_memory
        self.assertIsNotNone(prev_disp)
        self.assertIsNotNone(curr_disp)
        self.assertEqual(prev_disp.status, MemoryStatus.DISPUTED.value)
        self.assertEqual(curr_disp.status, MemoryStatus.DISPUTED.value)

        # Both retain conflict references
        self.assertIn("mem-session", prev_disp.relationships.get("conflicts_with", []))
        self.assertIn("mem-jwt", curr_disp.relationships.get("conflicts_with", []))

    def test_distinct_new_information(self) -> None:
        """Rule A: Separate distinct knowledge preserves original memory and creates separate new memory."""
        existing = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL as its database.",
            project_id="proj-distinct",
        )
        incoming = Memory.create(
            memory_type=MemoryType.PREFERENCE,
            content="The user prefers concise code.",
            project_id="proj-distinct",
        )

        result = self.updater.update(existing, incoming)
        self.assertTrue(result.created)
        self.assertFalse(result.superseded)
        self.assertFalse(result.duplicate)
        self.assertFalse(result.conflict)
        self.assertEqual(result.action, UpdateAction.CREATED_NEW)
        self.assertEqual(result.current_memory.content, incoming.content)

    def test_provenance_preservation_across_updates(self) -> None:
        """Verify source event and agent IDs are consolidated across versions."""
        v1 = Memory.create(
            memory_type=MemoryType.FACT,
            content="Initial fact",
            project_id="proj-prov",
            source_event_ids=["evt-1"],
            source_agent_ids=["claude"],
        )
        v2_candidate = Memory.create(
            memory_type=MemoryType.FACT,
            content="Updated fact",
            project_id="proj-prov",
            source_event_ids=["evt-2"],
            source_agent_ids=["gemini"],
        )

        res = self.updater.update(v1, v2_candidate, strategy="supersede")
        v2 = res.current_memory
        self.assertIsNotNone(v2)
        # Consolidated provenance
        self.assertEqual(set(v2.source_event_ids), {"evt-1", "evt-2"})
        self.assertEqual(set(v2.source_agent_ids), {"claude", "gemini"})

    def test_invalid_update_inputs(self) -> None:
        """Test validation error handling on invalid update inputs."""
        valid_mem = Memory.create(
            memory_type=MemoryType.FACT,
            content="Some fact",
            project_id="proj-1",
        )

        # Non-memory existing
        with self.assertRaises(InvalidUpdateOperationError):
            self.updater.update("not-a-memory", valid_mem)  # type: ignore

        # Non-memory incoming
        with self.assertRaises(InvalidUpdateOperationError):
            self.updater.update(valid_mem, 12345)  # type: ignore

        # Cross-project update
        other_proj_mem = Memory.create(
            memory_type=MemoryType.FACT,
            content="Some other fact",
            project_id="proj-2",
        )
        with self.assertRaises(InvalidUpdateOperationError):
            self.updater.update(valid_mem, other_proj_mem)

    def test_update_many_batch_operations(self) -> None:
        """Test processing multiple incoming memories against known existing memories."""
        m_db = Memory.create(
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
            project_id="proj-batch",
        )
        m_pref = Memory.create(
            memory_type=MemoryType.PREFERENCE,
            content="The user prefers concise code.",
            project_id="proj-batch",
        )

        # Batch with 1 update (database), 1 duplicate (preference), 1 new (caching)
        incoming_batch = [
            Memory.create(
                memory_type=MemoryType.FACT,
                content="The project uses PostgreSQL as its database.",
                project_id="proj-batch",
            ),
            Memory.create(
                memory_type=MemoryType.PREFERENCE,
                content="The user prefers concise code.",
                project_id="proj-batch",
            ),
            Memory.create(
                memory_type=MemoryType.DECISION,
                content="Redis was removed because it caused stale cache problems.",
                project_id="proj-batch",
            ),
        ]

        results = self.updater.update_many(incoming_batch, existing_memories=[m_db, m_pref])
        self.assertEqual(len(results), 3)

        # Item 0 was superseded (MySQL -> PostgreSQL)
        self.assertEqual(results[0].action, UpdateAction.SUPERSEDED)
        self.assertTrue(results[0].superseded)

        # Item 1 was duplicate
        self.assertEqual(results[1].action, UpdateAction.DUPLICATE_NOOP)
        self.assertTrue(results[1].duplicate)

        # Item 2 was created new
        self.assertEqual(results[2].action, UpdateAction.CREATED_NEW)
        self.assertTrue(results[2].created)

    def test_version_tracker_and_history_inspection(self) -> None:
        """Test MemoryVersionHistory inspection methods."""
        m1 = Memory.create(memory_id="m-v1", memory_type=MemoryType.FACT, content="C1", project_id="p")
        history = MemoryVersionHistory(m1.root_memory_id)
        history.add_version(m1)

        m2 = m1.with_updates(memory_id="m-v2", content="C2", version=2, parent_memory_id="m-v1")
        history.add_version(m2)

        self.assertEqual(history.version_count, 2)
        self.assertEqual(history.get_version(1).memory_id, "m-v1")
        self.assertEqual(history.get_version(2).memory_id, "m-v2")
        self.assertIsNone(history.get_version(99))
        self.assertEqual(history.get_by_id("m-v2").content, "C2")
        self.assertTrue(history.is_current("m-v2"))
        self.assertFalse(history.is_current("m-v1"))

        # Version conflict: non-monotonic version or bad parent
        m_bad = m1.with_updates(memory_id="m-bad", version=1)
        with self.assertRaises(VersionConflictError):
            history.add_version(m_bad)

    def test_serialization_compatibility_of_versioned_memories(self) -> None:
        """Test that memories with root_memory_id, parent_memory_id, and version roundtrip properly."""
        mem = Memory.create(
            memory_id="mem-test-serial-v2",
            memory_type=MemoryType.DECISION,
            content="Versioned decision",
            project_id="p-ser",
            root_memory_id="mem-test-serial-v1",
            parent_memory_id="mem-test-serial-v1",
            version=2,
            relationships={"supersedes": "mem-test-serial-v1"},
        )

        d = mem.to_dict()
        self.assertEqual(d["root_memory_id"], "mem-test-serial-v1")
        self.assertEqual(d["parent_memory_id"], "mem-test-serial-v1")
        self.assertEqual(d["version"], 2)

        reconstructed = Memory.from_dict(d)
        self.assertEqual(reconstructed.memory_id, mem.memory_id)
        self.assertEqual(reconstructed.root_memory_id, "mem-test-serial-v1")
        self.assertEqual(reconstructed.parent_memory_id, "mem-test-serial-v1")
        self.assertEqual(reconstructed.version, 2)

        # JSON roundtrip
        j = mem.to_json()
        reconstructed_j = Memory.from_json(j)
        self.assertEqual(reconstructed_j.version, 2)
        self.assertEqual(reconstructed_j.parent_memory_id, "mem-test-serial-v1")


if __name__ == "__main__":
    unittest.main()

