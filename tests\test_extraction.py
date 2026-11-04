"""
Automated tests for AI Brain Memory Extraction component.
"""

from datetime import datetime, timezone, timedelta
import unittest

from ai_brain.events.models import Event
from ai_brain.events.types import EventType
from ai_brain.memories.exceptions import (
    ExtractionRuleError,
    InvalidEventError,
)
from ai_brain.memories.extraction import (
    ExtractionRule,
    FactRule,
    MemoryExtractor,
)
from ai_brain.memories.extraction.helpers import CandidateMemory
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType


class TestMemoryExtraction(unittest.TestCase):
    """Tests covering single and multi-event memory extraction, rules, and error handling."""

    def setUp(self) -> None:
        self.extractor = MemoryExtractor()
        self.now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)

    def test_single_event_extraction_fact(self) -> None:
        """Test extracting a FACT memory from a single event."""
        event = Event.create(
            event_id="evt-fact-1",
            event_type=EventType.USER_MESSAGE,
            timestamp=self.now,
            project_id="proj-db",
            agent_id="user-agent",
            session_id="sess-1",
            actor="user",
            input={"message": "The project uses PostgreSQL as its database."},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "FACT")
        self.assertEqual(mem.content, "The project uses PostgreSQL as its database.")
        self.assertEqual(mem.project_id, "proj-db")
        self.assertEqual(mem.source_event_ids, ("evt-fact-1",))
        self.assertEqual(mem.source_agent_ids, ("user-agent",))
        self.assertEqual(mem.status, "ACTIVE")
        self.assertEqual(mem.created_at, self.now)
        self.assertEqual(mem.updated_at, self.now)
        self.assertGreaterEqual(mem.confidence, 0.9)

    def test_single_event_extraction_decision(self) -> None:
        """Test extracting a DECISION memory from an event with explicit reasoning."""
        event = Event.create(
            event_id="evt-dec-1",
            event_type=EventType.DECISION,
            timestamp=self.now,
            project_id="proj-cache",
            agent_id="agent-claude",
            session_id="sess-cache",
            actor="agent:claude",
            metadata={
                "decision": "Redis was removed",
                "reason": "it caused stale cache problems",
            },
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "DECISION")
        self.assertEqual(mem.content, "Redis was removed because it caused stale cache problems.")
        self.assertEqual(mem.source_event_ids, ("evt-dec-1",))
        self.assertEqual(mem.source_agent_ids, ("agent-claude",))
        self.assertEqual(mem.importance, 0.9)

    def test_single_event_extraction_preference(self) -> None:
        """Test extracting a PREFERENCE memory from user preference statement."""
        event = Event.create(
            event_id="evt-pref-1",
            event_type=EventType.USER_MESSAGE,
            timestamp=self.now,
            project_id="proj-pref",
            session_id="sess-pref",
            actor="user",
            input={"message": "The user prefers concise code."},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "PREFERENCE")
        self.assertEqual(mem.content, "The user prefers concise code.")

    def test_single_event_extraction_task(self) -> None:
        """Test extracting a TASK memory from an unresolved bug event."""
        event = Event.create(
            event_id="evt-task-1",
            event_type=EventType.TASK_CREATED,
            timestamp=self.now,
            project_id="proj-bug",
            session_id="sess-bug",
            actor="system",
            output={"message": "The authentication bug remains unresolved."},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "TASK")
        self.assertEqual(mem.content, "The authentication bug remains unresolved.")
        self.assertEqual(mem.status, "ACTIVE")

    def test_task_created_produces_active_task(self) -> None:
        """Verify that TASK_CREATED produces an ACTIVE task memory."""
        event = Event.create(
            event_id="evt-task-created-1",
            event_type=EventType.TASK_CREATED,
            timestamp=self.now,
            project_id="proj-tasks",
            session_id="sess-tasks",
            actor="agent:codex",
            metadata={"task": "Implement JWT token refresh endpoint"},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "TASK")
        self.assertEqual(mem.status, "ACTIVE")
        self.assertEqual(mem.content, "Implement JWT token refresh endpoint.")
        self.assertFalse(mem.metadata.get("completed", False))

    def test_task_completed_never_produces_active_task(self) -> None:
        """Verify that TASK_COMPLETED never produces an ACTIVE task memory."""
        event = Event.create(
            event_id="evt-task-completed-1",
            event_type=EventType.TASK_COMPLETED,
            timestamp=self.now,
            project_id="proj-tasks",
            session_id="sess-tasks",
            actor="agent:codex",
            metadata={"task": "Implement JWT token refresh endpoint"},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "TASK")
        # Must never be ACTIVE
        self.assertNotEqual(mem.status, "ACTIVE")
        self.assertEqual(mem.status, "ARCHIVED")

    def test_completed_task_history_preserved_as_archived(self) -> None:
        """Verify completed task history is preserved as ARCHIVED and not mistaken for active work."""
        event = Event.create(
            event_id="evt-task-completed-history",
            event_type=EventType.TASK_COMPLETED,
            timestamp=self.now,
            project_id="proj-tasks",
            session_id="sess-tasks",
            actor="system",
            output={"message": "Task completed: Database migration to PostgreSQL."},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "TASK")
        self.assertEqual(mem.status, "ARCHIVED")
        self.assertTrue(mem.metadata.get("completed", False))
        self.assertIn("Completed", mem.content)
        self.assertEqual(mem.source_event_ids, ("evt-task-completed-history",))
        # Ensure that filtering for active tasks would exclude this completed task
        active_tasks = [m for m in memories if m.status == MemoryStatus.ACTIVE.value]
        self.assertEqual(len(active_tasks), 0)

    def test_single_event_extraction_error(self) -> None:
        """Test extracting an ERROR memory from an error event."""
        event = Event.create(
            event_id="evt-err-1",
            event_type=EventType.ERROR,
            timestamp=self.now,
            project_id="proj-err",
            session_id="sess-err",
            actor="agent:codex",
            tool_used="bash_runner",
            files_involved=["scripts/deploy.sh"],
            result={"error": "Connection refused to remote host", "exit_code": 1},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "ERROR")
        self.assertIn("Connection refused", mem.content)
        self.assertIn("bash_runner", mem.content)

    def test_single_event_extraction_attempt(self) -> None:
        """Test extracting an ATTEMPT memory from a tool execution result."""
        event = Event.create(
            event_id="evt-att-1",
            event_type=EventType.TOOL_RESULT,
            timestamp=self.now,
            project_id="proj-att",
            session_id="sess-att",
            actor="agent:codex",
            tool_used="edit_file",
            files_involved=["ai_brain/events/models.py"],
            result={"status": "success"},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "ATTEMPT")
        self.assertIn("edit_file", mem.content)
        self.assertIn("success", mem.content)

    def test_single_event_extraction_code_knowledge(self) -> None:
        """Test extracting CODE_KNOWLEDGE from code modification event."""
        event = Event.create(
            event_id="evt-code-1",
            event_type=EventType.CODE_CHANGE,
            timestamp=self.now,
            project_id="proj-code",
            session_id="sess-code",
            actor="agent:antigravity",
            files_involved=["ai_brain/events/models.py", "ai_brain/events/storage/sqlite.py"],
            metadata={"pattern": "append-only immutable event stream"},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "CODE_KNOWLEDGE")
        self.assertIn("append-only immutable event stream", mem.content)

    def test_single_event_extraction_context(self) -> None:
        """Test extracting CONTEXT memory from session environment info."""
        event = Event.create(
            event_id="evt-ctx-1",
            event_type=EventType.SESSION_START,
            timestamp=self.now,
            project_id="proj-ctx",
            session_id="sess-ctx",
            actor="system",
            metadata={"os": "windows", "python_version": "3.13"},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.memory_type, "CONTEXT")
        self.assertIn("windows", mem.content)
        self.assertIn("3.13", mem.content)

    def test_multiple_related_events_extraction(self) -> None:
        """
        Test extracting a consolidated decision memory from a multi-event sequence:
        Event 1: Agent proposes PostgreSQL.
        Event 2: User approves PostgreSQL.
        Event 3: Agent configures PostgreSQL.
        """
        t0 = self.now
        t1 = self.now + timedelta(minutes=1)
        t2 = self.now + timedelta(minutes=2)

        e1 = Event.create(
            event_id="evt-seq-1",
            event_type=EventType.AGENT_MESSAGE,
            timestamp=t0,
            project_id="proj-pg",
            agent_id="claude",
            session_id="sess-pg-setup",
            actor="agent:claude",
            output={"message": "We propose using PostgreSQL for relational data storage."},
        )
        e2 = Event.create(
            event_id="evt-seq-2",
            event_type=EventType.USER_MESSAGE,
            timestamp=t1,
            project_id="proj-pg",
            session_id="sess-pg-setup",
            actor="user",
            input={"message": "Approved: configure PostgreSQL."},
        )
        e3 = Event.create(
            event_id="evt-seq-3",
            event_type=EventType.TOOL_CALL,
            timestamp=t2,
            project_id="proj-pg",
            agent_id="codex",
            session_id="sess-pg-setup",
            actor="agent:codex",
            tool_used="bash_runner",
            input={"command": "docker run --name pg-test -e POSTGRES_PASSWORD=secret -d postgres"},
        )

        memories = self.extractor.extract_many([e1, e2, e3])
        # Find the consolidated decision memory
        decision_memories = [m for m in memories if m.memory_type == "DECISION"]
        self.assertGreaterEqual(len(decision_memories), 1)

        dec_mem = decision_memories[0]
        self.assertEqual(dec_mem.content, "Decided to use PostgreSQL.")
        self.assertEqual(dec_mem.project_id, "proj-pg")
        self.assertEqual(dec_mem.created_at, t0)
        self.assertEqual(dec_mem.updated_at, t2)
        # Provenance includes contributing events
        self.assertIn("evt-seq-1", dec_mem.source_event_ids)
        self.assertIn("evt-seq-2", dec_mem.source_event_ids)
        self.assertIn("evt-seq-3", dec_mem.source_event_ids)

    def test_events_producing_no_memory(self) -> None:
        """Test that casual chat, speculative assertions, and transient requests produce no memories."""
        # 1. Casual chit-chat
        e_casual = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="proj-noop",
            session_id="s1",
            actor="user",
            input={"message": "Hello! Sounds good, thanks."},
        )
        self.assertEqual(self.extractor.extract(e_casual), [])

        # 2. Speculation / unverified assumptions
        e_spec = Event.create(
            event_type=EventType.AGENT_MESSAGE,
            project_id="proj-noop",
            session_id="s1",
            actor="agent:gemini",
            output={"message": "Maybe we could use Redis, assuming we have enough RAM."},
        )
        self.assertEqual(self.extractor.extract(e_spec), [])

        # 3. Transient preference (should NOT become permanent preference memory)
        e_transient = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="proj-noop",
            session_id="s1",
            actor="user",
            input={"message": "For now, I prefer to just use sqlite temporarily."},
        )
        self.assertEqual(self.extractor.extract(e_transient), [])

    def test_repeated_information_deduplication(self) -> None:
        """Test that identical statements in the same batch are merged without creating duplicate candidates."""
        t1 = self.now
        t2 = self.now + timedelta(minutes=5)

        e1 = Event.create(
            event_id="evt-dup-1",
            event_type=EventType.USER_MESSAGE,
            timestamp=t1,
            project_id="proj-dedup",
            agent_id="agent-1",
            session_id="s1",
            actor="user",
            input={"message": "The project uses PostgreSQL as its database."},
        )
        e2 = Event.create(
            event_id="evt-dup-2",
            event_type=EventType.USER_MESSAGE,
            timestamp=t2,
            project_id="proj-dedup",
            agent_id="agent-2",
            session_id="s2",
            actor="user",
            input={"message": "The project uses PostgreSQL as its database."},
        )

        memories = self.extractor.extract_many([e1, e2])
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        self.assertEqual(mem.content, "The project uses PostgreSQL as its database.")
        # Provenance combined
        self.assertEqual(mem.source_event_ids, ("evt-dup-1", "evt-dup-2"))
        self.assertIn("agent-1", mem.source_agent_ids)
        self.assertIn("agent-2", mem.source_agent_ids)
        # Timestamps span earliest to latest
        self.assertEqual(mem.created_at, t1)
        self.assertEqual(mem.updated_at, t2)

    def test_conflicting_information_handling(self) -> None:
        """Test that contradictory assertions in the same batch are preserved as DISPUTED."""
        e1 = Event.create(
            event_id="evt-conf-1",
            event_type=EventType.USER_MESSAGE,
            timestamp=self.now,
            project_id="proj-conflict",
            session_id="s1",
            actor="user",
            input={"message": "The project uses PostgreSQL as its database."},
        )
        e2 = Event.create(
            event_id="evt-conf-2",
            event_type=EventType.USER_MESSAGE,
            timestamp=self.now + timedelta(seconds=10),
            project_id="proj-conflict",
            session_id="s2",
            actor="user",
            input={"message": "The project uses MySQL as its database."},
            metadata={"fact": "The project uses MySQL as its database."},
        )

        memories = self.extractor.extract_many([e1, e2])
        self.assertEqual(len(memories), 2)

        # Both memories are preserved
        statuses = {m.status for m in memories}
        self.assertEqual(statuses, {MemoryStatus.DISPUTED.value})

        # Confidence is lowered and conflicts are documented
        for m in memories:
            self.assertLessEqual(m.confidence, 0.5)
            self.assertIn("conflicts_with", m.relationships)

    def test_multiple_projects_kept_separate(self) -> None:
        """Test that events from different projects are never merged."""
        e_proj1 = Event.create(
            event_id="p1-evt",
            event_type=EventType.USER_MESSAGE,
            timestamp=self.now,
            project_id="project-alpha",
            session_id="s1",
            actor="user",
            input={"message": "The project uses PostgreSQL as its database."},
        )
        e_proj2 = Event.create(
            event_id="p2-evt",
            event_type=EventType.USER_MESSAGE,
            timestamp=self.now,
            project_id="project-beta",
            session_id="s1",
            actor="user",
            input={"message": "The project uses PostgreSQL as its database."},
        )

        memories = self.extractor.extract_many([e_proj1, e_proj2])
        self.assertEqual(len(memories), 2)

        proj_ids = {m.project_id for m in memories}
        self.assertEqual(proj_ids, {"project-alpha", "project-beta"})

    def test_empty_collections(self) -> None:
        """Test that passing empty event collections returns empty list."""
        self.assertEqual(self.extractor.extract_many([]), [])

    def test_invalid_event_inputs(self) -> None:
        """Test handling of invalid event objects."""
        # Non-event object passed to extract()
        with self.assertRaises(InvalidEventError):
            self.extractor.extract("not-an-event")  # type: ignore

        # Invalid item inside extract_many with default non-strict mode
        e_valid = Event.create(
            event_id="valid-e",
            event_type=EventType.USER_MESSAGE,
            project_id="p",
            session_id="s",
            actor="u",
            input={"message": "The project uses PostgreSQL as its database."},
        )
        memories = self.extractor.extract_many([e_valid, "invalid-item", None])  # type: ignore
        self.assertEqual(len(memories), 1)
        self.assertEqual(len(self.extractor.last_errors), 2)

        # Strict mode raises InvalidEventError
        strict_extractor = MemoryExtractor(strict=True)
        with self.assertRaises(InvalidEventError):
            strict_extractor.extract_many([e_valid, 12345])  # type: ignore

    def test_unsupported_event_type(self) -> None:
        """Test that events without matching rules produce no memories without error."""
        e_unsupported = Event.create(
            event_type=EventType.AGENT_SWITCH,
            project_id="p",
            session_id="s",
            actor="sys",
            metadata={"from": "claude", "to": "codex"},
        )
        self.assertEqual(self.extractor.extract(e_unsupported), [])

    def test_extraction_rule_failure_handling(self) -> None:
        """Test safe handling when an individual rule raises an unexpected exception."""
        class FailingRule(ExtractionRule):
            @property
            def rule_name(self) -> str:
                return "failing_rule"

            @property
            def target_memory_type(self) -> MemoryType:
                return MemoryType.FACT

            def can_extract(self, event: Event) -> bool:
                return True

            def extract_single(self, event: Event) -> list[CandidateMemory]:
                raise RuntimeError("Simulated rule failure")

        # Non-strict mode: logs, records error, continues
        fault_tolerant = MemoryExtractor(rules=[FailingRule(), FactRule()], strict=False)
        e = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="p",
            session_id="s",
            actor="u",
            input={"message": "The project uses PostgreSQL as its database."},
        )
        memories = fault_tolerant.extract(e)
        self.assertEqual(len(memories), 1)
        self.assertEqual(len(fault_tolerant.last_errors), 1)
        self.assertIsInstance(fault_tolerant.last_errors[0], ExtractionRuleError)

        # Strict mode: raises ExtractionRuleError
        strict_faulty = MemoryExtractor(rules=[FailingRule()], strict=True)
        with self.assertRaises(ExtractionRuleError):
            strict_faulty.extract(e)

    def test_serialization_compatibility_of_extracted_memories(self) -> None:
        """Test that all extracted Memory objects serialize cleanly to dict and JSON."""
        event = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="proj-serial",
            session_id="sess-serial",
            actor="user",
            input={"message": "Authentication is handled through JWT."},
        )
        memories = self.extractor.extract(event)
        self.assertEqual(len(memories), 1)

        mem = memories[0]
        # Dict roundtrip
        d = mem.to_dict()
        reconstructed = Memory.from_dict(d)
        self.assertEqual(reconstructed.memory_id, mem.memory_id)
        self.assertEqual(reconstructed.content, mem.content)

        # JSON roundtrip
        j = mem.to_json()
        reconstructed_j = Memory.from_json(j)
        self.assertEqual(reconstructed_j.memory_id, mem.memory_id)
        self.assertEqual(reconstructed_j.content, mem.content)


if __name__ == "__main__":
    unittest.main()

