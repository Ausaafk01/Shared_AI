"""
Automated tests for persistent event storage, retrieval, ordering, filtering, and safety.
"""

from datetime import datetime, timezone, timedelta
import gc
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
import warnings

from ai_brain.agents.registry.sqlite import SQLiteAgentRegistryStore
from ai_brain.agents.sessions.storage.sqlite import SQLiteSessionStore
from ai_brain.events.exceptions import (
    DuplicateEventError,
    StorageCorruptionError,
    StorageError,
)
from ai_brain.events.models import Event
from ai_brain.events.storage.sqlite import SQLiteEventStore
from ai_brain.events.types import EventType
from ai_brain.loop.storage.sqlite import SQLiteInteractionStore
from ai_brain.ml.feedback.store import SQLiteFeedbackStore
from ai_brain.vectors.storage.sqlite import SQLiteVectorStore


class TestSQLiteEventStore(unittest.TestCase):
    """Tests for SQLiteEventStore operations, persistence, and append-only constraints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"
        self.store = SQLiteEventStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    def test_store_and_retrieve_by_id(self) -> None:
        """Test storing an event and retrieving it by event_id."""
        event = Event.create(
            event_id="evt-100",
            event_type=EventType.USER_MESSAGE,
            project_id="proj-1",
            session_id="sess-1",
            actor="user",
            input={"query": "Explain event sourcing"},
            metadata={"source": "cli"},
        )
        self.store.store_event(event)

        retrieved = self.store.get_event_by_id("evt-100")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.event_id, "evt-100")
        self.assertEqual(retrieved.event_type, "USER_MESSAGE")
        self.assertEqual(retrieved.project_id, "proj-1")
        self.assertEqual(retrieved.actor, "user")
        self.assertEqual(dict(retrieved.input), {"query": "Explain event sourcing"})
        self.assertEqual(retrieved.metadata["source"], "cli")

        # Non-existent ID returns None
        self.assertIsNone(self.store.get_event_by_id("non-existent-id"))

    def test_persistence_across_store_restarts(self) -> None:
        """Test that events persist on disk after the store is closed and reopened."""
        event1 = Event.create(
            event_id="evt-p1",
            event_type=EventType.SESSION_START,
            project_id="proj-persist",
            session_id="sess-persist",
            actor="system",
        )
        event2 = Event.create(
            event_id="evt-p2",
            event_type=EventType.SESSION_END,
            project_id="proj-persist",
            session_id="sess-persist",
            actor="system",
        )
        self.store.store_event(event1)
        self.store.store_event(event2)
        self.store.close()

        # Reopen with a new instance on the same file path
        reopened_store = SQLiteEventStore(self.db_path)
        try:
            self.assertEqual(reopened_store.count(), 2)
            retrieved1 = reopened_store.get_event_by_id("evt-p1")
            retrieved2 = reopened_store.get_event_by_id("evt-p2")
            self.assertIsNotNone(retrieved1)
            self.assertIsNotNone(retrieved2)
            self.assertEqual(retrieved1.event_type, "SESSION_START")
            self.assertEqual(retrieved2.event_type, "SESSION_END")
        finally:
            reopened_store.close()

    def test_duplicate_id_handling(self) -> None:
        """Test that storing an event with a duplicate ID raises DuplicateEventError."""
        event = Event.create(
            event_id="dup-id",
            event_type=EventType.TASK_CREATED,
            project_id="proj-dup",
            session_id="sess-dup",
            actor="agent",
        )
        self.store.store_event(event)

        # Attempt to store identical event or different event with same ID
        duplicate_event = Event.create(
            event_id="dup-id",
            event_type=EventType.TASK_COMPLETED,
            project_id="proj-different",
            session_id="sess-different",
            actor="user",
        )
        with self.assertRaises(DuplicateEventError):
            self.store.store_event(duplicate_event)

        # Ensure the original event was NOT overwritten
        original = self.store.get_event_by_id("dup-id")
        self.assertEqual(original.event_type, "TASK_CREATED")
        self.assertEqual(original.project_id, "proj-dup")

    def test_chronological_ordering(self) -> None:
        """Test that retrieval preserves chronological order based on timestamp."""
        t0 = datetime(2026, 11, 2, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 2, 10, 1, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 2, 10, 2, 0, tzinfo=timezone.utc)

        e1 = Event.create(
            event_id="e1",
            event_type=EventType.USER_MESSAGE,
            timestamp=t0,
            project_id="proj-order",
            session_id="sess-order",
            actor="user",
        )
        e2 = Event.create(
            event_id="e2",
            event_type=EventType.AGENT_MESSAGE,
            timestamp=t1,
            project_id="proj-order",
            session_id="sess-order",
            actor="agent:codex",
        )
        e3 = Event.create(
            event_id="e3",
            event_type=EventType.DECISION,
            timestamp=t2,
            project_id="proj-order",
            session_id="sess-order",
            actor="agent:codex",
        )

        # Store in scrambled order
        self.store.store_event(e2)
        self.store.store_event(e3)
        self.store.store_event(e1)

        # Retrieve all ascending
        events_asc = self.store.get_all_events(ascending=True)
        self.assertEqual([e.event_id for e in events_asc], ["e1", "e2", "e3"])

        # Retrieve all descending
        events_desc = self.store.get_all_events(ascending=False)
        self.assertEqual([e.event_id for e in events_desc], ["e3", "e2", "e1"])

    def test_retrieval_by_project(self) -> None:
        """Test retrieving events filtered by project ID."""
        e1 = Event.create(event_id="p1-1", event_type=EventType.USER_MESSAGE, project_id="proj-A", session_id="s1", actor="u")
        e2 = Event.create(event_id="p1-2", event_type=EventType.AGENT_MESSAGE, project_id="proj-A", session_id="s1", actor="a")
        e3 = Event.create(event_id="p2-1", event_type=EventType.USER_MESSAGE, project_id="proj-B", session_id="s2", actor="u")
        self.store.store_event(e1)
        self.store.store_event(e2)
        self.store.store_event(e3)

        proj_a_events = self.store.get_events_by_project("proj-A")
        self.assertEqual(len(proj_a_events), 2)
        self.assertEqual([e.event_id for e in proj_a_events], ["p1-1", "p1-2"])

        proj_b_events = self.store.get_events_by_project("proj-B")
        self.assertEqual(len(proj_b_events), 1)
        self.assertEqual(proj_b_events[0].event_id, "p2-1")

    def test_retrieval_by_agent(self) -> None:
        """Test retrieving events filtered by agent ID."""
        e1 = Event.create(event_id="a1", event_type=EventType.AGENT_MESSAGE, project_id="p", agent_id="agent-claude", session_id="s", actor="a")
        e2 = Event.create(event_id="a2", event_type=EventType.TOOL_CALL, project_id="p", agent_id="agent-codex", session_id="s", actor="a")
        e3 = Event.create(event_id="a3", event_type=EventType.AGENT_MESSAGE, project_id="p", agent_id="agent-claude", session_id="s", actor="a")
        self.store.store_event(e1)
        self.store.store_event(e2)
        self.store.store_event(e3)

        claude_events = self.store.get_events_by_agent("agent-claude")
        self.assertEqual([e.event_id for e in claude_events], ["a1", "a3"])

        codex_events = self.store.get_events_by_agent("agent-codex")
        self.assertEqual([e.event_id for e in codex_events], ["a2"])

    def test_retrieval_by_session(self) -> None:
        """Test retrieving events filtered by session ID."""
        e1 = Event.create(event_id="s1-1", event_type=EventType.SESSION_START, project_id="p", session_id="sess-alpha", actor="sys")
        e2 = Event.create(event_id="s2-1", event_type=EventType.SESSION_START, project_id="p", session_id="sess-beta", actor="sys")
        self.store.store_event(e1)
        self.store.store_event(e2)

        session_events = self.store.get_events_by_session("sess-alpha")
        self.assertEqual(len(session_events), 1)
        self.assertEqual(session_events[0].event_id, "s1-1")

    def test_retrieval_by_type(self) -> None:
        """Test retrieving events filtered by event type."""
        e1 = Event.create(event_id="t1", event_type=EventType.ERROR, project_id="p", session_id="s", actor="agent")
        e2 = Event.create(event_id="t2", event_type=EventType.TOOL_CALL, project_id="p", session_id="s", actor="agent")
        e3 = Event.create(event_id="t3", event_type=EventType.ERROR, project_id="p", session_id="s", actor="agent")
        self.store.store_event(e1)
        self.store.store_event(e2)
        self.store.store_event(e3)

        errors = self.store.get_events_by_type(EventType.ERROR)
        self.assertEqual([e.event_id for e in errors], ["t1", "t3"])

    def test_multi_criteria_filtering(self) -> None:
        """Test filtering across combinations of project, agent, type, time range, limit, offset."""
        base_time = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            self.store.store_event(
                Event.create(
                    event_id=f"filt-{i}",
                    event_type=EventType.TOOL_RESULT if i % 2 == 0 else EventType.AGENT_MESSAGE,
                    timestamp=base_time + timedelta(minutes=i),
                    project_id="proj-filter",
                    agent_id="agent-007",
                    session_id="sess-filter",
                    actor="agent:007",
                )
            )

        # Filter by time range
        window_events = self.store.filter_events(
            project_id="proj-filter",
            start_time=base_time + timedelta(minutes=1),
            end_time=base_time + timedelta(minutes=3),
        )
        self.assertEqual([e.event_id for e in window_events], ["filt-1", "filt-2", "filt-3"])

        # Filter with pagination (limit and offset)
        page1 = self.store.filter_events(project_id="proj-filter", limit=2, offset=0)
        self.assertEqual([e.event_id for e in page1], ["filt-0", "filt-1"])
        page2 = self.store.filter_events(project_id="proj-filter", limit=2, offset=2)
        self.assertEqual([e.event_id for e in page2], ["filt-2", "filt-3"])

    def test_append_only_database_trigger_protection(self) -> None:
        """Test that database triggers strictly prohibit UPDATE and DELETE SQL operations."""
        event = Event.create(
            event_id="protected-evt",
            event_type=EventType.USER_MESSAGE,
            project_id="p",
            session_id="s",
            actor="u",
        )
        self.store.store_event(event)

        # Attempt direct SQL UPDATE on events table
        with self.assertRaises(sqlite3.DatabaseError) as ctx:
            self.store._conn.execute("UPDATE events SET project_id = 'hacked' WHERE event_id = 'protected-evt';")
        self.assertIn("prohibited", str(ctx.exception).lower())

        # Attempt direct SQL DELETE on events table
        with self.assertRaises(sqlite3.DatabaseError) as ctx:
            self.store._conn.execute("DELETE FROM events WHERE event_id = 'protected-evt';")
        self.assertIn("prohibited", str(ctx.exception).lower())

        # Verify record remains intact
        rechecked = self.store.get_event_by_id("protected-evt")
        self.assertIsNotNone(rechecked)
        self.assertEqual(rechecked.project_id, "p")

    def test_storage_closed_error_handling(self) -> None:
        """Test operations on a closed store raise StorageError."""
        self.store.close()
        event = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="p",
            session_id="s",
            actor="u",
        )
        with self.assertRaises(StorageError):
            self.store.store_event(event)

        with self.assertRaises(StorageError):
            self.store.get_event_by_id("some-id")

        with self.assertRaises(StorageError):
            self.store.get_all_events()

    def test_storage_corruption_handling(self) -> None:
        """Test safe handling of corrupt database records."""
        event = Event.create(
            event_id="valid-id",
            event_type=EventType.USER_MESSAGE,
            project_id="p",
            session_id="s",
            actor="u",
        )
        self.store.store_event(event)

        # Inject a raw corrupted JSON record using a raw bypass
        # Triggers only prevent UPDATE and DELETE; we insert a corrupted row directly to test recovery
        cur = self.store._conn.cursor()
        cur.execute("""
        INSERT INTO events (
            event_id, event_type, timestamp, project_id, session_id, actor,
            input_json, created_at
        ) VALUES (
            'corrupt-id', 'USER_MESSAGE', 'not-a-valid-date', 'p', 's', 'u',
            'not-valid-json', '2026-11-02T00:00:00Z'
        );
        """)
        cur.close()

        # By default, accessing corrupt record raises StorageCorruptionError
        with self.assertRaises(StorageCorruptionError):
            self.store.get_event_by_id("corrupt-id")

        # When ignore_corrupt is True, corrupt record is skipped without crashing
        self.store.close()
        tolerant_store = SQLiteEventStore(self.db_path, ignore_corrupt=True)
        try:
            events = tolerant_store.get_all_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_id, "valid-id")
        finally:
            tolerant_store.close()


class TestSQLiteResourceManagementAndLifecycle(unittest.TestCase):
    """Regression tests verifying context managers and GC connection cleanup without ResourceWarning."""

    def test_sqlite_stores_context_manager_and_gc_cleanup(self) -> None:
        """Verify context manager and finalizer close connections without leaking or warning."""
        store_factories = [
            lambda: SQLiteEventStore(":memory:"),
            lambda: SQLiteInteractionStore(":memory:"),
            lambda: SQLiteVectorStore(":memory:"),
            lambda: SQLiteAgentRegistryStore(":memory:"),
            lambda: SQLiteSessionStore(":memory:"),
            lambda: SQLiteFeedbackStore(":memory:"),
        ]

        for factory in store_factories:
            # 1. Test context manager closes store
            with factory() as store:
                self.assertFalse(store._closed)
            self.assertTrue(store._closed)

            # 2. Test GC deallocation does not emit ResourceWarning
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ResourceWarning)
                unreferenced_store = factory()
                del unreferenced_store
                gc.collect()
                res_warns = [w for w in captured if issubclass(w.category, ResourceWarning)]
                self.assertEqual(len(res_warns), 0, f"ResourceWarning emitted during GC for {factory}")


if __name__ == "__main__":
    unittest.main()

