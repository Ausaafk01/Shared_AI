"""
Automated tests for Event System extensibility (custom event types).
"""

import tempfile
from pathlib import Path
import unittest

from ai_brain.events.exceptions import EventValidationError
from ai_brain.events.models import Event
from ai_brain.events.storage.sqlite import SQLiteEventStore
from ai_brain.events.types import (
    is_valid_event_type,
    register_event_type,
    unregister_event_type,
)


class TestEventExtensibility(unittest.TestCase):
    """Tests covering custom event type registration, validation, and storage."""

    def tearDown(self) -> None:
        unregister_event_type("CUSTOM_AUDIT_LOG")
        unregister_event_type("MEMORY_SNAPSHOT")

    def test_custom_event_type_registration(self) -> None:
        """Test registering new event types dynamically."""
        self.assertFalse(is_valid_event_type("CUSTOM_AUDIT_LOG"))

        registered = register_event_type("CUSTOM_AUDIT_LOG")
        self.assertEqual(registered, "CUSTOM_AUDIT_LOG")
        self.assertTrue(is_valid_event_type("CUSTOM_AUDIT_LOG"))

    def test_invalid_custom_event_type_names(self) -> None:
        """Test that malformed event type names are rejected."""
        with self.assertRaises(ValueError):
            register_event_type("")

        with self.assertRaises(ValueError):
            register_event_type("   ")

        with self.assertRaises(ValueError):
            register_event_type("INVALID EVENT WITH SPACES")

        with self.assertRaises(ValueError):
            register_event_type("BAD#CHAR$")

    def test_create_and_store_custom_event(self) -> None:
        """Test creating, storing, and retrieving an event with a registered custom type."""
        register_event_type("MEMORY_SNAPSHOT")

        event = Event.create(
            event_type="MEMORY_SNAPSHOT",
            project_id="proj-memory",
            session_id="sess-snap",
            actor="agent:gemini",
            input={"nodes_count": 150},
        )
        self.assertEqual(event.event_type, "MEMORY_SNAPSHOT")

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_ext.db"
            with SQLiteEventStore(db_path) as store:
                store.store_event(event)

                retrieved = store.get_event_by_id(event.event_id)
                self.assertIsNotNone(retrieved)
                self.assertEqual(retrieved.event_type, "MEMORY_SNAPSHOT")

                type_events = store.get_events_by_type("MEMORY_SNAPSHOT")
                self.assertEqual(len(type_events), 1)
                self.assertEqual(type_events[0].event_id, event.event_id)

    def test_unregistered_event_type_rejected(self) -> None:
        """Test that unregistered custom event types are rejected by Event.create."""
        with self.assertRaises(EventValidationError):
            Event.create(
                event_type="UNREGISTERED_ACTION",
                project_id="proj",
                session_id="sess",
                actor="user",
            )


if __name__ == "__main__":
    unittest.main()

