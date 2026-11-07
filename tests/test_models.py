"""
Automated tests for the Event model, immutability, validation, and serialization.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone, timedelta
import json
from types import MappingProxyType
import unittest

import ai_brain.common
import ai_brain.common.freeze
from ai_brain.common.freeze import _freeze_container, _unfreeze_container
import ai_brain.context.models
import ai_brain.events.models
from ai_brain.events.exceptions import EventValidationError
from ai_brain.events.models import Event
from ai_brain.events.types import EventType
import ai_brain.graph.models
import ai_brain.loop.models
import ai_brain.memories.conflicts.models
import ai_brain.memories.models
import ai_brain.retrieval.models
import ai_brain.vectors.models


class TestEventModel(unittest.TestCase):
    """Tests covering Event model creation, validation, immutability, and serialization."""

    def test_event_creation_minimal(self) -> None:
        """Test creating an event with minimal required fields."""
        event = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="proj-alpha",
            session_id="sess-001",
            actor="user-42",
        )
        self.assertIsNotNone(event.event_id)
        self.assertEqual(len(event.event_id), 36)  # UUID4
        self.assertEqual(event.event_type, "USER_MESSAGE")
        self.assertEqual(event.project_id, "proj-alpha")
        self.assertEqual(event.session_id, "sess-001")
        self.assertEqual(event.actor, "user-42")
        self.assertIsNone(event.agent_id)
        self.assertIsNone(event.input)
        self.assertIsNone(event.output)
        self.assertIsNone(event.tool_used)
        self.assertIsNone(event.result)
        self.assertEqual(event.files_involved, ())
        self.assertEqual(dict(event.metadata), {})
        self.assertIsNotNone(event.timestamp)
        self.assertEqual(event.timestamp.tzinfo, timezone.utc)

    def test_event_creation_all_fields(self) -> None:
        """Test creating an event with all optional fields specified."""
        now = datetime.now(timezone.utc)
        event = Event.create(
            event_id="custom-id-123",
            event_type=EventType.TOOL_CALL,
            timestamp=now,
            project_id="proj-beta",
            agent_id="agent-gemini",
            session_id="sess-002",
            actor="agent:gemini",
            input={"command": "git status"},
            output={"stdout": "clean"},
            tool_used="bash_runner",
            files_involved=["repo/main.py", "repo/utils.py"],
            result={"exit_code": 0},
            metadata={"priority": "high", "nested": {"key": "val"}},
        )
        self.assertEqual(event.event_id, "custom-id-123")
        self.assertEqual(event.event_type, "TOOL_CALL")
        self.assertEqual(event.timestamp, now)
        self.assertEqual(event.project_id, "proj-beta")
        self.assertEqual(event.agent_id, "agent-gemini")
        self.assertEqual(event.session_id, "sess-002")
        self.assertEqual(event.actor, "agent:gemini")
        self.assertEqual(dict(event.input), {"command": "git status"})
        self.assertEqual(dict(event.output), {"stdout": "clean"})
        self.assertEqual(event.tool_used, "bash_runner")
        self.assertEqual(event.files_involved, ("repo/main.py", "repo/utils.py"))
        self.assertEqual(dict(event.result), {"exit_code": 0})
        self.assertEqual(event.metadata["priority"], "high")

    def test_required_fields_validation(self) -> None:
        """Test that missing or empty required fields raise EventValidationError."""
        # Empty project_id
        with self.assertRaises(EventValidationError):
            Event.create(
                event_type=EventType.USER_MESSAGE,
                project_id="",
                session_id="s1",
                actor="user",
            )

        # Empty session_id
        with self.assertRaises(EventValidationError):
            Event.create(
                event_type=EventType.USER_MESSAGE,
                project_id="p1",
                session_id="   ",
                actor="user",
            )

        # Empty actor
        with self.assertRaises(EventValidationError):
            Event.create(
                event_type=EventType.USER_MESSAGE,
                project_id="p1",
                session_id="s1",
                actor="",
            )

        # Empty event_id
        with self.assertRaises(EventValidationError):
            Event(
                event_id="  ",
                event_type="USER_MESSAGE",
                timestamp=datetime.now(timezone.utc),
                project_id="p1",
                session_id="s1",
                actor="user",
            )

    def test_event_type_validation(self) -> None:
        """Test that invalid event types are rejected."""
        with self.assertRaises(EventValidationError):
            Event.create(
                event_type="NON_EXISTENT_UNKNOWN_TYPE",
                project_id="p1",
                session_id="s1",
                actor="user",
            )

    def test_timestamp_handling(self) -> None:
        """Test UTC normalization for naive and non-UTC timezone-aware datetimes."""
        # Naive datetime should be assigned UTC timezone
        naive_dt = datetime(2026, 11, 2, 10, 30, 0)
        event_naive = Event.create(
            event_type=EventType.SESSION_START,
            project_id="p1",
            session_id="s1",
            actor="system",
            timestamp=naive_dt,
        )
        self.assertEqual(event_naive.timestamp.tzinfo, timezone.utc)
        self.assertEqual(event_naive.timestamp.hour, 10)

        # Offset-aware datetime (UTC+5:30) should be converted to UTC
        tz_offset = timezone(timedelta(hours=5, minutes=30))
        aware_dt = datetime(2026, 11, 2, 15, 30, 0, tzinfo=tz_offset)
        event_aware = Event.create(
            event_type=EventType.SESSION_START,
            project_id="p1",
            session_id="s1",
            actor="system",
            timestamp=aware_dt,
        )
        self.assertEqual(event_aware.timestamp.tzinfo, timezone.utc)
        self.assertEqual(event_aware.timestamp.hour, 10)
        self.assertEqual(event_aware.timestamp.minute, 0)

    def test_immutability_attribute_assignment(self) -> None:
        """Test that direct attribute assignment raises FrozenInstanceError."""
        event = Event.create(
            event_type=EventType.USER_MESSAGE,
            project_id="p1",
            session_id="s1",
            actor="user",
        )
        with self.assertRaises(FrozenInstanceError):
            event.project_id = "new_project"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            event.actor = "new_actor"  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            del event.session_id  # type: ignore

    def test_immutability_collection_fields(self) -> None:
        """Test that collections inside an event cannot be mutated after creation."""
        raw_files = ["a.py", "b.py"]
        raw_metadata = {"env": "test", "tags": ["tag1"]}
        event = Event.create(
            event_type=EventType.FILE_CHANGE,
            project_id="p1",
            session_id="s1",
            actor="agent",
            files_involved=raw_files,
            metadata=raw_metadata,
        )

        # Modifying original list outside does not affect event
        raw_files.append("c.py")
        self.assertEqual(event.files_involved, ("a.py", "b.py"))

        # Modifying original dict outside does not affect event
        raw_metadata["env"] = "modified"
        self.assertEqual(event.metadata["env"], "test")

        # Mutating files_involved directly fails
        with self.assertRaises(AttributeError):
            event.files_involved.append("c.py")  # type: ignore

        # Mutating metadata directly fails
        with self.assertRaises(TypeError):
            event.metadata["new_key"] = "val"  # type: ignore

    def test_serialization_to_dict_and_from_dict(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        event = Event.create(
            event_id="roundtrip-id",
            event_type=EventType.DECISION,
            project_id="proj-1",
            agent_id="agent-codex",
            session_id="sess-9",
            actor="agent:codex",
            input={"choice": "A"},
            output={"status": "accepted"},
            tool_used=None,
            files_involved=["plan.md"],
            result={"confidence": 0.95},
            metadata={"mode": "fast"},
        )

        event_dict = event.to_dict()
        self.assertIsInstance(event_dict, dict)
        self.assertEqual(event_dict["event_id"], "roundtrip-id")
        self.assertEqual(event_dict["event_type"], "DECISION")
        self.assertEqual(event_dict["files_involved"], ["plan.md"])
        self.assertIsInstance(event_dict["metadata"], dict)

        # Reconstruct from dict
        reconstructed = Event.from_dict(event_dict)
        self.assertEqual(reconstructed.event_id, event.event_id)
        self.assertEqual(reconstructed.event_type, event.event_type)
        self.assertEqual(reconstructed.timestamp, event.timestamp)
        self.assertEqual(reconstructed.project_id, event.project_id)
        self.assertEqual(reconstructed.agent_id, event.agent_id)
        self.assertEqual(reconstructed.session_id, event.session_id)
        self.assertEqual(reconstructed.actor, event.actor)
        self.assertEqual(reconstructed.input, event.input)
        self.assertEqual(reconstructed.output, event.output)
        self.assertEqual(reconstructed.files_involved, event.files_involved)
        self.assertEqual(reconstructed.result, event.result)
        self.assertEqual(dict(reconstructed.metadata), dict(event.metadata))

    def test_serialization_to_json_and_from_json(self) -> None:
        """Test JSON string serialization and deserialization."""
        event = Event.create(
            event_type=EventType.CODE_CHANGE,
            project_id="proj-code",
            session_id="sess-code",
            actor="agent:antigravity",
            files_involved=["ai_brain/events.py"],
            metadata={"lines_added": 42},
        )

        json_str = event.to_json()
        self.assertIsInstance(json_str, str)

        parsed = Event.from_json(json_str)
        self.assertEqual(parsed.event_id, event.event_id)
        self.assertEqual(parsed.event_type, event.event_type)
        self.assertEqual(parsed.timestamp, event.timestamp)
        self.assertEqual(parsed.files_involved, ("ai_brain/events.py",))
        self.assertEqual(parsed.metadata["lines_added"], 42)

    def test_deserialization_invalid_json(self) -> None:
        """Test deserialization with invalid JSON string raises EventValidationError."""
        with self.assertRaises(EventValidationError):
            Event.from_json("not valid json {[[")

        with self.assertRaises(EventValidationError):
            Event.from_json(12345)  # type: ignore


class TestContainerImmutabilityAndFreezing(unittest.TestCase):
    """Tests covering container freezing, unfreezing, immutability, and multi-module compatibility."""

    def test_nested_dictionaries_containing_lists_and_tuples(self) -> None:
        source = {
            "top_key": "top_val",
            "nested_dict": {"inner_key": [1, 2, 3]},
            "nested_list": [{"a": 1}, (4, 5, [6, 7])],
        }
        frozen = _freeze_container(source)
        self.assertIsInstance(frozen, MappingProxyType)
        self.assertIsInstance(frozen["nested_dict"], MappingProxyType)
        self.assertIsInstance(frozen["nested_dict"]["inner_key"], tuple)
        self.assertEqual(frozen["nested_dict"]["inner_key"], (1, 2, 3))
        self.assertIsInstance(frozen["nested_list"], tuple)
        self.assertIsInstance(frozen["nested_list"][0], MappingProxyType)
        self.assertIsInstance(frozen["nested_list"][1], tuple)
        self.assertIsInstance(frozen["nested_list"][1][2], tuple)

    def test_sets_containing_hashable_scalar_elements(self) -> None:
        source = {1, 2, "alpha", (10, 20)}
        frozen = _freeze_container(source)
        self.assertIsInstance(frozen, frozenset)
        self.assertEqual(frozen, frozenset({1, 2, "alpha", (10, 20)}))

    def test_frozensets_preservation(self) -> None:
        source = frozenset(["x", "y", "z"])
        frozen = _freeze_container(source)
        self.assertIsInstance(frozen, frozenset)
        self.assertEqual(frozen, source)

    def test_mixed_nested_structures(self) -> None:
        source = {
            "tags": {"tag1", "tag2"},
            "counts": [10, 20],
            "frozen_meta": frozenset([1, 2]),
            "sub": {"enabled": True, "ratio": 3.14, "empty_list": []},
        }
        frozen = _freeze_container(source)
        self.assertIsInstance(frozen["tags"], frozenset)
        self.assertIsInstance(frozen["counts"], tuple)
        self.assertIsInstance(frozen["frozen_meta"], frozenset)
        self.assertIsInstance(frozen["sub"], MappingProxyType)
        self.assertIsInstance(frozen["sub"]["empty_list"], tuple)
        self.assertEqual(frozen["sub"]["enabled"], True)
        self.assertEqual(frozen["sub"]["ratio"], 3.14)

    def test_mutation_attempts_fail_at_all_immutable_levels(self) -> None:
        source = {
            "config": {"setting": "active"},
            "items": ["a", "b"],
            "unique": {1, 2},
        }
        frozen = _freeze_container(source)
        # Top-level dict mutation fails
        with self.assertRaises(TypeError):
            frozen["config"] = {}  # type: ignore
        with self.assertRaises(TypeError):
            frozen["new_key"] = 123  # type: ignore

        # Nested dict mutation fails
        with self.assertRaises(TypeError):
            frozen["config"]["setting"] = "disabled"  # type: ignore

        # Tuple mutation fails
        with self.assertRaises(TypeError):
            frozen["items"][0] = "c"  # type: ignore
        self.assertFalse(hasattr(frozen["items"], "append"))

        # Frozenset mutation fails
        self.assertFalse(hasattr(frozen["unique"], "add"))
        self.assertFalse(hasattr(frozen["unique"], "remove"))

    def test_freeze_unfreeze_round_trips(self) -> None:
        source = {
            "name": "project",
            "tags": ["a", "b"],
            "metrics": [1, 2, 3],
            "nested": {"threshold": 0.95, "flags": [True, False]},
        }
        frozen = _freeze_container(source)
        unfrozen = _unfreeze_container(frozen)

        self.assertIsInstance(unfrozen, dict)
        self.assertIsInstance(unfrozen["tags"], list)
        self.assertIsInstance(unfrozen["metrics"], list)
        self.assertIsInstance(unfrozen["nested"], dict)
        self.assertIsInstance(unfrozen["nested"]["flags"], list)
        self.assertEqual(unfrozen, source)

        # Mutate unfrozen copy to prove independence
        unfrozen["nested"]["flags"].append(True)
        self.assertEqual(len(unfrozen["nested"]["flags"]), 3)
        self.assertEqual(len(frozen["nested"]["flags"]), 2)

    def test_compatibility_across_all_eight_model_modules(self) -> None:
        modules = [
            ai_brain.common,
            ai_brain.common.freeze,
            ai_brain.context.models,
            ai_brain.events.models,
            ai_brain.graph.models,
            ai_brain.loop.models,
            ai_brain.memories.conflicts.models,
            ai_brain.memories.models,
            ai_brain.retrieval.models,
            ai_brain.vectors.models,
        ]

        test_data = {"key": [1, 2, {"k": 3}], "items": ["x", "y"]}

        for mod in modules:
            self.assertTrue(hasattr(mod, "_freeze_container"), f"{mod.__name__} missing _freeze_container")
            self.assertTrue(hasattr(mod, "_unfreeze_container"), f"{mod.__name__} missing _unfreeze_container")

            # Check identity with canonical implementation
            self.assertIs(mod._freeze_container, _freeze_container, f"{mod.__name__}._freeze_container not canonical")
            self.assertIs(mod._unfreeze_container, _unfreeze_container, f"{mod.__name__}._unfreeze_container not canonical")

            # Test execution from module re-export
            f = mod._freeze_container(test_data)
            self.assertIsInstance(f, MappingProxyType)
            u = mod._unfreeze_container(f)
            self.assertEqual(u, test_data)


if __name__ == "__main__":
    unittest.main()

