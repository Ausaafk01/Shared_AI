"""
Demonstration of AI Brain Event System.
Shows multi-agent event recording, persistence, chronological replay, and filtering.
"""

import sys
from pathlib import Path
import tempfile

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_brain.events import (
    Event,
    EventType,
    SQLiteEventStore,
    create_event,
    register_event_type,
)


def run_demo() -> None:
    print("=== AI Brain: Event System Demonstration ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "demo_brain.db"
        print(f"Creating persistent store at: {db_path}\n")

        with SQLiteEventStore(db_path) as store:
            # 1. User initiates a session
            start_event = create_event(
                event_type=EventType.SESSION_START,
                project_id="ai-brain-demo",
                session_id="session-42",
                actor="user",
                input={"prompt": "Design an extensible event system in Python"},
            )
            store.store_event(start_event)
            print(f"[STORED] {start_event.event_type} (ID: {start_event.event_id})")

            # 2. Agent Claude generates a plan
            plan_event = create_event(
                event_type=EventType.AGENT_MESSAGE,
                project_id="ai-brain-demo",
                agent_id="claude-3-opus",
                session_id="session-42",
                actor="agent:claude-3-opus",
                output={"plan": "1. Create immutable Event model. 2. Implement SQLite store."},
            )
            store.store_event(plan_event)
            print(f"[STORED] {plan_event.event_type} from {plan_event.actor}")

            # 3. Agent Switch to Codex for implementation
            switch_event = create_event(
                event_type=EventType.AGENT_SWITCH,
                project_id="ai-brain-demo",
                agent_id="codex",
                session_id="session-42",
                actor="system",
                metadata={"from_agent": "claude-3-opus", "to_agent": "codex"},
            )   
            store.store_event(switch_event)
            print(f"[STORED] {switch_event.event_type}: Claude -> Codex")

            # 4. Codex makes a code change
            code_event = create_event(
                event_type=EventType.CODE_CHANGE,
                project_id="ai-brain-demo",
                agent_id="codex",
                session_id="session-42",
                actor="agent:codex",
                tool_used="edit_file",
                files_involved=["ai_brain/events/models.py"],
                result={"status": "success", "lines_written": 120},
            )
            store.store_event(code_event)
            print(f"[STORED] {code_event.event_type} on {code_event.files_involved}")

            # 5. Extensible custom event type
            register_event_type("BENCHMARK_RECORD")
            benchmark_event = create_event(
                event_type="BENCHMARK_RECORD",
                project_id="ai-brain-demo",
                agent_id="codex",
                session_id="session-42",
                actor="agent:codex",
                output={"benchmark_ops_sec": 50000},
            )
            store.store_event(benchmark_event)
            print(f"[STORED] Custom Extensible Type: {benchmark_event.event_type}")

            # 6. Session End
            end_event = create_event(
                event_type=EventType.SESSION_END,
                project_id="ai-brain-demo",
                session_id="session-42",
                actor="system",
                metadata={"duration_sec": 12.5},
            )
            store.store_event(end_event)
            print(f"[STORED] {end_event.event_type}\n")

            # 7. Querying and Chronological Replay
            print("--- Chronological Replay ---")
            events = store.get_all_events(ascending=True)
            for i, evt in enumerate(events, 1):
                agent_str = f" [Agent: {evt.agent_id}]" if evt.agent_id else ""
                print(f"{i}. [{evt.timestamp.strftime('%H:%M:%S.%f')}] {evt.event_type}{agent_str} by {evt.actor}")

            # 8. Filtering by Agent
            print("\n--- Filtering by Agent 'codex' ---")
            codex_events = store.get_events_by_agent("codex")
            for evt in codex_events:
                print(f"- {evt.event_type} ({evt.event_id})")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
