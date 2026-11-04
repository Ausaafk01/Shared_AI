"""
Automated unit tests for Session Management, Persistence, and Agent Switching in AI Brain.
"""

from datetime import datetime, timezone
import os
import tempfile
import unittest

from ai_brain.agents import (
    AgentExecutionRequest,
    AgentSwitcher,
    CodingAgent,
    ResearchAgent,
    ReviewAgent,
    SessionManager,
    SessionRecord,
    SessionStatus,
    SQLiteSessionStore,
)
from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import SQLiteEventStore
from ai_brain.events.types import EventType
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import BrainLoop, SQLiteInteractionStore
from ai_brain.memories import (
    ConflictDetector,
    ConflictManager,
    MemoryExtractor,
    MemoryStatus,
    MemoryUpdater,
)
from ai_brain.retrieval import RetrievalEngine
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
)


class TestSessionManagementAndSwitching(unittest.TestCase):
    """
    Tests covering:
    - Session creation, retrieval, resume, close.
    - Persistent session records surviving process restarts.
    - Agent switching within a session and across sessions.
    - Audit event generation (SESSION_START, AGENT_SWITCH, SESSION_END).
    - Cross-agent memory retrieval and dispute handling.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_db_path = os.path.join(self.temp_dir.name, "sessions.db")
        self.event_store = SQLiteEventStore(":memory:")
        self.session_store = SQLiteSessionStore(self.session_db_path)
        self.session_manager = SessionManager(
            store=self.session_store,
            event_store=self.event_store,
        )

        # Full Brain Loop setup
        self.vector_provider = MockEmbeddingProvider()
        self.vector_store = SQLiteVectorStore(":memory:")
        self.vector_index = VectorIndex(storage=self.vector_store, provider=self.vector_provider)
        self.vector_manager = VectorMemoryManager(storage=self.vector_store, provider=self.vector_provider)

        self.graph = KnowledgeGraph()
        self.graph_integrator = GraphIntegrator(self.graph)

        self.retrieval_engine = RetrievalEngine(
            vector_index=self.vector_index,
            graph=self.graph,
            allow_partial_results=True,
        )
        self.context_builder = ContextBuilder(
            tokenizer=DeterministicTokenizer(),
            default_max_tokens=600,
        )
        self.brain_loop = BrainLoop(
            event_store=self.event_store,
            retrieval_engine=self.retrieval_engine,
            context_builder=self.context_builder,
            memory_extractor=MemoryExtractor(),
            memory_updater=MemoryUpdater(),
            conflict_detector=ConflictDetector(),
            conflict_manager=ConflictManager(),
            graph=self.graph,
            graph_integrator=self.graph_integrator,
            vector_manager=self.vector_manager,
        )

        self.research_agent = ResearchAgent(agent_id="researcher-1")
        self.coding_agent = CodingAgent(agent_id="coder-1")
        self.review_agent = ReviewAgent(agent_id="reviewer-1")

        self.agent_switcher = AgentSwitcher(
            brain_loop=self.brain_loop,
            session_manager=self.session_manager,
            event_store=self.event_store,
        )

    def tearDown(self) -> None:
        try:
            self.session_manager.close()
        except Exception:
            pass
        try:
            self.brain_loop.close()
        except Exception:
            pass
        try:
            self.event_store.close()
        except Exception:
            pass
        try:
            self.vector_store.close()
        except Exception:
            pass
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_session_creation_and_audit_event(self) -> None:
        """Creating a session records session record and emits SESSION_START event."""
        session = self.session_manager.create_session(
            project_id="proj-web",
            session_id="sess-start-01",
            initial_agent_id="researcher-1",
            title="Database Architecture Session",
        )
        self.assertEqual(session.session_id, "sess-start-01")
        self.assertEqual(session.current_agent_id, "researcher-1")
        self.assertIn("researcher-1", session.participating_agent_ids)
        self.assertEqual(session.status, SessionStatus.ACTIVE)

        # Verify SESSION_START event in event store
        events = self.event_store.get_events_by_session("sess-start-01")
        self.assertTrue(any(e.event_type == EventType.SESSION_START for e in events))

    def test_session_persistence_survives_process_restart(self) -> None:
        """Session records persist in SQLite across store reloads."""
        session = self.session_manager.create_session(
            project_id="proj-p",
            session_id="sess-persist-100",
            initial_agent_id="coder-1",
            title="Persistent Coding Session",
        )
        self.session_manager.switch_agent("sess-persist-100", "reviewer-1")

        # Close and reopen store
        self.session_manager.close()
        reopened_store = SQLiteSessionStore(self.session_db_path)
        reopened_manager = SessionManager(store=reopened_store)

        try:
            loaded = reopened_manager.get_session("sess-persist-100")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.project_id, "proj-p")
            self.assertEqual(loaded.current_agent_id, "reviewer-1")
            self.assertEqual(set(loaded.participating_agent_ids), {"coder-1", "reviewer-1"})
            self.assertEqual(loaded.status, SessionStatus.ACTIVE)
        finally:
            reopened_manager.close()

    def test_agent_switching_within_same_session(self) -> None:
        """Switching agents updates session participants and emits AGENT_SWITCH event."""
        session = self.session_manager.create_session(
            project_id="proj-switch",
            session_id="sess-switch-01",
            initial_agent_id="researcher-1",
        )
        self.assertEqual(session.current_agent_id, "researcher-1")

        # Switch to Coding Agent
        ctx = self.agent_switcher.switch(
            from_agent_id="researcher-1",
            to_agent=self.coding_agent,
            project_id="proj-switch",
            session_id="sess-switch-01",
            reason="Implement gathered research findings",
        )
        self.assertIsNotNone(ctx)

        # Verify session state
        updated_session = self.session_manager.get_session("sess-switch-01")
        self.assertEqual(updated_session.current_agent_id, "coder-1")
        self.assertIn("coder-1", updated_session.participating_agent_ids)
        self.assertIn("researcher-1", updated_session.participating_agent_ids)

        # Verify AGENT_SWITCH event was recorded
        events = self.event_store.get_events_by_session("sess-switch-01")
        switch_events = [e for e in events if e.event_type == EventType.AGENT_SWITCH]
        self.assertEqual(len(switch_events), 1)
        self.assertEqual(switch_events[0].metadata["from_agent"], "researcher-1")
        self.assertEqual(switch_events[0].metadata["to_agent"], "coder-1")

    def test_agent_switching_across_separate_sessions(self) -> None:
        """Switching across sessions preserves project context while keeping session IDs distinct."""
        # Session 1: Research Agent establishes fact
        self.session_manager.create_session(
            project_id="proj-cross",
            session_id="sess-1-res",
            initial_agent_id="researcher-1",
        )
        self.research_agent.execute(
            AgentExecutionRequest(
                project_id="proj-cross",
                session_id="sess-1-res",
                content="Research database for persistence.",
            ),
            self.brain_loop,
        )

        # Session 2: New session created for Coding Agent
        self.session_manager.create_session(
            project_id="proj-cross",
            session_id="sess-2-code",
            initial_agent_id="coder-1",
        )

        # Switch into session 2
        ctx = self.agent_switcher.switch(
            from_agent_id="researcher-1",
            to_agent=self.coding_agent,
            project_id="proj-cross",
            session_id="sess-2-code",
            handover_query="database architecture",
            reason="Begin coding in dedicated development session",
        )
        self.assertIsNotNone(ctx)
        # Shared knowledge from session 1 is available to Coding Agent in session 2!
        self.assertGreater(len(ctx.context_package.items), 0)
        self.assertTrue(any("PostgreSQL" in item.content for item in ctx.context_package.items))

    def test_session_close_records_session_end_event(self) -> None:
        """Closing a session sets COMPLETED status and emits SESSION_END event."""
        self.session_manager.create_session(
            project_id="proj-end",
            session_id="sess-to-close",
            initial_agent_id="researcher-1",
        )
        closed = self.session_manager.close_session("sess-to-close")
        self.assertEqual(closed.status, SessionStatus.COMPLETED)

        events = self.event_store.get_events_by_session("sess-to-close")
        self.assertTrue(any(e.event_type == EventType.SESSION_END for e in events))

    def test_cross_agent_disputed_memory_handling(self) -> None:
        """Contradictory assertions across agents are surfaced with dispute warnings to subsequent agents."""
        # 1. Researcher establishes PostgreSQL
        self.research_agent.execute(
            AgentExecutionRequest(
                project_id="proj-dispute",
                session_id="sess-1",
                content="The project uses PostgreSQL as its database.",
            ),
            self.brain_loop,
        )

        # 2. Another agent contradicts with MySQL
        contradictor = CodingAgent(agent_id="coder-rogue")
        contradictor.execute(
            AgentExecutionRequest(
                project_id="proj-dispute",
                session_id="sess-2",
                content="The project uses MySQL as its database.",
            ),
            self.brain_loop,
        )

        # 3. Reviewer enters project: Brain context prepares dispute warning
        ctx = self.review_agent.prepare_context(
            AgentExecutionRequest(
                project_id="proj-dispute",
                session_id="sess-3",
                content="Evaluate database architecture status.",
                query="database",
            ),
            self.brain_loop,
        )

        # Disputed knowledge is identified and rendered with warning in markdown context
        self.assertTrue(ctx.context_package.has_disputes)
        md = ctx.context_package.to_markdown()
        self.assertIn("DISPUTED", md)


if __name__ == "__main__":
    unittest.main()

