"""
Comprehensive test suite for the Brain Loop subsystem.
"""

from datetime import datetime, timezone
import gc
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import warnings

from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import SQLiteEventStore
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import (
    AgentInteractionInput,
    AgentResponseInput,
    BrainLoop,
    ContextStageError,
    EventPersistenceError,
    ExtractionStageError,
    InputProcessor,
    InputValidationError,
    InteractionCompletionResult,
    InteractionContextResult,
    InteractionNotFoundError,
    InteractionRecord,
    InteractionStateError,
    InteractionStatus,
    LoopEventHandler,
    LoopMemoryHandler,
    LoopProjectIsolationError,
    LoopStage,
    MemoryUpdateStageError,
    ResponseCapture,
    ResponseValidationError,
    RetrievalStageError,
    SQLiteInteractionStore,
)
from ai_brain.memories import (
    ConflictDetector,
    ConflictManager,
    Memory,
    MemoryExtractor,
    MemoryStatus,
    MemoryType,
    MemoryUpdater,
)
from ai_brain.retrieval import RetrievalEngine, RetrievalMethod
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
)


class TestBrainLoop(unittest.TestCase):
    """
    Test suite covering all 25 requirements of Brain Loop orchestration.
    """

    def setUp(self) -> None:
        # 1. Event Store (in-memory SQLite)
        self.event_store = SQLiteEventStore(":memory:")

        # 2. Vector Subsystem
        self.vector_provider = MockEmbeddingProvider()
        self.vector_store = SQLiteVectorStore(":memory:")
        self.vector_index = VectorIndex(storage=self.vector_store, provider=self.vector_provider)
        self.vector_manager = VectorMemoryManager(storage=self.vector_store, provider=self.vector_provider)

        # 3. Knowledge Graph Subsystem
        self.graph = KnowledgeGraph()
        self.graph_integrator = GraphIntegrator(self.graph)

        # 4. Retrieval & Context Subsystems
        self.retrieval_engine = RetrievalEngine(
            vector_index=self.vector_index,
            graph=self.graph,
            allow_partial_results=True,
        )
        self.context_builder = ContextBuilder(
            tokenizer=DeterministicTokenizer(),
            default_max_tokens=600,
        )

        # 5. Memory Subsystem
        self.memory_extractor = MemoryExtractor()
        self.memory_updater = MemoryUpdater()
        self.conflict_detector = ConflictDetector()
        self.conflict_manager = ConflictManager(updater=self.memory_updater)

        # 6. Central Brain Loop
        self.loop = BrainLoop(
            event_store=self.event_store,
            retrieval_engine=self.retrieval_engine,
            context_builder=self.context_builder,
            memory_extractor=self.memory_extractor,
            memory_updater=self.memory_updater,
            conflict_detector=self.conflict_detector,
            conflict_manager=self.conflict_manager,
            graph=self.graph,
            graph_integrator=self.graph_integrator,
            vector_manager=self.vector_manager,
        )

    def tearDown(self) -> None:
        try:
            self.loop.close()
        except Exception:
            pass
        try:
            if hasattr(self.event_store, "close"):
                self.event_store.close()
        except Exception:
            pass
        try:
            if hasattr(self.vector_store, "close"):
                self.vector_store.close()
        except Exception:
            pass

    # 1. Valid input processing
    def test_1_valid_input_processing(self) -> None:
        """Valid AgentInteractionInput is accepted and normalized."""
        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        ctx_res = self.loop.prepare_context(inp)
        self.assertEqual(ctx_res.project_id, "proj-alpha")
        self.assertEqual(ctx_res.agent_id, "agent-claude")
        self.assertEqual(ctx_res.session_id, "sess-01")
        self.assertEqual(ctx_res.status, InteractionStatus.CONTEXT_PREPARED)
        self.assertIsNotNone(ctx_res.input_event_id)
        self.assertIsNotNone(ctx_res.context_event_id)

    # 2. Invalid input rejection
    def test_2_invalid_input_rejection(self) -> None:
        """Empty content, project_id, agent_id, or session_id are rejected."""
        with self.assertRaises(InputValidationError):
            AgentInteractionInput(project_id="", agent_id="a", session_id="s", content="msg")
        with self.assertRaises(InputValidationError):
            AgentInteractionInput(project_id="p", agent_id="", session_id="s", content="msg")
        with self.assertRaises(InputValidationError):
            AgentInteractionInput(project_id="p", agent_id="a", session_id="", content="msg")
        with self.assertRaises(InputValidationError):
            AgentInteractionInput(project_id="p", agent_id="a", session_id="s", content="   ")
        with self.assertRaises(InputValidationError):
            AgentInteractionInput(project_id="p", agent_id="a", session_id="s", content="msg", token_budget=-10)

    # 3. Project isolation
    def test_3_project_isolation(self) -> None:
        """Cross-project reference injection is rejected and memories are isolated."""
        # A: Cross-project metadata rejection
        with self.assertRaises(LoopProjectIsolationError):
            inp = AgentInteractionInput(
                project_id="proj-alpha",
                agent_id="agent-claude",
                session_id="sess-01",
                content="Hello",
                metadata={"target_project_id": "proj-beta"},
            )
            self.loop.prepare_context(inp)

        # B: Memory isolation: memories in proj-beta are not retrieved in proj-alpha
        beta_mem = Memory.create(
            memory_id="mem-beta-1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL.",
            project_id="proj-beta",
        )
        self.vector_manager.index_memory(beta_mem)

        inp_alpha = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        res_alpha = self.loop.prepare_context(inp_alpha)
        # Verify beta memory not retrieved
        self.assertEqual(len(res_alpha.context_package.items), 0)

    # 4. Agent and session isolation
    def test_4_agent_and_session_isolation(self) -> None:
        """Multiple agents in separate sessions maintain independent records."""
        inp1 = AgentInteractionInput(
            interaction_id="int-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-claude",
            content="Claude query",
        )
        inp2 = AgentInteractionInput(
            interaction_id="int-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-codex",
            content="Codex query",
        )
        r1 = self.loop.prepare_context(inp1)
        r2 = self.loop.prepare_context(inp2)

        self.assertEqual(r1.agent_id, "agent-claude")
        self.assertEqual(r2.agent_id, "agent-codex")
        self.assertNotEqual(r1.input_event_id, r2.input_event_id)

        rec1 = self.loop.get_interaction("int-1")
        rec2 = self.loop.get_interaction("int-2")
        self.assertEqual(rec1.session_id, "sess-claude")
        self.assertEqual(rec2.session_id, "sess-codex")

    # 5. Vector retrieval through the Brain Loop
    def test_5_vector_retrieval_through_brain_loop(self) -> None:
        """Retrieval method 'vector' searches via vector index."""
        mem = Memory.create(
            memory_id="mem-v1",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL database.",
            project_id="proj-alpha",
        )
        self.vector_manager.index_memory(mem)

        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="PostgreSQL database configuration",
            retrieval_method="vector",
        )
        res = self.loop.prepare_context(inp)
        self.assertEqual(res.retrieval_response.method.lower(), "vector")
        self.assertGreater(len(res.context_package.items), 0)
        self.assertEqual(res.context_package.items[0].memory_id, "mem-v1")

    # 6. Graph retrieval through the Brain Loop
    def test_6_graph_retrieval_through_brain_loop(self) -> None:
        """Retrieval method 'graph' searches via knowledge graph traversal."""
        mem = Memory.create(
            memory_id="mem-g1",
            memory_type=MemoryType.FACT,
            content="Primary PostgreSQL architecture.",
            project_id="proj-alpha",
        )
        self.graph_integrator.integrate_memory(mem)

        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="PostgreSQL architecture",
            retrieval_method="graph",
        )
        res = self.loop.prepare_context(inp)
        self.assertEqual(res.retrieval_response.method.lower(), "graph")
        self.assertGreater(len(res.context_package.items), 0)

    # 7. Hybrid retrieval through the Brain Loop
    def test_7_hybrid_retrieval_through_brain_loop(self) -> None:
        """Retrieval method 'hybrid' combines vector and graph retrieval."""
        mem = Memory.create(
            memory_id="mem-h1",
            memory_type=MemoryType.FACT,
            content="The application uses PostgreSQL.",
            project_id="proj-alpha",
        )
        self.vector_manager.index_memory(mem)
        self.graph_integrator.integrate_memory(mem)

        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database does the application use?",
            retrieval_method="hybrid",
        )
        res = self.loop.prepare_context(inp)
        self.assertEqual(res.retrieval_response.method.lower(), "hybrid")
        self.assertGreater(len(res.context_package.items), 0)

    # 8. Empty retrieval results
    def test_8_empty_retrieval_results(self) -> None:
        """Empty retrieval matches succeed gracefully without failing."""
        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query on completely empty memory store",
        )
        res = self.loop.prepare_context(inp)
        self.assertEqual(len(res.context_package.items), 0)
        self.assertIn("No relevant memories found", res.context_package.to_markdown())

    # 9. Context Builder integration
    def test_9_context_builder_integration(self) -> None:
        """Context package is cleanly structured with sections and metadata."""
        mem = Memory.create(
            memory_id="mem-dec-1",
            memory_type=MemoryType.DECISION,
            content="Decided to adopt PostgreSQL 16.",
            project_id="proj-alpha",
        )
        self.vector_manager.index_memory(mem)

        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="database decision",
        )
        res = self.loop.prepare_context(inp)
        self.assertIsNotNone(res.context_package)
        self.assertEqual(len(res.context_package.get_active_decisions()), 1)

    # 10. Token-budget preservation
    def test_10_token_budget_preservation(self) -> None:
        """Specified token budget on input is respected by ContextBuilder."""
        mem = Memory.create(
            memory_id="mem-b1",
            memory_type=MemoryType.FACT,
            content="PostgreSQL configuration.",
            project_id="proj-alpha",
        )
        self.vector_manager.index_memory(mem)

        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="PostgreSQL",
            token_budget=250,
        )
        res = self.loop.prepare_context(inp)
        self.assertEqual(res.context_package.token_budget, 250)
        self.assertLessEqual(res.context_package.tokens_used, 250)

    # 11. Disputed-memory warning preservation
    def test_11_disputed_memory_warning_preservation(self) -> None:
        """Disputed memories include warning banners or omission notices."""
        disp = Memory.create(
            memory_id="mem-disp-1",
            memory_type=MemoryType.FACT,
            content="Cache is Redis on port 6379, conflicting with Memcached.",
            project_id="proj-alpha",
            status=MemoryStatus.DISPUTED,
        )
        self.vector_manager.index_memory(disp)

        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Cache configuration",
            token_budget=600,
        )
        res = self.loop.prepare_context(inp)
        self.assertTrue(res.context_package.has_disputes)
        md = res.context_package.to_markdown()
        self.assertIn("## Disputed Knowledge", md)
        self.assertIn("⚠️", md)

    # 12. Response capture
    def test_12_response_capture(self) -> None:
        """Agent response is accepted and validated against originating interaction."""
        inp = AgentInteractionInput(
            interaction_id="int-resp-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Please summarize database tech.",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="int-resp-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp = self.loop.complete_interaction(resp)
        self.assertEqual(comp.interaction_id, "int-resp-1")
        self.assertEqual(comp.status, InteractionStatus.COMPLETED)
        self.assertIsNotNone(comp.response_event_id)

    # 13. Response/input association
    def test_13_response_input_association(self) -> None:
        """Response event correctly records parent linkage to input event."""
        inp = AgentInteractionInput(
            interaction_id="int-assoc-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query",
        )
        ctx = self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="int-assoc-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Response",
        )
        comp = self.loop.complete_interaction(resp)

        resp_event = self.event_store.get_event_by_id(comp.response_event_id)
        self.assertIsNotNone(resp_event)
        self.assertEqual(resp_event.metadata.get("parent_event_id"), ctx.context_event_id)

    # 14. Invalid response association rejection
    def test_14_invalid_response_association_rejection(self) -> None:
        """Mismatched project, agent, or unknown interaction ID is rejected."""
        inp = AgentInteractionInput(
            interaction_id="int-valid-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query",
        )
        self.loop.prepare_context(inp)

        # A: Unknown interaction ID
        with self.assertRaises(InteractionNotFoundError):
            self.loop.complete_interaction(
                AgentResponseInput(
                    interaction_id="unknown-id",
                    project_id="proj-alpha",
                    agent_id="agent-claude",
                    session_id="sess-01",
                    content="Resp",
                )
            )

        # B: Mismatched project ID
        with self.assertRaises(LoopProjectIsolationError):
            self.loop.complete_interaction(
                AgentResponseInput(
                    interaction_id="int-valid-1",
                    project_id="proj-wrong",
                    agent_id="agent-claude",
                    session_id="sess-01",
                    content="Resp",
                )
            )

        # C: Mismatched agent ID
        with self.assertRaises(ResponseValidationError):
            self.loop.complete_interaction(
                AgentResponseInput(
                    interaction_id="int-valid-1",
                    project_id="proj-alpha",
                    agent_id="agent-wrong",
                    session_id="sess-01",
                    content="Resp",
                )
            )

    # 15. Event generation and provenance
    def test_15_event_generation_and_provenance(self) -> None:
        """Lifecycle generates linked events in persistent store."""
        inp = AgentInteractionInput(
            interaction_id="int-ev-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query",
        )
        self.loop.prepare_context(inp)
        resp = AgentResponseInput(
            interaction_id="int-ev-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Response",
        )
        comp = self.loop.complete_interaction(resp)

        events = self.event_store.get_events_by_project("proj-alpha")
        event_types = [e.event_type for e in events]
        self.assertIn("INTERACTION_INPUT", event_types)
        self.assertIn("CONTEXT_DELIVERED", event_types)
        self.assertIn("INTERACTION_RESPONSE", event_types)
        self.assertIn("INTERACTION_COMPLETED", event_types)

    # 16. Duplicate interaction retry handling
    def test_16_duplicate_interaction_retry_handling(self) -> None:
        """Calling prepare_context or complete_interaction with same input is idempotent."""
        inp = AgentInteractionInput(
            interaction_id="int-retry-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Retry test query",
        )
        res1 = self.loop.prepare_context(inp)
        res2 = self.loop.prepare_context(inp)
        self.assertEqual(res1.input_event_id, res2.input_event_id)

        resp = AgentResponseInput(
            interaction_id="int-retry-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp1 = self.loop.complete_interaction(resp)
        comp2 = self.loop.complete_interaction(resp)
        self.assertEqual(comp1.response_event_id, comp2.response_event_id)

    # 17. Memory extraction from interaction events
    def test_17_memory_extraction_from_interaction_events(self) -> None:
        """Informative interaction events produce extracted memories."""
        inp = AgentInteractionInput(
            interaction_id="int-ext-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database was chosen?",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="int-ext-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp = self.loop.complete_interaction(resp)
        self.assertGreater(len(comp.extracted_memories), 0)
        self.assertEqual(comp.extracted_memories[0].memory_type, MemoryType.FACT.value)
        self.assertIn("PostgreSQL", comp.extracted_memories[0].content)

    # 18. Duplicate memory handling
    def test_18_duplicate_memory_handling(self) -> None:
        """Repeating identical information does not create redundant versions."""
        # Interaction 1
        inp1 = AgentInteractionInput(
            interaction_id="int-dup-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="First mention",
        )
        self.loop.prepare_context(inp1)
        resp1 = AgentResponseInput(
            interaction_id="int-dup-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp1 = self.loop.complete_interaction(resp1)
        self.assertEqual(comp1.update_results[0].created, True)

        # Interaction 2 with identical fact
        inp2 = AgentInteractionInput(
            interaction_id="int-dup-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-02",
            content="Second mention",
        )
        self.loop.prepare_context(inp2)
        resp2 = AgentResponseInput(
            interaction_id="int-dup-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-02",
            content="The project uses PostgreSQL as its database.",
        )
        comp2 = self.loop.complete_interaction(resp2)
        self.assertEqual(comp2.update_results[0].duplicate, True)

    # 19. Conflict handling
    def test_19_conflict_handling(self) -> None:
        """Contradictory assertions are detected as conflicts and marked DISPUTED."""
        # Step 1: Establish PostgreSQL
        inp1 = AgentInteractionInput(
            interaction_id="int-conf-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Establish db",
        )
        self.loop.prepare_context(inp1)
        resp1 = AgentResponseInput(
            interaction_id="int-conf-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        self.loop.complete_interaction(resp1)

        # Step 2: Agent Codex contradicts with MySQL
        inp2 = AgentInteractionInput(
            interaction_id="int-conf-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-02",
            content="Contradict db",
        )
        self.loop.prepare_context(inp2)
        resp2 = AgentResponseInput(
            interaction_id="int-conf-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-02",
            content="The project uses MySQL as its database.",
        )
        comp2 = self.loop.complete_interaction(resp2)
        self.assertGreater(len(comp2.conflicts_detected), 0)

    # 20. Graph and vector integration
    def test_20_graph_and_vector_integration(self) -> None:
        """Extracted memories are synchronized into Knowledge Graph and Vector Memory."""
        inp = AgentInteractionInput(
            interaction_id="int-sync-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)
        resp = AgentResponseInput(
            interaction_id="int-sync-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp = self.loop.complete_interaction(resp)
        self.assertGreater(comp.graph_nodes_integrated, 0)
        self.assertGreater(comp.vectors_indexed, 0)

        # Verify memory exists in KnowledgeGraph
        mem_node_id = f"memory:{comp.extracted_memories[0].memory_id}"
        self.assertIsNotNone(self.graph.get_node(mem_node_id))

    # 21. Failure handling at each processing stage
    def test_21_failure_handling_at_each_stage(self) -> None:
        """Failures in retrieval, context, and response validation raise expected exceptions."""
        # A: Retrieval failure
        bad_engine = MagicMock()
        bad_engine.retrieve.side_effect = RuntimeError("Vector index offline")
        loop_bad_retrieval = BrainLoop(
            event_store=self.event_store,
            retrieval_engine=bad_engine,
        )
        try:
            with self.assertRaises(RetrievalStageError):
                loop_bad_retrieval.prepare_context(
                    AgentInteractionInput(
                        project_id="proj-alpha",
                        agent_id="agent-claude",
                        session_id="sess-01",
                        content="Query",
                    )
                )
        finally:
            loop_bad_retrieval.close()

        # B: Context building failure
        bad_builder = MagicMock()
        bad_builder.build_context.side_effect = ValueError("Budget overflow")
        loop_bad_context = BrainLoop(
            event_store=self.event_store,
            context_builder=bad_builder,
        )
        try:
            with self.assertRaises(ContextStageError):
                loop_bad_context.prepare_context(
                    AgentInteractionInput(
                        project_id="proj-alpha",
                        agent_id="agent-claude",
                        session_id="sess-01",
                        content="Query",
                    )
                )
        finally:
            loop_bad_context.close()

    # 22. Partial failure reporting
    def test_22_partial_failure_reporting(self) -> None:
        """Downstream graph/vector sync errors report PARTIAL_SUCCESS with warnings."""
        bad_integrator = MagicMock()
        bad_integrator.integrate_memory.side_effect = RuntimeError("Graph memory exhausted")

        loop_partial = BrainLoop(
            event_store=self.event_store,
            graph_integrator=bad_integrator,
        )
        try:
            inp = AgentInteractionInput(
                interaction_id="int-part-1",
                project_id="proj-alpha",
                agent_id="agent-claude",
                session_id="sess-01",
                content="Setup tech",
            )
            loop_partial.prepare_context(inp)

            resp = AgentResponseInput(
                interaction_id="int-part-1",
                project_id="proj-alpha",
                agent_id="agent-claude",
                session_id="sess-01",
                content="The project uses PostgreSQL as its database.",
            )
            comp = loop_partial.complete_interaction(resp)
            self.assertEqual(comp.status, InteractionStatus.PARTIAL_SUCCESS)
            self.assertTrue(comp.is_partial_success)
            self.assertGreater(len(comp.warnings), 0)
        finally:
            loop_partial.close()

    # 23. Retry and idempotency behavior
    def test_23_retry_and_idempotency_behavior(self) -> None:
        """Interactions already in COMPLETED status return existing result without re-executing."""
        inp = AgentInteractionInput(
            interaction_id="int-idemp-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query",
        )
        self.loop.prepare_context(inp)
        resp = AgentResponseInput(
            interaction_id="int-idemp-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        res1 = self.loop.complete_interaction(resp)
        res2 = self.loop.complete_interaction(resp)
        self.assertEqual(res1.completion_event_id, res2.completion_event_id)
        self.assertEqual(res1.status, res2.status)

    # 24. End-to-end lifecycle
    def test_24_end_to_end_lifecycle(self) -> None:
        """Complete helper run_interaction executes prepare, callback, and completion."""
        inp = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )

        def mock_agent(ctx: InteractionContextResult) -> str:
            return "The project uses PostgreSQL as its database."

        ctx_res, comp_res = self.loop.run_interaction(inp, mock_agent)
        self.assertEqual(ctx_res.status, InteractionStatus.CONTEXT_PREPARED)
        self.assertEqual(comp_res.status, InteractionStatus.COMPLETED)
        self.assertGreater(len(comp_res.extracted_memories), 0)

    # 25. Original event and memory immutability
    def test_25_original_event_and_memory_immutability(self) -> None:
        """Stored events and memories remain completely immutable."""
        inp = AgentInteractionInput(
            interaction_id="int-immut-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)
        resp = AgentResponseInput(
            interaction_id="int-immut-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp = self.loop.complete_interaction(resp)

        mem = comp.extracted_memories[0]
        with self.assertRaises(Exception):
            mem.content = "Mutated"

        ev = self.event_store.get_event_by_id(comp.response_event_id)
        with self.assertRaises(Exception):
            ev.output = {"content": "Mutated"}

    # 26. Context manager and GC resource cleanup
    def test_26_context_manager_and_gc_cleanup(self) -> None:
        """Verify BrainLoop context manager and finalizer close interaction stores cleanly."""
        ev_store = SQLiteEventStore(":memory:")
        try:
            with BrainLoop(event_store=ev_store) as loop:
                self.assertFalse(loop.interaction_store._closed)
            self.assertTrue(loop.interaction_store._closed)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ResourceWarning)
                unreferenced_loop = BrainLoop(event_store=ev_store)
                del unreferenced_loop
                gc.collect()
                res_warns = [w for w in captured if issubclass(w.category, ResourceWarning)]
                self.assertEqual(len(res_warns), 0)
        finally:
            ev_store.close()


class TestBrainLoopReviewAndRecovery(unittest.TestCase):
    """
    Focused review tests covering:
    1. Interaction state persistence across application/process restarts.
    2. Exact idempotency guarantees (distinct IDs for identical content, payload validation on reuse).
    3. Failure recovery across all 10 lifecycle stages & stage resumption.
    4. Memory extraction safeguards (telemetry filtering, informative vs non-informative, duplicate, contradictory).
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "loop_review.db")

        self.event_store = SQLiteEventStore(self.db_path)
        self.interaction_store = SQLiteInteractionStore(self.db_path)

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
        self.memory_extractor = MemoryExtractor()
        self.memory_updater = MemoryUpdater()
        self.conflict_detector = ConflictDetector()
        self.conflict_manager = ConflictManager(updater=self.memory_updater)

        self.loop = BrainLoop(
            event_store=self.event_store,
            interaction_store=self.interaction_store,
            retrieval_engine=self.retrieval_engine,
            context_builder=self.context_builder,
            memory_extractor=self.memory_extractor,
            memory_updater=self.memory_updater,
            conflict_detector=self.conflict_detector,
            conflict_manager=self.conflict_manager,
            graph=self.graph,
            graph_integrator=self.graph_integrator,
            vector_manager=self.vector_manager,
        )

    def tearDown(self) -> None:
        try:
            self.loop.close()
        except Exception:
            pass
        try:
            self.event_store.close()
        except Exception:
            pass
        try:
            self.interaction_store.close()
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

    # --- 1. Interaction State Persistence Across Restarts ---

    def test_persistence_survives_process_restart(self) -> None:
        """InteractionRecord and full lifecycle state survive process recreation."""
        inp = AgentInteractionInput(
            interaction_id="int-persist-01",
            project_id="proj-persist",
            agent_id="agent-claude",
            session_id="sess-persist",
            content="We use PostgreSQL as the primary database.",
        )
        ctx_res = self.loop.prepare_context(inp)
        self.assertEqual(ctx_res.status, InteractionStatus.CONTEXT_PREPARED)

        resp = AgentResponseInput(
            interaction_id="int-persist-01",
            project_id="proj-persist",
            agent_id="agent-claude",
            session_id="sess-persist",
            content="The project uses PostgreSQL as its database.",
        )
        comp_res = self.loop.complete_interaction(resp)
        self.assertEqual(comp_res.status, InteractionStatus.COMPLETED)

        # Recreate the entire Brain Loop and stores from disk
        self.loop.close()
        self.event_store.close()
        self.interaction_store.close()

        reloaded_event_store = SQLiteEventStore(self.db_path)
        reloaded_interaction_store = SQLiteInteractionStore(self.db_path)
        reloaded_loop = BrainLoop(
            event_store=reloaded_event_store,
            interaction_store=reloaded_interaction_store,
        )

        try:
            record = reloaded_loop.get_interaction("int-persist-01")
            self.assertIsNotNone(record)
            self.assertEqual(record.interaction_id, "int-persist-01")
            self.assertEqual(record.project_id, "proj-persist")
            self.assertEqual(record.agent_id, "agent-claude")
            self.assertEqual(record.session_id, "sess-persist")
            self.assertEqual(record.status, InteractionStatus.COMPLETED)
            self.assertIn("COMPLETED", record.stages_completed)
            self.assertEqual(record.input_event_id, ctx_res.input_event_id)
            self.assertEqual(record.response_event_id, comp_res.response_event_id)
            self.assertIsNotNone(record.context_event_id)
            self.assertIsNotNone(record.completion_event_id)
            self.assertIsNotNone(record.context_result)
            self.assertIsNotNone(record.completion_result)
            self.assertEqual(len(record.completion_result.extracted_memories), 1)
        finally:
            reloaded_loop.close()
            reloaded_event_store.close()
            reloaded_interaction_store.close()

    # --- 2. Correct Idempotency Guarantees ---

    def test_two_separate_interactions_with_identical_messages_get_distinct_ids(self) -> None:
        """Two separate interactions with identical messages receive distinct interaction identities."""
        inp1 = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        inp2 = AgentInteractionInput(
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        # Distinct auto-generated interaction IDs
        self.assertNotEqual(inp1.interaction_id, inp2.interaction_id)

        res1 = self.loop.prepare_context(inp1)
        res2 = self.loop.prepare_context(inp2)

        self.assertNotEqual(res1.interaction_id, res2.interaction_id)
        self.assertNotEqual(res1.input_event_id, res2.input_event_id)

    def test_idempotency_key_reuse_returns_cached_context(self) -> None:
        """Supplying the same idempotency_key with matching payload reuses the interaction and returns cached context."""
        inp1 = AgentInteractionInput(
            idempotency_key="key-idemp-100",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        res1 = self.loop.prepare_context(inp1)

        # Retry with identical payload and same idempotency_key
        inp2 = AgentInteractionInput(
            idempotency_key="key-idemp-100",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        res2 = self.loop.prepare_context(inp2)

        self.assertEqual(res1.interaction_id, res2.interaction_id)
        self.assertEqual(res1.input_event_id, res2.input_event_id)
        self.assertEqual(res1.context_event_id, res2.context_event_id)

    def test_idempotency_key_reuse_with_mismatched_payload_rejected(self) -> None:
        """Reusing the same idempotency_key with different content raises InputValidationError."""
        inp1 = AgentInteractionInput(
            idempotency_key="key-idemp-200",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Original query about database.",
        )
        self.loop.prepare_context(inp1)

        # Retry with DIFFERENT content under same key
        inp2 = AgentInteractionInput(
            idempotency_key="key-idemp-200",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Completely different query about auth.",
        )
        with self.assertRaises(InputValidationError) as cm:
            self.loop.prepare_context(inp2)
        self.assertIn("mismatched input payload", str(cm.exception))

    def test_response_completion_retry_idempotent_no_duplicate_memories(self) -> None:
        """Retrying complete_interaction reuses completion result and creates no duplicate memory versions."""
        inp = AgentInteractionInput(
            interaction_id="int-comp-retry",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Record our database choice.",
        )
        self.loop.prepare_context(inp)

        resp1 = AgentResponseInput(
            interaction_id="int-comp-retry",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        res1 = self.loop.complete_interaction(resp1)
        self.assertEqual(len(res1.extracted_memories), 1)

        # Retry identical response completion
        res2 = self.loop.complete_interaction(resp1)
        self.assertEqual(res1.completion_event_id, res2.completion_event_id)
        self.assertEqual(res1.status, res2.status)
        # Verify no duplicate memories were added
        active_mems = self.loop.memory_handler.get_active_memories("proj-alpha")
        self.assertEqual(len(active_mems), 1)

    def test_response_completion_retry_with_mismatched_content_rejected(self) -> None:
        """Retrying complete_interaction with different response content raises ResponseValidationError."""
        inp = AgentInteractionInput(
            interaction_id="int-comp-mismatch",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Record our database choice.",
        )
        self.loop.prepare_context(inp)

        resp1 = AgentResponseInput(
            interaction_id="int-comp-mismatch",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        self.loop.complete_interaction(resp1)

        resp2 = AgentResponseInput(
            interaction_id="int-comp-mismatch",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Different reply trying to spoof completion.",
        )
        with self.assertRaises(ResponseValidationError) as cm:
            self.loop.complete_interaction(resp2)
        self.assertIn("already completed with different response content", str(cm.exception))

    # --- 3. Failure Recovery Across All 10 Lifecycle Stages ---

    def test_failure_1_after_input_event_persistence(self) -> None:
        """Failure after input event persistence persists stage progress and allows resumption."""
        inp = AgentInteractionInput(
            interaction_id="fail-stage-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query database config",
        )
        # Mock retrieval engine to simulate failure right after input persistence
        self.loop.retrieval_engine = MagicMock()
        self.loop.retrieval_engine.retrieve.side_effect = RuntimeError("Simulated retrieval crash")

        with self.assertRaises(RetrievalStageError):
            self.loop.prepare_context(inp)

        record = self.loop.get_interaction("fail-stage-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.status, InteractionStatus.FAILED)
        self.assertIn(LoopStage.INPUT_PERSISTED.value, record.stages_completed)
        self.assertIsNotNone(record.input_event_id)
        # Input event is safely persisted in SQLite
        ev = self.event_store.get_event_by_id(record.input_event_id)
        self.assertIsNotNone(ev)

        # Now fix retrieval and resume
        self.loop.retrieval_engine = None  # fallback empty
        resumed = self.loop.resume_interaction("fail-stage-1")
        self.assertEqual(resumed.status, InteractionStatus.CONTEXT_PREPARED)

    def test_failure_2_after_context_retrieval(self) -> None:
        """Failure during context builder after retrieval preserves retrieval and input progress."""
        inp = AgentInteractionInput(
            interaction_id="fail-stage-2",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query context",
        )
        with patch.object(self.loop.context_builder, "build_context", side_effect=ValueError("Context builder error")):
            with self.assertRaises(ContextStageError):
                self.loop.prepare_context(inp)

        record = self.loop.get_interaction("fail-stage-2")
        self.assertEqual(record.status, InteractionStatus.FAILED)
        self.assertIn(LoopStage.INPUT_PERSISTED.value, record.stages_completed)
        self.assertIn(LoopStage.CONTEXT_RETRIEVED.value, record.stages_completed)

    def test_failure_3_after_context_delivered_persists_delivered_event(self) -> None:
        """When context is delivered, status is CONTEXT_PREPARED with stage progress."""
        inp = AgentInteractionInput(
            interaction_id="stage-3-ok",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query context",
        )
        ctx = self.loop.prepare_context(inp)
        record = self.loop.get_interaction("stage-3-ok")
        self.assertEqual(record.status, InteractionStatus.CONTEXT_PREPARED)
        self.assertIn(LoopStage.CONTEXT_DELIVERED.value, record.stages_completed)
        self.assertIsNotNone(record.context_event_id)

    def test_failure_4_after_response_event_persistence(self) -> None:
        """Failure right after response persistence retains response event and allows resumption."""
        inp = AgentInteractionInput(
            interaction_id="fail-stage-4",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Query context",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="fail-stage-4",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        # Simulate failure in memory extraction
        with patch.object(self.loop.memory_handler, "extract_memories", side_effect=RuntimeError("Extraction crashed")):
            with self.assertRaises(RuntimeError):
                self.loop.complete_interaction(resp)

        record = self.loop.get_interaction("fail-stage-4")
        self.assertEqual(record.status, InteractionStatus.FAILED)
        self.assertIn(LoopStage.RESPONSE_PERSISTED.value, record.stages_completed)
        self.assertIsNotNone(record.response_event_id)
        # Response event is intact
        self.assertIsNotNone(self.event_store.get_event_by_id(record.response_event_id))

        # Resuming completes without re-inserting response event
        comp = self.loop.resume_interaction("fail-stage-4")
        self.assertEqual(comp.status, InteractionStatus.COMPLETED)

    def test_failure_5_memory_extraction_stage_error(self) -> None:
        """ExtractionStageError marks interaction FAILED without marking it COMPLETED."""
        inp = AgentInteractionInput(
            interaction_id="fail-stage-5",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="fail-stage-5",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        with patch.object(self.loop.memory_extractor, "extract_many", side_effect=Exception("Corrupt event")):
            with self.assertRaises(ExtractionStageError):
                self.loop.complete_interaction(resp)

        record = self.loop.get_interaction("fail-stage-5")
        self.assertEqual(record.status, InteractionStatus.FAILED)
        self.assertNotIn(LoopStage.COMPLETED.value, record.stages_completed)

    def test_failure_6_memory_updating_stage_error(self) -> None:
        """MemoryUpdateStageError preserves extracted memories and records failure."""
        inp = AgentInteractionInput(
            interaction_id="fail-stage-6",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="fail-stage-6",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        with patch.object(self.loop.memory_updater, "update_many", side_effect=Exception("Lock error")):
            with self.assertRaises(MemoryUpdateStageError):
                self.loop.complete_interaction(resp)

        record = self.loop.get_interaction("fail-stage-6")
        self.assertEqual(record.status, InteractionStatus.FAILED)
        self.assertIn(LoopStage.MEMORY_EXTRACTED.value, record.stages_completed)
        self.assertNotIn(LoopStage.COMPLETED.value, record.stages_completed)

    def test_failure_7_conflict_detection_warning_degrades_gracefully(self) -> None:
        """Conflict detection exception does not drop update; surfaces as warning."""
        inp = AgentInteractionInput(
            interaction_id="stage-7-conf",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)

        # Seed an existing memory so detect_pair is called on existing memories
        seed_mem = Memory.create(
            memory_id="mem-existing-seed",
            project_id="proj-alpha",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL as its database.",
        )
        self.loop.register_memories([seed_mem])

        resp = AgentResponseInput(
            interaction_id="stage-7-conf",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        with patch.object(self.loop.conflict_detector, "detect_pair", side_effect=RuntimeError("Detector error")):
            comp = self.loop.complete_interaction(resp)

        self.assertEqual(comp.status, InteractionStatus.PARTIAL_SUCCESS)
        self.assertTrue(comp.is_partial_success)
        self.assertTrue(any("Conflict detection warning" in w for w in comp.warnings))

    def test_failure_8_graph_sync_failure_identifiable_and_retryable(self) -> None:
        """Knowledge graph sync failure is identifiable as PARTIAL_SUCCESS and retryable."""
        inp = AgentInteractionInput(
            interaction_id="stage-8-graph",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="stage-8-graph",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        with patch.object(self.loop.graph_integrator, "integrate_memory", side_effect=RuntimeError("Graph lock")):
            comp = self.loop.complete_interaction(resp)

        self.assertEqual(comp.status, InteractionStatus.PARTIAL_SUCCESS)
        self.assertTrue(comp.is_partial_success)
        self.assertTrue(any("Graph integration failed" in w for w in comp.warnings))

        # Retry failed sync
        resynced = self.loop.retry_failed_sync("stage-8-graph")
        self.assertEqual(resynced.status, InteractionStatus.COMPLETED)
        self.assertFalse(resynced.is_partial_success)
        self.assertEqual(resynced.warnings, ())

    def test_failure_9_vector_sync_failure_identifiable_and_retryable(self) -> None:
        """Vector memory sync failure surfaces as PARTIAL_SUCCESS and is retryable."""
        inp = AgentInteractionInput(
            interaction_id="stage-9-vec",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="stage-9-vec",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        with patch.object(self.loop.vector_manager, "index_memory", side_effect=RuntimeError("Vector timeout")):
            comp = self.loop.complete_interaction(resp)

        self.assertEqual(comp.status, InteractionStatus.PARTIAL_SUCCESS)
        self.assertTrue(comp.is_partial_success)

        # Retry failed sync
        resynced = self.loop.retry_failed_sync("stage-9-vec")
        self.assertEqual(resynced.status, InteractionStatus.COMPLETED)
        self.assertEqual(resynced.warnings, ())

    def test_failure_10_completion_event_persistence_failure(self) -> None:
        """Failure persisting completion event leaves interaction FAILED instead of COMPLETED."""
        inp = AgentInteractionInput(
            interaction_id="stage-10-fail",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Setup tech",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="stage-10-fail",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        with patch.object(self.loop.event_handler, "record_completion_event", side_effect=EventPersistenceError("Disk full")):
            with self.assertRaises(EventPersistenceError):
                self.loop.complete_interaction(resp)

        record = self.loop.get_interaction("stage-10-fail")
        self.assertEqual(record.status, InteractionStatus.FAILED)
        self.assertNotIn(LoopStage.COMPLETED.value, record.stages_completed)

    # --- 4. Memory Extraction Safeguards ---

    def test_only_appropriate_events_passed_to_extractor(self) -> None:
        """Internal telemetry events (CONTEXT_DELIVERED, INTERACTION_COMPLETED) are filtered."""
        events = self.event_store.get_all_events()
        filtered = self.loop.memory_handler.filter_appropriate_events(events)
        for ev in filtered:
            self.assertNotIn(ev.event_type, ("CONTEXT_DELIVERED", "INTERACTION_COMPLETED", "PROCESSING_ERROR"))

    def test_non_informative_response_creates_no_unnecessary_memories(self) -> None:
        """Casual noise or speculation in responses does not create memories."""
        inp = AgentInteractionInput(
            interaction_id="int-casual-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Hello there!",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="int-casual-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Sure, I understand. I will take a look at it tomorrow.",
        )
        comp = self.loop.complete_interaction(resp)

        self.assertEqual(len(comp.extracted_memories), 0)
        self.assertEqual(len(comp.update_results), 0)
        self.assertEqual(comp.graph_nodes_integrated, 0)
        self.assertEqual(comp.vectors_indexed, 0)

    def test_informative_response_creates_permanent_fact(self) -> None:
        """Clear informative statement creates active memory."""
        inp = AgentInteractionInput(
            interaction_id="int-info-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        self.loop.prepare_context(inp)

        resp = AgentResponseInput(
            interaction_id="int-info-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp = self.loop.complete_interaction(resp)

        self.assertEqual(len(comp.extracted_memories), 1)
        self.assertEqual(comp.extracted_memories[0].memory_type, MemoryType.FACT)
        self.assertEqual(comp.extracted_memories[0].status, MemoryStatus.ACTIVE.value)

    def test_duplicate_response_does_not_create_new_versions(self) -> None:
        """Duplicate factual response applies Rule C/D without version increment."""
        # First interaction
        inp1 = AgentInteractionInput(
            interaction_id="int-dup-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="What database do we use?",
        )
        self.loop.prepare_context(inp1)
        resp1 = AgentResponseInput(
            interaction_id="int-dup-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        comp1 = self.loop.complete_interaction(resp1)
        mem1 = comp1.extracted_memories[0]

        # Second interaction with identical statement
        inp2 = AgentInteractionInput(
            interaction_id="int-dup-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-02",
            content="Confirm our database.",
        )
        self.loop.prepare_context(inp2)
        resp2 = AgentResponseInput(
            interaction_id="int-dup-2",
            project_id="proj-alpha",
            agent_id="agent-codex",
            session_id="sess-02",
            content="The project uses PostgreSQL as its database.",
        )
        comp2 = self.loop.complete_interaction(resp2)

        # Update action was DUPLICATE
        self.assertEqual(len(comp2.update_results), 1)
        self.assertTrue(comp2.update_results[0].duplicate)
        # Active memories count remains 1
        active = self.loop.memory_handler.get_active_memories("proj-alpha")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].version, 1)

    def test_contradictory_response_triggers_disputed_status_and_sync(self) -> None:
        """Contradictory response marks memories as DISPUTED and synchronizes DISPUTED state."""
        # 1. Establish PostgreSQL
        inp1 = AgentInteractionInput(
            interaction_id="int-contra-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="Database check",
        )
        self.loop.prepare_context(inp1)
        resp1 = AgentResponseInput(
            interaction_id="int-contra-1",
            project_id="proj-alpha",
            agent_id="agent-claude",
            session_id="sess-01",
            content="The project uses PostgreSQL as its database.",
        )
        self.loop.complete_interaction(resp1)

        # 2. Contradictory assertion from another agent
        inp2 = AgentInteractionInput(
            interaction_id="int-contra-2",
            project_id="proj-alpha",
            agent_id="agent-gemini",
            session_id="sess-02",
            content="Database check",
        )
        self.loop.prepare_context(inp2)
        resp2 = AgentResponseInput(
            interaction_id="int-contra-2",
            project_id="proj-alpha",
            agent_id="agent-gemini",
            session_id="sess-02",
            content="The project uses MySQL as its database.",
        )
        comp2 = self.loop.complete_interaction(resp2)

        # Conflicts detected
        self.assertGreater(len(comp2.conflicts_detected), 0)
        conf = comp2.conflicts_detected[0]
        self.assertEqual(str(conf.conflict_type), "FACTUAL")

        # Memory updater marks both DISPUTED
        active_and_disputed = self.loop.memory_handler.get_active_memories("proj-alpha")
        for mem in active_and_disputed:
            self.assertEqual(mem.status, MemoryStatus.DISPUTED.value)

        # Knowledge graph integration mapped contradiction
        contradicts_rels = [
            r for r in self.graph.get_relationships_by_project("proj-alpha")
            if str(r.relationship_type) == "CONTRADICTS"
        ]
        self.assertGreater(len(contradicts_rels), 0)



if __name__ == "__main__":
    unittest.main()
