"""
Automated unit tests for Agent Abstraction, Registry, and Simulated Agents in AI Brain.
"""

from datetime import datetime
import os
import tempfile
import unittest

from ai_brain.agents import (
    Agent,
    AgentCapability,
    AgentExecutionRequest,
    AgentRecord,
    AgentRegistry,
    AgentStateError,
    AgentStatus,
    AgentTimeoutError,
    AgentType,
    CodingAgent,
    DuplicateAgentError,
    ResearchAgent,
    ReviewAgent,
    SQLiteAgentRegistryStore,
)
from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import SQLiteEventStore
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import BrainLoop, SQLiteInteractionStore
from ai_brain.memories import (
    ConflictDetector,
    ConflictManager,
    MemoryExtractor,
    MemoryStatus,
    MemoryType,
    MemoryUpdater,
)
from ai_brain.retrieval import RetrievalEngine
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
)


class TestAgentAbstractionAndRegistry(unittest.TestCase):
    """
    Tests covering:
    - Agent model immutability and validation.
    - Prohibition of secrets/API keys in ordinary configuration.
    - Agent Registry registration, duplicate protection, status updates.
    - Persistent registry storage surviving process restarts.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_agents.db")
        self.store = SQLiteAgentRegistryStore(self.db_path)
        self.registry = AgentRegistry(self.store)

    def tearDown(self) -> None:
        try:
            self.registry.close()
        except Exception:
            pass
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_agent_record_validation_and_immutability(self) -> None:
        """AgentRecord validates non-empty identifiers and is frozen."""
        record = AgentRecord(
            agent_id="agent-01",
            name="Test Researcher",
            provider="simulated",
            agent_type=AgentType.RESEARCH,
            capabilities=(AgentCapability.INFORMATION_GATHERING.value,),
            configuration={"topic": "databases"},
        )
        self.assertEqual(record.agent_id, "agent-01")
        self.assertEqual(record.status, AgentStatus.ACTIVE)
        self.assertEqual(record.configuration["topic"], "databases")

        with self.assertRaises(Exception):
            record.name = "Mutated"

    def test_prohibition_of_secrets_in_agent_configuration(self) -> None:
        """Attempting to store api_key, secret, or password in configuration raises an error."""
        with self.assertRaises(Exception) as cm:
            AgentRecord(
                agent_id="agent-bad",
                name="Leaky Agent",
                provider="simulated",
                configuration={"api_key": "sk-12345secret"},
            )
        self.assertIn("Prohibited sensitive credential", str(cm.exception))

        with self.assertRaises(Exception) as cm:
            AgentRecord(
                agent_id="agent-bad-nested",
                name="Leaky Agent",
                provider="simulated",
                configuration={"auth": {"access_token": "token-123"}},
            )
        self.assertIn("Prohibited sensitive credential", str(cm.exception))

    def test_agent_registration_and_retrieval(self) -> None:
        """Registering an agent makes it retrievable by ID and in listings."""
        agent = ResearchAgent(agent_id="researcher-1", name="Alpha Researcher")
        self.registry.register(agent)

        retrieved_agent = self.registry.get("researcher-1")
        self.assertIsNotNone(retrieved_agent)
        self.assertEqual(retrieved_agent.name, "Alpha Researcher")

        retrieved_record = self.registry.get_record("researcher-1")
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record.agent_type, AgentType.RESEARCH)

    def test_duplicate_registration_protection(self) -> None:
        """Attempting to register the same agent ID twice raises DuplicateAgentError."""
        agent1 = ResearchAgent(agent_id="unique-agent-id", name="Agent One")
        agent2 = CodingAgent(agent_id="unique-agent-id", name="Agent Two")

        self.registry.register(agent1)
        with self.assertRaises(DuplicateAgentError):
            self.registry.register(agent2)

    def test_agent_activation_and_deactivation(self) -> None:
        """Activating and deactivating agents updates lifecycle status in registry and instance."""
        agent = CodingAgent(agent_id="coder-status", name="Status Coder")
        self.registry.register(agent)

        self.assertEqual(agent.status, AgentStatus.ACTIVE)
        self.registry.deactivate("coder-status")
        self.assertEqual(self.registry.get_record("coder-status").status, AgentStatus.INACTIVE)
        self.assertEqual(agent.status, AgentStatus.INACTIVE)

        self.registry.activate("coder-status")
        self.assertEqual(self.registry.get_record("coder-status").status, AgentStatus.ACTIVE)
        self.assertEqual(agent.status, AgentStatus.ACTIVE)

    def test_agent_configuration_update(self) -> None:
        """Updating agent configuration persists without allowing secrets."""
        agent = ResearchAgent(agent_id="agent-config", name="Config Agent")
        self.registry.register(agent)

        self.registry.update_configuration("agent-config", {"max_depth": 3, "domains": ["security"]})
        record = self.registry.get_record("agent-config")
        self.assertEqual(record.configuration["max_depth"], 3)
        self.assertEqual(list(record.configuration["domains"]), ["security"])

        with self.assertRaises(Exception):
            self.registry.update_configuration("agent-config", {"auth_token": "secret"})

    def test_agent_registry_persistence_across_restarts(self) -> None:
        """Agent records survive closing and recreating the registry from persistent SQLite storage."""
        agent1 = ResearchAgent(agent_id="p-researcher", name="Persistent Researcher")
        agent2 = CodingAgent(agent_id="p-coder", name="Persistent Coder")
        self.registry.register(agent1)
        self.registry.register(agent2)

        # Close and reopen from disk
        self.registry.close()
        new_store = SQLiteAgentRegistryStore(self.db_path)
        new_registry = AgentRegistry(new_store)

        try:
            records = new_registry.list()
            self.assertEqual(len(records), 2)
            ids = {r.agent_id for r in records}
            self.assertEqual(ids, {"p-researcher", "p-coder"})

            r1 = new_registry.get_record("p-researcher")
            self.assertEqual(r1.name, "Persistent Researcher")
            self.assertEqual(r1.agent_type, AgentType.RESEARCH)
        finally:
            new_registry.close()


class TestSimulatedAgentsAndBrainLoop(unittest.TestCase):
    """
    Tests covering:
    - Shared Brain access via BrainLoop.
    - Research, Coding, and Review simulated agents.
    - Project isolation across agents.
    - Agent-specific interaction history.
    - Timeout and error handling.
    """

    def setUp(self) -> None:
        self.event_store = SQLiteEventStore(":memory:")
        self.interaction_store = SQLiteInteractionStore(":memory:")

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

        self.brain_loop = BrainLoop(
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

        self.research_agent = ResearchAgent(agent_id="sim-researcher")
        self.coding_agent = CodingAgent(agent_id="sim-coder")
        self.review_agent = ReviewAgent(agent_id="sim-reviewer")

    def tearDown(self) -> None:
        try:
            self.brain_loop.close()
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

    def test_research_agent_creates_factual_memory(self) -> None:
        """Research Agent gathers information and Brain Loop extracts MemoryType.FACT."""
        req = AgentExecutionRequest(
            project_id="proj-db-sys",
            session_id="sess-01",
            content="Research primary database options for persistence.",
        )
        resp = self.research_agent.execute(req, self.brain_loop)

        self.assertEqual(resp.agent_id, "sim-researcher")
        self.assertEqual(resp.project_id, "proj-db-sys")
        self.assertIn("PostgreSQL", resp.content)
        self.assertIsNotNone(resp.completion_result)
        self.assertGreater(len(resp.completion_result.extracted_memories), 0)

        # Verify fact was extracted and provenance was preserved
        mem = resp.completion_result.extracted_memories[0]
        self.assertEqual(mem.memory_type, MemoryType.FACT)
        self.assertEqual(mem.status, MemoryStatus.ACTIVE.value)
        self.assertIn("sim-researcher", mem.source_agent_ids)

    def test_coding_agent_utilizes_research_context(self) -> None:
        """Coding Agent receives Research Agent's memory via Brain context and proposes implementation."""
        # 1. Research Agent establishes PostgreSQL
        req_res = AgentExecutionRequest(
            project_id="proj-shared",
            session_id="sess-res",
            content="Research storage engine.",
        )
        self.research_agent.execute(req_res, self.brain_loop)

        # 2. Coding Agent in separate session retrieves Research Agent's memory through Brain
        req_code = AgentExecutionRequest(
            project_id="proj-shared",
            session_id="sess-code",
            content="Implement database persistence layer.",
            query="database architecture",
        )
        resp_code = self.coding_agent.execute(req_code, self.brain_loop)

        self.assertEqual(resp_code.agent_id, "sim-coder")
        self.assertIn("PostgreSQL", resp_code.content)
        # Verify Coding Agent's proposal created an architectural decision
        self.assertIsNotNone(resp_code.completion_result)
        self.assertGreater(len(resp_code.completion_result.extracted_memories), 0)

    def test_review_agent_identifies_contradiction_against_brain_context(self) -> None:
        """Review Agent detects contradictions when a proposal conflicts with established context."""
        # 1. Establish PostgreSQL via Research Agent
        req1 = AgentExecutionRequest(
            project_id="proj-review",
            session_id="sess-01",
            content="Research database infrastructure.",
        )
        self.research_agent.execute(req1, self.brain_loop)

        # 2. Review Agent evaluates a conflicting proposal suggesting MySQL
        req2 = AgentExecutionRequest(
            project_id="proj-review",
            session_id="sess-02",
            content="Review proposal: Configure MySQL for the application database.",
            query="database architecture",
        )
        resp2 = self.review_agent.execute(req2, self.brain_loop)

        self.assertIn("Contradiction detected", resp2.content)
        self.assertIn("PostgreSQL", resp2.content)

    def test_project_isolation_between_agents(self) -> None:
        """Agent memories in proj-alpha are strictly isolated from proj-beta."""
        # Agent 1 establishes database in proj-alpha
        req_alpha = AgentExecutionRequest(
            project_id="proj-alpha",
            session_id="sess-alpha",
            content="Research database.",
        )
        self.research_agent.execute(req_alpha, self.brain_loop)

        # Agent 2 in proj-beta queries database
        req_beta = AgentExecutionRequest(
            project_id="proj-beta",
            session_id="sess-beta",
            content="Implement database persistence.",
            query="database",
        )
        ctx_beta = self.coding_agent.prepare_context(req_beta, self.brain_loop)

        # Zero memories from proj-alpha leak into proj-beta
        self.assertEqual(len(ctx_beta.context_package.items), 0)

    def test_agent_specific_history_retrieval(self) -> None:
        """Agent get_history returns only interactions for that specific agent."""
        # Research Agent executes twice
        self.research_agent.execute(
            AgentExecutionRequest(project_id="proj-hist", session_id="sess-1", content="Research db."),
            self.brain_loop,
        )
        self.research_agent.execute(
            AgentExecutionRequest(project_id="proj-hist", session_id="sess-2", content="Research auth."),
            self.brain_loop,
        )
        # Coding Agent executes once
        self.coding_agent.execute(
            AgentExecutionRequest(project_id="proj-hist", session_id="sess-3", content="Code db."),
            self.brain_loop,
        )

        res_hist = self.research_agent.get_history(self.brain_loop, project_id="proj-hist")
        code_hist = self.coding_agent.get_history(self.brain_loop, project_id="proj-hist")

        self.assertEqual(len(res_hist), 2)
        self.assertEqual(len(code_hist), 1)
        self.assertTrue(all(r.agent_id == "sim-researcher" for r in res_hist))
        self.assertTrue(all(r.agent_id == "sim-coder" for r in code_hist))

    def test_inactive_agent_cannot_execute(self) -> None:
        """An INACTIVE or DECOMMISSIONED agent raises AgentStateError on execution."""
        inactive_agent = ResearchAgent(agent_id="agent-off")
        inactive_agent._record = inactive_agent.record.with_status(AgentStatus.INACTIVE)

        req = AgentExecutionRequest(project_id="p1", session_id="s1", content="test")
        with self.assertRaises(AgentStateError):
            inactive_agent.execute(req, self.brain_loop)

    def test_agent_execution_timeout_handling(self) -> None:
        """Agent with non-positive timeout raises AgentTimeoutError."""
        req = AgentExecutionRequest(project_id="p1", session_id="s1", content="test")
        with self.assertRaises(AgentTimeoutError):
            self.research_agent.execute(req, self.brain_loop, timeout_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
