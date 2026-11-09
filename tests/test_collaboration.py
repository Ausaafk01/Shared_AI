"""
Automated unit tests for Multi-Agent Collaboration Workflows in AI Brain.
"""

import unittest
from unittest.mock import MagicMock

from ai_brain.agents import (
    CodingAgent,
    CollaborationEngine,
    CollaborationWorkflowError,
    ResearchAgent,
    ReviewAgent,
    WorkflowStatus,
)
from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import SQLiteEventStore
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import BrainLoop, SQLiteInteractionStore
from ai_brain.memories import (
    ConflictDetector,
    ConflictManager,
    MemoryExtractor,
    MemoryUpdater,
)
from ai_brain.retrieval import RetrievalEngine
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorIndex,
    VectorMemoryManager,
)


class TestMultiAgentCollaboration(unittest.TestCase):
    """
    Tests covering:
    - Multi-agent collaboration workflow creation and execution.
    - Context passing between agents through AI Brain.
    - Partial failure handling and state tracking.
    - Interrupted workflow resumption without re-executing completed steps.
    - Preserving conflicting perspectives during collaboration.
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
        self.brain_loop = BrainLoop(
            event_store=self.event_store,
            interaction_store=self.interaction_store,
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

        self.researcher = ResearchAgent(agent_id="researcher-collab")
        self.coder = CodingAgent(agent_id="coder-collab")
        self.reviewer = ReviewAgent(agent_id="reviewer-collab")

        self.agents = {
            "researcher-collab": self.researcher,
            "coder-collab": self.coder,
            "reviewer-collab": self.reviewer,
        }

        self.engine = CollaborationEngine()

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

    def test_complete_three_agent_collaboration_workflow(self) -> None:
        """Full ordered collaboration: Researcher -> Coder -> Reviewer passing context through Brain."""
        workflow = self.engine.create_research_coding_review_workflow(
            project_id="proj-collab",
            session_id="sess-collab-1",
            task_description="Evaluate and implement primary database architecture",
            research_agent_id="researcher-collab",
            coding_agent_id="coder-collab",
            review_agent_id="reviewer-collab",
        )
        self.assertEqual(workflow.status, WorkflowStatus.PENDING)
        self.assertEqual(len(workflow.steps), 3)

        result = self.engine.execute_workflow(workflow, self.agents, self.brain_loop)

        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(len(result.completed_steps), 3)
        self.assertTrue(all(s.status == "COMPLETED" for s in result.completed_steps))

        # Step 1: Research output establishes PostgreSQL
        self.assertIn("PostgreSQL", result.completed_steps[0].output_content)
        # Step 2: Coder proposed implementation based on Research output
        self.assertIn("PostgreSQL", result.completed_steps[1].output_content)
        # Step 3: Reviewer validated proposal against established facts
        self.assertIn("Review verdict", result.completed_steps[2].output_content)

    def test_collaboration_partial_failure_and_resumption(self) -> None:
        """If step 2 fails, step 1 is preserved and workflow can be resumed without re-executing step 1."""
        # Create a failing coder agent
        failing_coder = CodingAgent(agent_id="coder-fail")
        failing_coder.generate_response = MagicMock(side_effect=RuntimeError("Compilation error"))

        agents_with_fail = {
            "researcher-collab": self.researcher,
            "coder-fail": failing_coder,
            "reviewer-collab": self.reviewer,
        }

        workflow = self.engine.create_research_coding_review_workflow(
            project_id="proj-resume",
            session_id="sess-resume-1",
            task_description="Implement database caching",
            research_agent_id="researcher-collab",
            coding_agent_id="coder-fail",
            review_agent_id="reviewer-collab",
        )

        with self.assertRaises(CollaborationWorkflowError):
            self.engine.execute_workflow(workflow, agents_with_fail, self.brain_loop)

        # Check recorded state
        persisted = self.engine.get_workflow(workflow.workflow_id)
        self.assertEqual(persisted.status, WorkflowStatus.FAILED)
        self.assertEqual(len(persisted.completed_steps), 2)
        self.assertEqual(persisted.completed_steps[0].status, "COMPLETED")
        self.assertEqual(persisted.completed_steps[1].status, "FAILED")

        # Now fix coder agent and resume workflow
        recovered_coder = CodingAgent(agent_id="coder-fail")
        agents_recovered = {
            "researcher-collab": self.researcher,
            "coder-fail": recovered_coder,
            "reviewer-collab": self.reviewer,
        }

        resumed = self.engine.resume_workflow(workflow.workflow_id, agents_recovered, self.brain_loop)
        self.assertEqual(resumed.status, WorkflowStatus.COMPLETED)
        # Step 1 was not re-executed; step 2 and 3 executed successfully
        self.assertEqual(len(resumed.completed_steps), 3)
        self.assertEqual(resumed.completed_steps[1].status, "COMPLETED")
        self.assertEqual(resumed.completed_steps[2].status, "COMPLETED")

    def test_collaboration_preserves_conflicting_perspectives(self) -> None:
        """When an agent in a workflow introduces a contradiction, conflict is captured and preserved."""
        # Custom reviewer that contradicts established database
        contrarian_reviewer = ReviewAgent(
            agent_id="reviewer-contra",
            response_generator=lambda ctx, req: "Review critique: The project uses MySQL as its database.",
        )
        agents_with_conflict = {
            "researcher-collab": self.researcher,
            "coder-collab": self.coder,
            "reviewer-contra": contrarian_reviewer,
        }

        workflow = self.engine.create_research_coding_review_workflow(
            project_id="proj-disagree",
            session_id="sess-disagree",
            task_description="Database specification",
            research_agent_id="researcher-collab",
            coding_agent_id="coder-collab",
            review_agent_id="reviewer-contra",
        )
        res = self.engine.execute_workflow(workflow, agents_with_conflict, self.brain_loop)

        self.assertEqual(res.status, WorkflowStatus.COMPLETED)
        # Step 3 had conflict detected
        self.assertGreater(res.completed_steps[2].conflicts_detected_count, 0)


if __name__ == "__main__":
    unittest.main()

