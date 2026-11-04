"""
Demonstration of the Brain Loop subsystem in AI Brain.

Demonstrates:
1. Agent input processing.
2. Retrieval from shared project memory.
3. Context generation within a token budget.
4. Context delivered to a simulated agent.
5. Simulated agent response submitted back to Brain Loop.
6. Persistent event generation across the lifecycle.
7. Memory extraction and updating.
8. A later interaction by a different agent retrieving knowledge learned from the earlier interaction.
9. Strict project and session isolation.
"""

from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import SQLiteEventStore
from ai_brain.graph import GraphIntegrator, KnowledgeGraph
from ai_brain.loop import (
    AgentInteractionInput,
    AgentResponseInput,
    BrainLoop,
    InteractionStatus,
)
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


def main() -> None:
    print("=" * 70)
    print("AI Brain — Brain Loop Subsystem Demonstration")
    print("=" * 70)

    # 1. Initialize persistent subsystems
    event_store = SQLiteEventStore(":memory:")
    vector_provider = MockEmbeddingProvider()
    vector_store = SQLiteVectorStore(":memory:")
    vector_index = VectorIndex(storage=vector_store, provider=vector_provider)
    vector_manager = VectorMemoryManager(storage=vector_store, provider=vector_provider)

    graph = KnowledgeGraph()
    graph_integrator = GraphIntegrator(graph)

    retrieval_engine = RetrievalEngine(
        vector_index=vector_index,
        graph=graph,
        allow_partial_results=True,
    )
    context_builder = ContextBuilder(
        tokenizer=DeterministicTokenizer(),
        default_max_tokens=500,
    )

    # 2. Instantiate central Brain Loop orchestrator
    brain_loop = BrainLoop(
        event_store=event_store,
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        graph=graph,
        graph_integrator=graph_integrator,
        vector_manager=vector_manager,
    )

    project_id = "proj-payment-service"

    print("\n--- Interaction 1: Agent Architect (Session 1) ---")
    # Agent 1 (Claude / Architect) receives a task to establish the database architecture
    input_1 = AgentInteractionInput(
        interaction_id="int-arch-001",
        project_id=project_id,
        agent_id="agent-architect",
        session_id="sess-arch-01",
        content="We need to evaluate and establish the primary database choice.",
        token_budget=400,
    )
    print(f"[INPUT 1] Agent: {input_1.agent_id} | Session: {input_1.session_id}")
    print(f"          Content: '{input_1.content}'")

    # Step A: Prepare Context
    ctx_res_1 = brain_loop.prepare_context(input_1)
    print(f"[CONTEXT 1 DELIVERED] Status: {ctx_res_1.status.value}")
    print(f"                      Tokens Used: {ctx_res_1.context_package.tokens_used}/{ctx_res_1.context_package.token_budget}")
    print(f"                      Memories Found: {len(ctx_res_1.context_package.items)} (fresh project)")

    # Step B: Simulated Architect produces architectural decision
    simulated_architect_reply = "The project uses PostgreSQL 16 as its database with Prisma ORM."
    print(f"[SIMULATED AGENT 1 RESPONSE]: '{simulated_architect_reply}'")

    # Step C: Submit response back to Brain Loop
    resp_1 = AgentResponseInput(
        interaction_id=input_1.interaction_id,
        project_id=project_id,
        agent_id="agent-architect",
        session_id="sess-arch-01",
        content=simulated_architect_reply,
    )
    comp_1 = brain_loop.complete_interaction(resp_1)
    print(f"[COMPLETION 1] Status: {comp_1.status.value}")
    print(f"               Memories Extracted: {len(comp_1.extracted_memories)}")
    for m in comp_1.extracted_memories:
        print(f"               - [{m.memory_type}] {m.content} (ID: {m.memory_id})")
    print(f"               Graph Nodes Integrated: {comp_1.graph_nodes_integrated}")
    print(f"               Vectors Indexed: {comp_1.vectors_indexed}")

    print("\n--- Interaction 2: Agent Backend (Session 2, Later Time) ---")
    # Agent 2 (Codex / Backend) starts a new session in the same project and asks about the database
    input_2 = AgentInteractionInput(
        interaction_id="int-backend-002",
        project_id=project_id,
        agent_id="agent-backend",
        session_id="sess-backend-02",
        content="What database was chosen for our persistence layer?",
        token_budget=400,
    )
    print(f"[INPUT 2] Agent: {input_2.agent_id} | Session: {input_2.session_id}")
    print(f"          Content: '{input_2.content}'")

    # Step A: Prepare Context — Brain Loop retrieves memory established in Interaction 1!
    ctx_res_2 = brain_loop.prepare_context(input_2)
    print(f"[CONTEXT 2 DELIVERED] Status: {ctx_res_2.status.value}")
    print(f"                      Tokens Used: {ctx_res_2.context_package.tokens_used}/{ctx_res_2.context_package.token_budget}")
    print(f"                      Memories Retrieved: {len(ctx_res_2.context_package.items)}")
    for it in ctx_res_2.context_package.items:
        print(f"                      * [{it.memory_type}] {it.content} (ID: {it.memory_id})")

    print("\n[PROMPT CONTEXT MARKDOWN FOR AGENT 2]:")
    print("-" * 50)
    print(ctx_res_2.context_package.to_markdown())
    print("-" * 50)

    # Step B: Backend simulated reply
    simulated_backend_reply = "Understood. I will configure connection pools for PostgreSQL 16."
    resp_2 = AgentResponseInput(
        interaction_id=input_2.interaction_id,
        project_id=project_id,
        agent_id="agent-backend",
        session_id="sess-backend-02",
        content=simulated_backend_reply,
    )
    comp_2 = brain_loop.complete_interaction(resp_2)
    print(f"[COMPLETION 2] Status: {comp_2.status.value}")

    print("\n--- Interaction 3: Project Isolation Verification ---")
    # Querying from a different project ('proj-foreign')
    input_foreign = AgentInteractionInput(
        interaction_id="int-foreign-003",
        project_id="proj-foreign",
        agent_id="agent-foreign",
        session_id="sess-foreign-01",
        content="What database was chosen?",
        token_budget=400,
    )
    ctx_res_foreign = brain_loop.prepare_context(input_foreign)
    print(f"[PROJECT ISOLATION CHECK]: Query in 'proj-foreign'")
    print(f"                           Items Retrieved: {len(ctx_res_foreign.context_package.items)}")
    print(f"                           PostgreSQL Leaked: {any('PostgreSQL' in it.content for it in ctx_res_foreign.context_package.items)}")

    print("\n--- Event Audit Stream in Persistent EventStore ---")
    events = event_store.get_events_by_project(project_id)
    print(f"Total events recorded for '{project_id}': {len(events)}")
    for i, ev in enumerate(events, 1):
        print(f"  {i}. [{ev.event_type}] by {ev.actor} (Session: {ev.session_id})")

    print("\nBrain Loop demonstration completed successfully!")


if __name__ == "__main__":
    main()

