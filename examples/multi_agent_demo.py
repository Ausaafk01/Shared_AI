"""
Multi-Agent System Demonstration for AI Brain.

Demonstrates:
1. Registering three simulated agents: Research, Coding, and Review Agents.
2. Knowledge sharing through AI Brain (Researcher discovers fact -> Coder uses fact in context -> Reviewer evaluates).
3. Agent switching within a session and across sessions with audit events.
4. Session management, continuity, and session resumption.
5. Multi-agent collaboration workflow execution using CollaborationEngine.
6. 100% offline execution without external network access or credentials.
"""

from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_brain.agents import (
    AgentCapability,
    AgentExecutionRequest,
    AgentRegistry,
    AgentSwitcher,
    CodingAgent,
    CollaborationEngine,
    ResearchAgent,
    ReviewAgent,
    SessionManager,
    SQLiteAgentRegistryStore,
    SQLiteSessionStore,
)
from ai_brain.context import ContextBuilder, DeterministicTokenizer
from ai_brain.events import EventType, SQLiteEventStore
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


def main() -> None:
    print("=" * 75)
    print("AI Brain — Phase 2: Multi-Agent System Demonstration")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Initialize Subsystems and Brain Loop
    # ---------------------------------------------------------
    print("\n[Step 1] Initializing Core Brain & Storage Subsystems...")
    event_store = SQLiteEventStore(":memory:")
    interaction_store = SQLiteInteractionStore(":memory:")
    agent_store = SQLiteAgentRegistryStore(":memory:")
    session_store = SQLiteSessionStore(":memory:")

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
        default_max_tokens=800,
    )
    brain_loop = BrainLoop(
        event_store=event_store,
        interaction_store=interaction_store,
        retrieval_engine=retrieval_engine,
        context_builder=context_builder,
        memory_extractor=MemoryExtractor(),
        memory_updater=MemoryUpdater(),
        conflict_detector=ConflictDetector(),
        conflict_manager=ConflictManager(),
        graph=graph,
        graph_integrator=graph_integrator,
        vector_manager=vector_manager,
    )
    session_manager = SessionManager(store=session_store, event_store=event_store)
    agent_switcher = AgentSwitcher(
        brain_loop=brain_loop,
        session_manager=session_manager,
        event_store=event_store,
    )
    registry = AgentRegistry(agent_store)
    print("  ✓ Brain Loop, Stores, Session Manager, and Switcher initialized.")

    # ---------------------------------------------------------
    # 2. Register Three Specialized Simulated Agents
    # ---------------------------------------------------------
    print("\n[Step 2] Registering Simulated Agents in Agent Registry...")
    researcher = ResearchAgent(
        agent_id="agent-research",
        name="Research Specialist",
        configuration={"domain": "database-architecture"},
    )
    coder = CodingAgent(
        agent_id="agent-coding",
        name="Lead Systems Architect",
        configuration={"language": "python", "framework": "asyncio"},
    )
    reviewer = ReviewAgent(
        agent_id="agent-review",
        name="Quality & Safety Reviewer",
        configuration={"strict_mode": True},
    )

    registry.register(researcher.record)
    registry.register(coder.record)
    registry.register(reviewer.record)

    registered_agents = registry.list()
    print(f"  ✓ Registered {len(registered_agents)} agents in persistent registry:")
    for a in registered_agents:
        print(f"    - [{a.agent_id}] '{a.name}' ({a.agent_type.value}) - Status: {a.status.value}")

    # ---------------------------------------------------------
    # 3. Create Session and Execute Research Agent
    # ---------------------------------------------------------
    project_id = "project-phoenix"
    session_id = "session-001"
    print(f"\n[Step 3] Starting Session '{session_id}' for Project '{project_id}'...")
    session_manager.create_session(
        session_id=session_id,
        project_id=project_id,
        initial_agent_id=researcher.agent_id,
        metadata={"objective": "Design high-throughput caching architecture"},
    )

    print(f"\n[3.1] Executing Research Agent '{researcher.agent_id}'...")
    research_req = AgentExecutionRequest(
        project_id=project_id,
        session_id=session_id,
        content="Investigate distributed caching requirements for Phoenix: The project uses Redis Sentinel for cache high availability.",
        query="caching architecture requirements",
    )
    research_resp = researcher.execute(research_req, brain_loop)
    print(f"  Research Output: {research_resp.content}")
    print(f"  Memories Extracted: {len(research_resp.completion_result.extracted_memories)}")
    for mem in research_resp.completion_result.extracted_memories:
        print(f"    * Stored {str(mem.memory_type).upper()}: '{mem.content}'")

    # ---------------------------------------------------------
    # 4. Agent Switching within Session (Researcher -> Coder)
    # ---------------------------------------------------------
    print(f"\n[Step 4] Handing over execution from '{researcher.agent_id}' to '{coder.agent_id}'...")
    switch_context = agent_switcher.switch(
        from_agent_id=researcher.agent_id,
        to_agent=coder,
        project_id=project_id,
        session_id=session_id,
        handover_query="caching architecture Redis",
        reason="Handoff to developer for implementation design",
    )
    current_session = session_manager.get_session(session_id)
    active_agent = current_session.current_agent_id if current_session else "unknown"
    print(f"  ✓ Handover complete. Session '{session_id}' active participant: '{active_agent}'")
    tokens = switch_context.context_package.tokens_used if switch_context.context_package else 0
    print(f"  ✓ Assembled Context Tokens: {tokens}")

    # Coding Agent executes using the retrieved Research context
    print(f"\n[4.1] Executing Coding Agent '{coder.agent_id}' with shared Brain context...")
    coder_req = AgentExecutionRequest(
        project_id=project_id,
        session_id=session_id,
        content="Formulate implementation plan based on the research findings.",
        query="caching architecture Redis",
    )
    coder_resp = coder.execute(coder_req, brain_loop)
    print(f"  Coder Output: {coder_resp.content}")
    print(f"  Memories Extracted: {len(coder_resp.completion_result.extracted_memories)}")
    for mem in coder_resp.completion_result.extracted_memories:
        print(f"    * Stored {str(mem.memory_type).upper()}: '{mem.content}'")

    # ---------------------------------------------------------
    # 5. Cross-Session Continuity & Review Agent
    # ---------------------------------------------------------
    session_id_2 = "session-002"
    print(f"\n[Step 5] Starting Cross-Session Review in '{session_id_2}'...")
    session_manager.create_session(
        session_id=session_id_2,
        project_id=project_id,
        initial_agent_id=reviewer.agent_id,
        metadata={"objective": "Evaluate architecture decisions against project facts"},
    )

    print(f"\n[5.1] Executing Review Agent '{reviewer.agent_id}' to evaluate consistency...")
    review_req = AgentExecutionRequest(
        project_id=project_id,
        session_id=session_id_2,
        content="Review proposal: We should use Memcached instead of Redis Sentinel.",
        query="Redis Sentinel cache",
    )
    review_resp = reviewer.execute(review_req, brain_loop)
    print(f"  Reviewer Output:\n{review_resp.content}")

    # Verify audit events recorded in EventStore
    events = event_store.get_events_by_project(project_id)
    switch_events = [e for e in events if e.event_type == EventType.AGENT_SWITCH]
    session_events = [e for e in events if e.event_type in (EventType.SESSION_START, EventType.SESSION_END)]
    print(f"\n[Step 6] Audit Event Trail for Project '{project_id}':")
    print(f"  ✓ Total Events: {len(events)}")
    print(f"  ✓ Session Lifecycle Events: {len(session_events)}")
    print(f"  ✓ Agent Switch Audit Events: {len(switch_events)}")

    # ---------------------------------------------------------
    # 6. Multi-Agent Collaboration Workflow
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("Running Multi-Agent Collaboration Workflow (CollaborationEngine)")
    print("=" * 75)

    collab_engine = CollaborationEngine()
    workflow_session = "session-collab-001"
    session_manager.create_session(
        session_id=workflow_session,
        project_id="project-collab",
        initial_agent_id=researcher.agent_id,
    )

    agents_map = {
        researcher.agent_id: researcher,
        coder.agent_id: coder,
        reviewer.agent_id: reviewer,
    }

    workflow = collab_engine.create_research_coding_review_workflow(
        project_id="project-collab",
        session_id=workflow_session,
        task_description="Build resilient event streaming pipeline",
        research_agent_id=researcher.agent_id,
        coding_agent_id=coder.agent_id,
        review_agent_id=reviewer.agent_id,
    )
    print(f"  Workflow Created: '{workflow.workflow_id}' with {len(workflow.steps)} stages.")

    executed_wf = collab_engine.execute_workflow(workflow, agents_map, brain_loop)
    print(f"  Workflow Execution Status: {executed_wf.status.value.upper()}")
    for step_res in executed_wf.completed_steps:
        print(f"    - Step {step_res.step_index}: '{step_res.name}' [{step_res.agent_id}] -> {step_res.status}")
        print(f"      Result: {step_res.output_content[:90]}...")

    print("\n" + "=" * 75)
    print("AI Brain Phase 2 Multi-Agent Demonstration Complete: All Stages Succeeded!")
    print("=" * 75)


if __name__ == "__main__":
    main()
