"""
Demonstration of AI Brain Retrieval Engine:
- Vector-only match without graph relationships
- Graph-supported match reached through actual traversed relationship
- Memory appearing in both sources with combined evidence
- Correct source attribution ('vector', 'graph', or ('vector', 'graph'))
- Default exclusion of archived and superseded memories (and explicit historical inclusion)
- Project isolation and transparent relevance scoring
"""

from datetime import datetime
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_brain.graph import KnowledgeGraph, GraphIntegrator, RelationshipType
from ai_brain.graph.models import GraphRelationship
from ai_brain.memories import Memory, MemoryType, MemoryStatus
from ai_brain.retrieval import (
    RetrievalEngine,
    QueryFilters,
    HybridFusion,
    RelevanceScorer,
)
from ai_brain.vectors import (
    MockEmbeddingProvider,
    SQLiteVectorStore,
    VectorMemoryManager,
)


def main() -> None:
    print("=== AI Brain: Retrieval Engine Demonstration ===\n")

    project_id = "proj-ecommerce"
    isolated_project = "proj-classified"

    # 1. Initialize Subsystems
    provider = MockEmbeddingProvider()
    store = SQLiteVectorStore(":memory:")
    manager = VectorMemoryManager(storage=store, provider=provider)
    graph = KnowledgeGraph()
    integrator = GraphIntegrator(graph)

    engine = RetrievalEngine(
        vector_index=manager.index,
        graph=graph,
        fusion=HybridFusion(rrf_k=60, vector_weight=1.0, graph_weight=1.0),
        scorer=RelevanceScorer(),
        allow_partial_results=True,
    )

    # 2. Seed Knowledge Base with Memories
    # mem_db: In vector store AND connected to mem_orm in graph
    mem_db = Memory.create(
        memory_id="mem-db-101",
        memory_type=MemoryType.FACT,
        content="The application persistence layer runs on PostgreSQL 15 with connection pooling.",
        project_id=project_id,
        importance=0.9,
        confidence=1.0,
        source_agent_ids=["agent:claude", "agent:codex"],
        source_event_ids=["evt-001", "evt-002"],
    )

    # mem_orm: In vector store AND points to mem_db via DEPENDS_ON relationship
    mem_orm = Memory.create(
        memory_id="mem-orm-102",
        memory_type=MemoryType.DECISION,
        content="Prisma ORM is used for relational database migrations and typesafe queries.",
        project_id=project_id,
        importance=0.8,
        confidence=0.95,
        source_agent_ids=["agent:claude"],
        source_event_ids=["evt-003"],
    )

    # mem_auth: Vector-only match that has NO graph relationships
    mem_auth = Memory.create(
        memory_id="mem-auth-103",
        memory_type=MemoryType.FACT,
        content="User authentication utilizes asymmetric JWT tokens with RS256 signing.",
        project_id=project_id,
        importance=0.85,
        confidence=1.0,
        source_agent_ids=["agent:auth-bot"],
        source_event_ids=["evt-004"],
    )

    # mem_graph_only: Graph-only match reached through an actual relationship from mem_db
    mem_graph_only = Memory.create(
        memory_id="mem-db-backup-105",
        memory_type=MemoryType.DECISION,
        content="Automated WAL archive shipping is configured for database disaster recovery.",
        project_id=project_id,
        importance=0.75,
        confidence=0.9,
        source_agent_ids=["agent:infra-bot"],
    )

    # mem_disputed: Disputed memory (penalized status multiplier 0.70)
    mem_disputed = Memory.create(
        memory_id="mem-cache-104",
        memory_type=MemoryType.FACT,
        content="Redis cluster was disabled due to memory leak issues.",
        project_id=project_id,
        status=MemoryStatus.DISPUTED.value,
        importance=0.7,
        confidence=0.6,
        source_agent_ids=["agent:cache-bot"],
    )

    # mem_superseded & mem_archived: Historical memories (excluded by default)
    mem_superseded = Memory.create(
        memory_id="mem-db-old-099",
        memory_type=MemoryType.FACT,
        content="Legacy database was MySQL 5.7.",
        project_id=project_id,
        status=MemoryStatus.SUPERSEDED.value,
    )
    mem_archived = Memory.create(
        memory_id="mem-task-archived-098",
        memory_type=MemoryType.TASK,
        content="Completed migration task to cloud provider.",
        project_id=project_id,
        status=MemoryStatus.ARCHIVED.value,
    )

    # mem_isolated: Belongs to different project
    mem_isolated = Memory.create(
        memory_id="mem-secret-999",
        memory_type=MemoryType.FACT,
        content="Secret vault credentials for classified deployment.",
        project_id=isolated_project,
        importance=1.0,
        confidence=1.0,
    )

    # Index into vector store: mem_db, mem_orm, mem_auth, mem_disputed, mem_superseded, mem_archived, mem_isolated
    # (mem_graph_only is left out of vector store to demonstrate pure graph retrieval)
    for mem in [mem_db, mem_orm, mem_auth, mem_disputed, mem_superseded, mem_archived, mem_isolated]:
        manager.index_memory(mem)

    # Index all into knowledge graph
    for mem in [mem_db, mem_orm, mem_auth, mem_graph_only, mem_disputed, mem_superseded, mem_archived, mem_isolated]:
        integrator.integrate_memory(mem)

    # Establish explicit graph relationships:
    # 1. mem_orm DEPENDS_ON mem_db
    graph.add_relationship(
        GraphRelationship.create(
            relationship_id="rel:depends:orm:db",
            source_id="memory:mem-orm-102",
            target_id="memory:mem-db-101",
            relationship_type=RelationshipType.DEPENDS_ON,
            project_id=project_id,
        )
    )
    # 2. mem_graph_only SUPPORTS mem_db
    graph.add_relationship(
        GraphRelationship.create(
            relationship_id="rel:supports:backup:db",
            source_id="memory:mem-db-backup-105",
            target_id="memory:mem-db-101",
            relationship_type=RelationshipType.SUPPORTS,
            project_id=project_id,
        )
    )

    print(f"Indexed 6 memories in '{project_id}' and 1 memory in '{isolated_project}'.\n")

    # 3. Hybrid Retrieval Demonstration: Showing Source Attribution
    query_text = "What relational database and migration tool do we use?"
    print(f"--- 1. Hybrid Retrieval & Honest Evidence Attribution (Query: '{query_text}') ---")
    res_hybrid = engine.retrieve_hybrid(query_text, project_id=project_id, top_k=4)
    for idx, item in enumerate(res_hybrid.results, 1):
        print(f"{idx}. [{item.score:.4f}] ({item.memory_id}) {item.content}")
        print(f"   Sources: {item.sources}")
        print(f"   Vector Score: {item.vector_score} | Graph Score: {item.graph_score}")
        if item.graph_metadata:
            print(f"   Graph Traversal Info: depth={item.graph_metadata.get('depth')}, "
                  f"seed={item.graph_metadata.get('seed_memory_id')}, "
                  f"traversed={item.graph_metadata.get('relationships_traversed')}")
        else:
            print(f"   Graph Traversal Info: None (Discovered purely via vector similarity)")

    # 4. Default Exclusion of SUPERSEDED and ARCHIVED Memories
    print(f"\n--- 2. Default Status Filtering (Exclusion of SUPERSEDED & ARCHIVED) ---")
    res_default = engine.retrieve("database", project_id=project_id)
    default_ids = [r.memory_id for r in res_default.results]
    print(f"Active/Disputed memories retrieved: {default_ids}")
    print(f"mem-db-old-099 (SUPERSEDED) excluded: {'mem-db-old-099' not in default_ids}")
    print(f"mem-task-archived-098 (ARCHIVED) excluded: {'mem-task-archived-098' not in default_ids}")

    # 5. Explicit Historical Status Inclusion
    print(f"\n--- 3. Explicit Historical Status Inclusion ---")
    res_hist = engine.retrieve(
        "database",
        project_id=project_id,
        filters=QueryFilters(allowed_statuses=("ACTIVE", "SUPERSEDED")),
    )
    hist_ids = [r.memory_id for r in res_hist.results]
    print(f"Memories with SUPERSEDED enabled: {hist_ids}")
    print(f"mem-db-old-099 (SUPERSEDED) included: {'mem-db-old-099' in hist_ids}")
    print(f"Response Warnings: {res_hist.warnings}")

    # 6. Disputed Memory Penalty Check
    print("\n--- 4. Disputed Memory Penalty Check ---")
    res_disp = engine.retrieve("Redis cluster cache", project_id=project_id)
    for item in res_disp.results:
        if item.is_disputed:
            print(f"Disputed memory '{item.memory_id}' status multiplier: {item.score_breakdown.status_multiplier} (Penalized)")

    # 7. Project Isolation Check
    print(f"\n--- 5. Project Isolation Check ---")
    res_iso = engine.retrieve("Secret vault credentials", project_id=project_id)
    matched_ids = [r.memory_id for r in res_iso.results]
    print(f"Results for '{project_id}': {len(matched_ids)} matches (mem-secret-999 excluded: {'mem-secret-999' not in matched_ids})")

    print("\n=== Retrieval Engine Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
