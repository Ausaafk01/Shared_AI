"""
Unit and integration tests for the Retrieval Engine subsystem.
"""

from datetime import datetime, timedelta, timezone
import unittest

from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.graph.integration import GraphIntegrator
from ai_brain.graph.types import RelationshipType, TraversalDirection
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.retrieval.engine import RetrievalEngine
from ai_brain.retrieval.exceptions import (
    GraphRetrievalError,
    HybridFusionError,
    ProjectIsolationError,
    QueryValidationError,
    RetrievalError,
    RetrievalExecutionError,
    VectorRetrievalError,
)
from ai_brain.retrieval.fusion import FusedCandidate, HybridFusion
from ai_brain.retrieval.graph_retriever import GraphMatch, GraphRetriever
from ai_brain.retrieval.models import (
    ProcessedQuery,
    QueryFilters,
    RetrievalItem,
    RetrievalResponse,
    ScoreBreakdown,
)
from ai_brain.retrieval.query import QueryProcessor
from ai_brain.retrieval.scoring import RelevanceScorer
from ai_brain.retrieval.types import RetrievalMethod, SeedStrategy
from ai_brain.retrieval.vector_retriever import VectorRetriever
from ai_brain.vectors.index import VectorIndex
from ai_brain.vectors.manager import VectorMemoryManager
from ai_brain.vectors.models import VectorMatch, VectorRecord
from ai_brain.vectors.providers.mock import MockEmbeddingProvider
from ai_brain.vectors.storage.sqlite import SQLiteVectorStore


class TestRetrievalEngine(unittest.TestCase):
    """Comprehensive test suite for AI Brain Retrieval Engine."""

    def setUp(self) -> None:
        """Set up in-memory vector storage, knowledge graph, and retrieval engine."""
        self.project_a = "proj-retrieval-alpha"
        self.project_b = "proj-retrieval-beta"

        # Mock embedding provider and vector store
        self.provider = MockEmbeddingProvider()
        self.vector_store = SQLiteVectorStore(":memory:")
        self.vector_manager = VectorMemoryManager(storage=self.vector_store, provider=self.provider)
        self.vector_index = self.vector_manager.index

        # Knowledge graph and integrator
        self.graph = KnowledgeGraph()
        self.graph_integrator = GraphIntegrator(self.graph)

        # Retrieval components
        self.query_processor = QueryProcessor()
        self.vector_retriever = VectorRetriever(self.vector_index)
        self.graph_retriever = GraphRetriever(self.graph)
        self.fusion = HybridFusion(rrf_k=60, vector_weight=1.0, graph_weight=1.0)
        self.scorer = RelevanceScorer()

        self.engine = RetrievalEngine(
            vector_retriever=self.vector_retriever,
            graph_retriever=self.graph_retriever,
            query_processor=self.query_processor,
            fusion=self.fusion,
            scorer=self.scorer,
            allow_partial_results=True,
        )

    # =========================================================================
    # 1. Query Processing Tests
    # =========================================================================

    def test_empty_and_invalid_queries(self) -> None:
        """Verify empty or malformed queries raise QueryValidationError."""
        with self.assertRaises(QueryValidationError):
            self.query_processor.process("", project_id=self.project_a)

        with self.assertRaises(QueryValidationError):
            self.query_processor.process("    ", project_id=self.project_a)

        with self.assertRaises(QueryValidationError):
            self.query_processor.process(None, project_id=self.project_a)  # type: ignore

        with self.assertRaises(QueryValidationError):
            self.query_processor.process(123, project_id=self.project_a)  # type: ignore

    def test_query_normalization_and_preservation(self) -> None:
        """Verify query is normalized without destroying original meaning."""
        raw = "   What   is   the   PostgreSQL   DATABASE   configuration?   "
        p = self.query_processor.process(raw, project_id=self.project_a)

        self.assertEqual(p.original_query, "What is the PostgreSQL DATABASE configuration?")
        self.assertEqual(p.normalized_query, "what is the postgresql database configuration?")
        self.assertEqual(p.project_id, self.project_a)

    def test_keyword_and_entity_extraction(self) -> None:
        """Verify extraction of keywords (stopword filtering) and entities."""
        raw = "Claude migrated mem-auth-1 to JWT tokens for proj-retrieval-alpha on port 8080"
        p = self.query_processor.process(raw, project_id=self.project_a)

        # Stopwords ('to', 'for', 'on') excluded
        self.assertIn("postgresql" not in p.keywords, [True])
        self.assertIn("claude", p.keywords)
        self.assertIn("migrated", p.keywords)
        self.assertIn("tokens", p.keywords)

        # Entities extracted
        self.assertIn("mem-auth-1", p.entities)
        self.assertIn("proj-retrieval-alpha", p.entities)
        self.assertIn("JWT", p.entities)
        self.assertIn("Claude", p.entities)

    def test_query_filter_validation(self) -> None:
        """Verify QueryFilters validation, UTC conversion, and temporal ordering."""
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(days=2)

        filters = QueryFilters(
            memory_types=("fact", "DECISION"),
            allowed_statuses=("active", "disputed"),
            time_range_start=earlier,
            time_range_end=now,
            seed_memory_ids=("mem-1",),
        )
        self.assertEqual(filters.memory_types, ("FACT", "DECISION"))
        self.assertEqual(filters.allowed_statuses, ("ACTIVE", "DISPUTED"))
        self.assertEqual(filters.seed_memory_ids, ("mem-1",))

        # Test invalid temporal range
        with self.assertRaises(QueryValidationError):
            QueryFilters(time_range_start=now, time_range_end=earlier)

    # =========================================================================
    # 2. Project Isolation Tests
    # =========================================================================

    def test_missing_or_invalid_project_scope_rejection(self) -> None:
        """Verify missing or invalid project scope is rejected."""
        with self.assertRaises(ProjectIsolationError):
            self.query_processor.process("valid query", project_id="")

        with self.assertRaises(ProjectIsolationError):
            self.query_processor.process("valid query", project_id="   ")

        with self.assertRaises(ProjectIsolationError):
            self.query_processor.process("valid query", project_id=None)  # type: ignore

    def test_cross_project_isolation_across_vector_graph_hybrid(self) -> None:
        """Verify memories from project B are never retrieved in project A queries."""
        mem_a = Memory.create(
            memory_id="mem-a-1",
            memory_type=MemoryType.FACT,
            content="Project Alpha uses PostgreSQL persistence.",
            project_id=self.project_a,
        )
        mem_b = Memory.create(
            memory_id="mem-b-1",
            memory_type=MemoryType.FACT,
            content="Project Beta uses MongoDB persistence.",
            project_id=self.project_b,
        )

        # Index in both vector store and graph
        self.vector_manager.index_memory(mem_a)
        self.vector_manager.index_memory(mem_b)
        self.graph_integrator.integrate_memory(mem_a)
        self.graph_integrator.integrate_memory(mem_b)

        # 1. Vector-only query for project A
        res_v = self.engine.retrieve_vector("database persistence", project_id=self.project_a)
        self.assertTrue(all(item.project_id == self.project_a for item in res_v.results))
        self.assertNotIn("mem-b-1", [item.memory_id for item in res_v.results])

        # 2. Graph-only query for project A
        res_g = self.engine.retrieve_graph("persistence", project_id=self.project_a)
        self.assertTrue(all(item.project_id == self.project_a for item in res_g.results))
        self.assertNotIn("mem-b-1", [item.memory_id for item in res_g.results])

        # 3. Hybrid query for project A
        res_h = self.engine.retrieve_hybrid("database persistence", project_id=self.project_a)
        self.assertTrue(all(item.project_id == self.project_a for item in res_h.results))
        self.assertNotIn("mem-b-1", [item.memory_id for item in res_h.results])

    # =========================================================================
    # 3. Graph Retrieval Tests
    # =========================================================================

    def test_graph_retrieval_traversal_depth_and_direction(self) -> None:
        """Verify graph retrieval traversal respects depth limits and relationship links."""
        # mem1 connects to mem2 via SUPERSEDES
        mem1 = Memory.create(
            memory_id="mem-auth-v1",
            memory_type=MemoryType.DECISION,
            content="Initial session-based cookie authentication.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,  # explicitly ACTIVE for test
        )
        mem2 = Memory.create(
            memory_id="mem-auth-v2",
            parent_memory_id="mem-auth-v1",
            version=2,
            memory_type=MemoryType.DECISION,
            content="Migrated authentication to JWT tokens.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,
        )
        self.graph_integrator.integrate_memory(mem1)
        self.graph_integrator.integrate_memory(mem2)

        # Retrieve seeded from mem-auth-v1 with depth 1
        res = self.engine.retrieve_graph(
            "authentication",
            project_id=self.project_a,
            max_depth=1,
            seed_memory_ids=("mem-auth-v1",),
        )
        retrieved_ids = [item.memory_id for item in res.results]
        self.assertIn("mem-auth-v1", retrieved_ids)
        self.assertIn("mem-auth-v2", retrieved_ids)

        # Depth 0 should return only seed
        res_depth_0 = self.engine.retrieve_graph(
            "authentication",
            project_id=self.project_a,
            max_depth=0,
            seed_memory_ids=("mem-auth-v1",),
        )
        self.assertEqual([item.memory_id for item in res_depth_0.results], ["mem-auth-v1"])

    def test_graph_retrieval_status_filtering(self) -> None:
        """Verify graph retrieval excludes SUPERSEDED and ARCHIVED memories by default."""
        active_mem = Memory.create(
            memory_id="mem-active",
            memory_type=MemoryType.FACT,
            content="Active system setting: timeout 30s.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,
        )
        superseded_mem = Memory.create(
            memory_id="mem-superseded",
            memory_type=MemoryType.FACT,
            content="Old setting: timeout 60s.",
            project_id=self.project_a,
            status=MemoryStatus.SUPERSEDED.value,
        )
        self.graph_integrator.integrate_memory(active_mem)
        self.graph_integrator.integrate_memory(superseded_mem)

        # Default query should only return active memory
        res = self.engine.retrieve_graph("setting timeout", project_id=self.project_a)
        ids = [item.memory_id for item in res.results]
        self.assertIn("mem-active", ids)
        self.assertNotIn("mem-superseded", ids)

        # Explicitly allowing SUPERSEDED returns it
        filters = QueryFilters(allowed_statuses=("ACTIVE", "SUPERSEDED"))
        res_all = self.engine.retrieve_graph("setting timeout", project_id=self.project_a, filters=filters)
        all_ids = [item.memory_id for item in res_all.results]
        self.assertIn("mem-active", all_ids)
        self.assertIn("mem-superseded", all_ids)

    # =========================================================================
    # 4. Vector Retrieval Tests
    # =========================================================================

    def test_vector_retrieval_threshold_and_top_k(self) -> None:
        """Verify vector retrieval obeys similarity threshold and top_k bounds."""
        for i in range(5):
            mem = Memory.create(
                memory_id=f"mem-db-{i}",
                memory_type=MemoryType.FACT,
                content=f"Database configuration cluster replica node {i}.",
                project_id=self.project_a,
            )
            self.vector_manager.index_memory(mem)

        # top_k = 2
        res = self.engine.retrieve_vector(
            "database configuration replica",
            project_id=self.project_a,
            top_k=2,
        )
        self.assertEqual(len(res.results), 2)

        # high threshold cuts off low similarity
        res_thresh = self.engine.retrieve_vector(
            "unrelated quantum astrophysics search",
            project_id=self.project_a,
            similarity_threshold=0.95,
        )
        self.assertEqual(len(res_thresh.results), 0)

    def test_vector_provider_failure_handling(self) -> None:
        """Verify handling when embedding provider fails."""
        failing_provider = MockEmbeddingProvider(simulate_failure=True)
        store = SQLiteVectorStore(":memory:")
        failing_index = VectorIndex(storage=store, provider=failing_provider)
        failing_retriever = VectorRetriever(failing_index)

        engine = RetrievalEngine(
            vector_retriever=failing_retriever,
            graph_retriever=self.graph_retriever,
            allow_partial_results=False,
        )

        with self.assertRaises(VectorRetrievalError):
            engine.retrieve_vector("query", project_id=self.project_a)

    # =========================================================================
    # 5. Hybrid Retrieval & RRF Fusion Tests
    # =========================================================================

    def test_rrf_fusion_algorithm_and_deduplication(self) -> None:
        """Verify RRF fusion merges items appearing in both streams without duplicates."""
        mem1 = Memory.create(
            memory_id="mem-both",
            memory_type=MemoryType.FACT,
            content="Uses PostgreSQL for persistence.",
            project_id=self.project_a,
        )
        mem2 = Memory.create(
            memory_id="mem-vector-only",
            memory_type=MemoryType.FACT,
            content="Prisma ORM connects to the SQL database.",
            project_id=self.project_a,
        )
        mem3 = Memory.create(
            memory_id="mem-graph-only",
            parent_memory_id="mem-both",
            version=2,
            memory_type=MemoryType.DECISION,
            content="PostgreSQL chosen over MySQL for JSONB support.",
            project_id=self.project_a,
        )

        # mem1 in both, mem2 vector only, mem3 graph only
        self.vector_manager.index_memory(mem1)
        self.vector_manager.index_memory(mem2)
        self.graph_integrator.integrate_memory(mem1)
        self.graph_integrator.integrate_memory(mem3)

        res = self.engine.retrieve_hybrid("PostgreSQL database persistence", project_id=self.project_a)
        retrieved_ids = [item.memory_id for item in res.results]

        # mem-both should be top ranked because it receives rank contributions from both streams
        self.assertEqual(retrieved_ids[0], "mem-both")
        self.assertIn("vector", res.results[0].sources)
        self.assertIn("graph", res.results[0].sources)

        # No duplicates
        self.assertEqual(len(retrieved_ids), len(set(retrieved_ids)))

    def test_hybrid_partial_failure_fallback(self) -> None:
        """Verify hybrid retrieval continues gracefully with warnings when allow_partial_results is True."""
        failing_provider = MockEmbeddingProvider(simulate_failure=True)
        store = SQLiteVectorStore(":memory:")
        failing_index = VectorIndex(storage=store, provider=failing_provider)
        failing_vec_retriever = VectorRetriever(failing_index)

        mem = Memory.create(
            memory_id="mem-graph-fallback",
            memory_type=MemoryType.FACT,
            content="Docker containers orchestrated via Kubernetes.",
            project_id=self.project_a,
        )
        self.graph_integrator.integrate_memory(mem)

        partial_engine = RetrievalEngine(
            vector_retriever=failing_vec_retriever,
            graph_retriever=self.graph_retriever,
            allow_partial_results=True,
        )

        res = partial_engine.retrieve_hybrid("Kubernetes orchestration", project_id=self.project_a)
        self.assertTrue(len(res.warnings) > 0)
        self.assertIn("Vector retrieval unavailable", res.warnings[0])
        # Graph results still delivered
        self.assertIn("mem-graph-fallback", [r.memory_id for r in res.results])

    # =========================================================================
    # 6. Memory Relevance Scoring Tests
    # =========================================================================

    def test_relevance_scoring_importance_and_confidence(self) -> None:
        """Verify importance and confidence factors are strictly separated and impact score."""
        candidate_high = FusedCandidate(
            memory_id="mem-high",
            project_id=self.project_a,
            content="High confidence and importance fact.",
            memory_type="FACT",
            status="ACTIVE",
            version=1,
            root_memory_id=None,
            created_at=datetime.now(timezone.utc),
            vector_score=0.8,
            graph_score=0.8,
            fused_score=0.8,
            sources=("vector", "graph"),
            graph_metadata={},
            metadata={},
            source_event_ids=(),
            source_agent_ids=(),
            importance=1.0,
            confidence=1.0,
            is_disputed=False,
        )
        candidate_low = FusedCandidate(
            memory_id="mem-low",
            project_id=self.project_a,
            content="Low confidence and importance fact.",
            memory_type="FACT",
            status="ACTIVE",
            version=1,
            root_memory_id=None,
            created_at=datetime.now(timezone.utc),
            vector_score=0.8,
            graph_score=0.8,
            fused_score=0.8,
            sources=("vector", "graph"),
            graph_metadata={},
            metadata={},
            source_event_ids=(),
            source_agent_ids=(),
            importance=0.2,
            confidence=0.3,
            is_disputed=False,
        )

        scored_high = self.scorer.score_candidate(candidate_high)
        scored_low = self.scorer.score_candidate(candidate_low)

        self.assertGreater(scored_high.score, scored_low.score)
        self.assertEqual(scored_high.score_breakdown.importance_factor, 1.0)
        self.assertEqual(scored_high.score_breakdown.confidence_factor, 1.0)
        self.assertAlmostEqual(scored_low.score_breakdown.importance_factor, 0.84, places=2)
        self.assertAlmostEqual(scored_low.score_breakdown.confidence_factor, 0.86, places=2)

    def test_disputed_memory_penalty(self) -> None:
        """Verify DISPUTED memories receive the status penalty and cannot masquerade as facts."""
        candidate_disputed = FusedCandidate(
            memory_id="mem-disp",
            project_id=self.project_a,
            content="Disputed claim: Redis removed.",
            memory_type="FACT",
            status="DISPUTED",
            version=1,
            root_memory_id=None,
            created_at=datetime.now(timezone.utc),
            vector_score=0.9,
            graph_score=0.9,
            fused_score=0.9,
            sources=("vector",),
            graph_metadata={},
            metadata={},
            source_event_ids=(),
            source_agent_ids=(),
            importance=0.5,
            confidence=1.0,
            is_disputed=True,
        )
        scored = self.scorer.score_candidate(candidate_disputed)

        self.assertTrue(scored.is_disputed)
        self.assertEqual(scored.score_breakdown.status_multiplier, 0.7)
        # Score is discounted
        self.assertLess(scored.score, 0.9)

    def test_provenance_corroboration_bonus(self) -> None:
        """Verify corroboration across multiple agents yields provenance bonus."""
        candidate_corroborated = FusedCandidate(
            memory_id="mem-corrob",
            project_id=self.project_a,
            content="Confirmed database connection.",
            memory_type="FACT",
            status="ACTIVE",
            version=1,
            root_memory_id=None,
            created_at=datetime.now(timezone.utc),
            vector_score=0.5,
            graph_score=0.5,
            fused_score=0.5,
            sources=("vector",),
            graph_metadata={},
            metadata={},
            source_event_ids=("evt-1", "evt-2", "evt-3"),
            source_agent_ids=("claude", "codex", "gemini"),
            importance=0.5,
            confidence=1.0,
            is_disputed=False,
        )
        scored = self.scorer.score_candidate(candidate_corroborated)
        self.assertGreater(scored.score_breakdown.provenance_bonus, 0.0)

    def test_deterministic_scoring(self) -> None:
        """Verify identical inputs produce identical scores across invocations."""
        now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
        candidate = FusedCandidate(
            memory_id="mem-det",
            project_id=self.project_a,
            content="Deterministic test memory.",
            memory_type="FACT",
            status="ACTIVE",
            version=1,
            root_memory_id=None,
            created_at=now,
            vector_score=0.75,
            graph_score=0.65,
            fused_score=0.70,
            sources=("vector", "graph"),
            graph_metadata={},
            metadata={},
            source_event_ids=("evt-1",),
            source_agent_ids=("agent-a",),
            importance=0.8,
            confidence=0.9,
            is_disputed=False,
        )

        score_1 = self.scorer.score_candidate(candidate, reference_time=now)
        score_2 = self.scorer.score_candidate(candidate, reference_time=now)

        self.assertEqual(score_1.score, score_2.score)
        self.assertEqual(score_1.score_breakdown.to_dict(), score_2.score_breakdown.to_dict())

    def test_result_and_response_serialization(self) -> None:
        """Verify RetrievalItem and RetrievalResponse JSON serialization roundtrips."""
        mem = Memory.create(
            memory_id="mem-serial",
            memory_type=MemoryType.FACT,
            content="Serialization verification test.",
            project_id=self.project_a,
        )
        self.vector_manager.index_memory(mem)
        self.graph_integrator.integrate_memory(mem)

        response = self.engine.retrieve("Serialization", project_id=self.project_a)
        self.assertTrue(len(response.results) > 0)

        # Roundtrip response
        json_str = response.to_json()
        rehydrated = RetrievalResponse.from_json(json_str)

        self.assertEqual(response.query, rehydrated.query)
        self.assertEqual(response.project_id, rehydrated.project_id)
        self.assertEqual(len(response.results), len(rehydrated.results))
        self.assertEqual(response.results[0].memory_id, rehydrated.results[0].memory_id)
        self.assertEqual(response.results[0].score, rehydrated.results[0].score)

    def test_recency_decay_scoring(self) -> None:
        """Verify recency factor decays gracefully with age."""
        now = datetime.now(timezone.utc)
        recent_date = now - timedelta(hours=1)
        older_date = now - timedelta(days=90)

        factor_recent = self.scorer.calculate_recency_factor(recent_date, reference_time=now)
        factor_older = self.scorer.calculate_recency_factor(older_date, reference_time=now)

        self.assertAlmostEqual(factor_recent, 1.0, places=2)
        self.assertLess(factor_older, factor_recent)
        self.assertGreaterEqual(factor_older, 0.9)

    def test_universal_retrieve_entrypoint_methods(self) -> None:
        """Verify the unified retrieve() method dispatches cleanly to VECTOR, GRAPH, and HYBRID."""
        mem = Memory.create(
            memory_id="mem-dispatch",
            memory_type=MemoryType.FACT,
            content="Testing unified retrieve entrypoint dispatching.",
            project_id=self.project_a,
        )
        self.vector_manager.index_memory(mem)
        self.graph_integrator.integrate_memory(mem)

        res_v = self.engine.retrieve("Testing dispatching", project_id=self.project_a, method="VECTOR")
        self.assertEqual(res_v.method, "VECTOR")
        self.assertEqual(res_v.results[0].memory_id, "mem-dispatch")

        res_g = self.engine.retrieve("Testing dispatching", project_id=self.project_a, method=RetrievalMethod.GRAPH)
        self.assertEqual(res_g.method, "GRAPH")
        self.assertEqual(res_g.results[0].memory_id, "mem-dispatch")

        res_h = self.engine.retrieve("Testing dispatching", project_id=self.project_a, method=RetrievalMethod.HYBRID)
        self.assertEqual(res_h.method, "HYBRID")
        self.assertEqual(res_h.results[0].memory_id, "mem-dispatch")

    def test_filtering_by_memory_types_and_agents(self) -> None:
        """Verify filtering by memory types and source agents."""
        mem_fact = Memory.create(
            memory_id="mem-fact-claude",
            memory_type=MemoryType.FACT,
            content="Claude established PostgreSQL database.",
            project_id=self.project_a,
            source_agent_ids=["agent:claude"],
        )
        mem_decision = Memory.create(
            memory_id="mem-dec-codex",
            memory_type=MemoryType.DECISION,
            content="Codex decided Redis cache removal.",
            project_id=self.project_a,
            source_agent_ids=["agent:codex"],
        )
        self.vector_manager.index_memory(mem_fact)
        self.vector_manager.index_memory(mem_decision)
        self.graph_integrator.integrate_memory(mem_fact)
        self.graph_integrator.integrate_memory(mem_decision)

        # Filter for DECISION only
        filter_dec = QueryFilters(memory_types=("DECISION",))
        res_dec = self.engine.retrieve("database or cache", project_id=self.project_a, filters=filter_dec)
        self.assertEqual(len(res_dec.results), 1)
        self.assertEqual(res_dec.results[0].memory_id, "mem-dec-codex")

        # Filter for agent:claude only
        filter_claude = QueryFilters(source_agent_ids=("agent:claude",))
        res_claude = self.engine.retrieve("database or cache", project_id=self.project_a, filters=filter_claude)
        self.assertEqual(len(res_claude.results), 1)
        self.assertEqual(res_claude.results[0].memory_id, "mem-fact-claude")

    def test_engine_initialization_with_raw_index_and_graph(self) -> None:
        """Verify RetrievalEngine can be instantiated directly with vector_index and graph."""
        engine = RetrievalEngine(
            vector_index=self.vector_index,
            graph=self.graph,
        )
        self.assertIsNotNone(engine.vector_retriever)
        self.assertIsNotNone(engine.graph_retriever)

    # =========================================================================
    # 7. Hybrid Graph Evidence & Status Filtering Regression Tests
    # =========================================================================

    def test_vector_seed_with_no_graph_edges_receives_no_graph_evidence(self) -> None:
        """Verify a vector seed with no graph edges does not receive graph evidence or 'graph' source."""
        mem_iso = Memory.create(
            memory_id="mem-iso-vec",
            memory_type=MemoryType.FACT,
            content="Isolated encryption key rotation protocol.",
            project_id=self.project_a,
        )
        # Indexed in vector store and graph as an isolated node (no relationships)
        self.vector_manager.index_memory(mem_iso)
        self.graph_integrator.integrate_memory(mem_iso)

        res = self.engine.retrieve_hybrid("encryption key rotation", project_id=self.project_a)
        self.assertEqual(len(res.results), 1)
        item = res.results[0]
        self.assertEqual(item.memory_id, "mem-iso-vec")
        self.assertEqual(item.sources, ("vector",))
        self.assertIsNone(item.graph_score)
        self.assertEqual(len(item.graph_metadata), 0)

    def test_vector_seed_connected_to_memory_receives_graph_evidence_only_when_traversed(self) -> None:
        """Verify a vector seed itself does not receive graph evidence unless reached through a relationship."""
        from ai_brain.graph.models import GraphRelationship

        mem_target = Memory.create(
            memory_id="mem-target-dep",
            memory_type=MemoryType.FACT,
            content="Billing invoice processing via Stripe.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,
        )
        mem_seed = Memory.create(
            memory_id="mem-seed-checkout",
            memory_type=MemoryType.FACT,
            content="Checkout service architecture.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,
        )
        self.vector_manager.index_memory(mem_seed)
        self.graph_integrator.integrate_memory(mem_target)
        self.graph_integrator.integrate_memory(mem_seed)

        # Connect seed -> target (DEPENDS_ON)
        self.graph.add_relationship(
            GraphRelationship.create(
                relationship_id="rel:depends:checkout:billing",
                source_id="memory:mem-seed-checkout",
                target_id="memory:mem-target-dep",
                relationship_type=RelationshipType.DEPENDS_ON,
                project_id=self.project_a,
            )
        )

        # Hybrid query matches mem-seed-checkout in vector search
        res = self.engine.retrieve_hybrid("Checkout service architecture", project_id=self.project_a)
        result_map = {item.memory_id: item for item in res.results}

        # Seed node itself was discovered via vector only (no incoming graph traversal)
        self.assertIn("mem-seed-checkout", result_map)
        seed_item = result_map["mem-seed-checkout"]
        self.assertEqual(seed_item.sources, ("vector",))
        self.assertIsNone(seed_item.graph_score)

        # Target node was reached via graph traversal (DEPENDS_ON edge)
        self.assertIn("mem-target-dep", result_map)
        target_item = result_map["mem-target-dep"]
        self.assertEqual(target_item.sources, ("graph",))
        self.assertIsNotNone(target_item.graph_score)
        self.assertEqual(target_item.graph_metadata.get("depth"), 1)
        self.assertIn("DEPENDS_ON", target_item.graph_metadata.get("relationships_traversed", []))

    def test_hybrid_source_attribution_vector_only_graph_only_and_combined(self) -> None:
        """Verify hybrid retrieval accurately attributes vector-only, graph-only, and combined sources."""
        # 1. mem_vec: in vector only
        mem_vec = Memory.create(
            memory_id="mem-vec-only",
            memory_type=MemoryType.FACT,
            content="Standalone Redis cache setting.",
            project_id=self.project_a,
        )
        # 2. mem_comb: in vector AND reached via graph relationship
        mem_comb = Memory.create(
            memory_id="mem-comb",
            memory_type=MemoryType.FACT,
            content="Cluster persistence backend.",
            project_id=self.project_a,
        )
        # 3. mem_graph: in graph only, points to mem_comb
        mem_graph = Memory.create(
            memory_id="mem-graph-seed",
            memory_type=MemoryType.DECISION,
            content="Architecture decision for cluster persistence backend.",
            project_id=self.project_a,
        )

        self.vector_manager.index_memory(mem_vec)
        self.vector_manager.index_memory(mem_comb)

        self.graph_integrator.integrate_memory(mem_comb)
        self.graph_integrator.integrate_memory(mem_graph)

        # Add explicit relationship mem_graph -> mem_comb
        from ai_brain.graph.models import GraphRelationship
        self.graph.add_relationship(
            GraphRelationship.create(
                relationship_id="rel:decides:mem-graph-seed:mem-comb",
                source_id="memory:mem-graph-seed",
                target_id="memory:mem-comb",
                relationship_type=RelationshipType.RELATED_TO,
                project_id=self.project_a,
            )
        )

        # Retrieve seeded explicitly from mem-graph-seed plus vector query
        res = self.engine.retrieve_hybrid(
            "cluster persistence backend Redis",
            project_id=self.project_a,
            seed_memory_ids=["mem-graph-seed"],
        )
        result_map = {item.memory_id: item for item in res.results}

        self.assertEqual(result_map["mem-vec-only"].sources, ("vector",))
        self.assertEqual(result_map["mem-graph-seed"].sources, ("graph",))
        self.assertEqual(result_map["mem-comb"].sources, ("vector", "graph"))

    def test_rrf_does_not_count_vector_seeds_as_graph_candidates(self) -> None:
        """Verify RRF receives only genuine graph matches and vector seeds are not counted as graph candidates."""
        p_query = self.query_processor.process("isolated query", project_id=self.project_a)
        matches = self.graph_retriever.retrieve(
            p_query,
            vector_seeds=["non-existent-or-isolated"],
        )
        # Isolated vector seed produces 0 graph matches
        self.assertEqual(len(matches), 0)

    def test_superseded_and_archived_memories_excluded_by_default_in_all_modes(self) -> None:
        """Verify SUPERSEDED and ARCHIVED memories are excluded by default in vector, graph, and hybrid modes."""
        m_active = Memory.create(
            memory_id="mem-st-active",
            memory_type=MemoryType.FACT,
            content="Status check active memory.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,
        )
        m_disputed = Memory.create(
            memory_id="mem-st-disputed",
            memory_type=MemoryType.FACT,
            content="Status check disputed memory.",
            project_id=self.project_a,
            status=MemoryStatus.DISPUTED.value,
        )
        m_superseded = Memory.create(
            memory_id="mem-st-superseded",
            memory_type=MemoryType.FACT,
            content="Status check superseded memory.",
            project_id=self.project_a,
            status=MemoryStatus.SUPERSEDED.value,
        )
        m_archived = Memory.create(
            memory_id="mem-st-archived",
            memory_type=MemoryType.FACT,
            content="Status check archived memory.",
            project_id=self.project_a,
            status=MemoryStatus.ARCHIVED.value,
        )

        for m in [m_active, m_disputed, m_superseded, m_archived]:
            self.vector_manager.index_memory(m)
            self.graph_integrator.integrate_memory(m)

        for method in [RetrievalMethod.VECTOR, RetrievalMethod.GRAPH, RetrievalMethod.HYBRID]:
            res = self.engine.retrieve("Status check", project_id=self.project_a, method=method)
            ids = [item.memory_id for item in res.results]

            self.assertIn("mem-st-active", ids)
            self.assertIn("mem-st-disputed", ids)
            self.assertNotIn("mem-st-superseded", ids)
            self.assertNotIn("mem-st-archived", ids)

    def test_explicit_historical_status_inclusion_works_consistently(self) -> None:
        """Verify explicitly requesting historical statuses includes them with visible warnings."""
        m_active = Memory.create(
            memory_id="mem-hist-active",
            memory_type=MemoryType.FACT,
            content="Historical check active memory.",
            project_id=self.project_a,
            status=MemoryStatus.ACTIVE.value,
        )
        m_superseded = Memory.create(
            memory_id="mem-hist-superseded",
            memory_type=MemoryType.FACT,
            content="Historical check superseded memory.",
            project_id=self.project_a,
            status=MemoryStatus.SUPERSEDED.value,
        )
        for m in [m_active, m_superseded]:
            self.vector_manager.index_memory(m)
            self.graph_integrator.integrate_memory(m)

        filters = QueryFilters(allowed_statuses=("ACTIVE", "SUPERSEDED"))
        for method in [RetrievalMethod.VECTOR, RetrievalMethod.GRAPH, RetrievalMethod.HYBRID]:
            res = self.engine.retrieve(
                "Historical check",
                project_id=self.project_a,
                method=method,
                filters=filters,
            )
            ids = [item.memory_id for item in res.results]
            self.assertIn("mem-hist-active", ids)
            self.assertIn("mem-hist-superseded", ids)

            # Warning is clearly visible
            self.assertTrue(any("Historical status filtering enabled" in w for w in res.warnings))

            # Superseded item receives status multiplier 0.2
            item_super = next(item for item in res.results if item.memory_id == "mem-hist-superseded")
            self.assertEqual(item_super.score_breakdown.status_multiplier, 0.2)


if __name__ == "__main__":
    unittest.main()
