"""
Comprehensive test suite for the Context Builder subsystem.
"""

from datetime import datetime, timezone
import json
import unittest

from ai_brain.context import (
    ContextBuilder,
    ContextPackage,
    ContextProjectIsolationError,
    DeterministicTokenizer,
    MemoryPrioritizer,
    MemorySelector,
    PriorityBreakdown,
    SelectionCriteria,
    TokenBudgetExceededError,
    TokenizerError,
)
from ai_brain.retrieval.models import RetrievalItem, RetrievalResponse, ScoreBreakdown


def _create_test_retrieval_item(
    *,
    memory_id: str,
    content: str,
    project_id: str = "proj-alpha",
    memory_type: str = "FACT",
    status: str = "ACTIVE",
    score: float = 0.85,
    root_memory_id: str = None,
    version: int = 1,
    is_disputed: bool = False,
    source_agent_ids: tuple = ("agent-claude",),
    source_event_ids: tuple = ("ev-100",),
    importance: float = 0.8,
    confidence: float = 0.95,
    graph_rels: tuple = (),
) -> RetrievalItem:
    """Helper to construct RetrievalItem fixture."""
    sb = ScoreBreakdown(
        base_score=score,
        vector_score=score,
        graph_score=None,
        importance_factor=0.8 + 0.2 * importance,
        confidence_factor=0.8 + 0.2 * confidence,
        recency_factor=1.0,
        status_multiplier=1.0 if status == "ACTIVE" and not is_disputed else 0.7,
        provenance_bonus=0.0,
        final_score=score,
    )
    meta = {
        "importance": importance,
        "confidence": confidence,
    }
    graph_meta = {}
    if graph_rels:
        graph_meta["relationships_traversed"] = list(graph_rels)

    return RetrievalItem(
        memory_id=memory_id,
        root_memory_id=root_memory_id or memory_id,
        version=version,
        project_id=project_id,
        content=content,
        memory_type=memory_type,
        status=status,
        score=score,
        score_breakdown=sb,
        vector_score=score,
        graph_score=None,
        graph_metadata=graph_meta,
        sources=("vector",),
        source_event_ids=source_event_ids,
        source_agent_ids=source_agent_ids,
        is_disputed=is_disputed,
        created_at=datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc),
        metadata=meta,
    )


class TestTokenizer(unittest.TestCase):
    """Tests for Tokenizer abstraction and deterministic offline implementation."""

    def setUp(self) -> None:
        self.tokenizer = DeterministicTokenizer()

    def test_deterministic_token_counting(self) -> None:
        """Token count is non-negative and identical for repeated calls."""
        text = "PostgreSQL is selected as the primary relational database."
        count1 = self.tokenizer.count_tokens(text)
        count2 = self.tokenizer.count_tokens(text)
        self.assertEqual(count1, count2)
        self.assertGreater(count1, 5)

    def test_empty_and_whitespace_handling(self) -> None:
        """Empty strings and purely blank inputs produce 0 tokens."""
        self.assertEqual(self.tokenizer.count_tokens(""), 0)
        self.assertEqual(self.tokenizer.count_tokens("   "), 0)

    def test_subword_heuristic_on_long_tokens(self) -> None:
        """Long continuous identifiers/words generate subword tokens."""
        short_word = "data"
        long_word = "supercalifragilisticexpialidocious"
        self.assertGreater(
            self.tokenizer.count_tokens(long_word),
            self.tokenizer.count_tokens(short_word),
        )

    def test_invalid_chars_per_token_rejected(self) -> None:
        """Invalid chars_per_token raises TokenizerError."""
        with self.assertRaises(TokenizerError):
            DeterministicTokenizer(chars_per_token=0)


class TestMemorySelection(unittest.TestCase):
    """Tests for relevant memory selection, filtering, and project isolation."""

    def setUp(self) -> None:
        self.selector = MemorySelector()

    def test_project_isolation_enforcement(self) -> None:
        """Memories from a different project trigger ContextProjectIsolationError."""
        foreign_item = _create_test_retrieval_item(
            memory_id="mem-foreign",
            content="Foreign database fact",
            project_id="proj-beta",
        )
        with self.assertRaises(ContextProjectIsolationError):
            self.selector.select([foreign_item], project_id="proj-alpha")

    def test_retrieval_response_project_mismatch(self) -> None:
        """RetrievalResponse with mismatched project_id raises ContextProjectIsolationError."""
        response = RetrievalResponse(
            query="database",
            normalized_query="database",
            project_id="proj-beta",
            method="vector",
            results=(),
        )
        with self.assertRaises(ContextProjectIsolationError):
            self.selector.select(response, project_id="proj-alpha")

    def test_version_deduplication(self) -> None:
        """Retains the latest version of a memory and marks older versions as duplicate_version."""
        v1 = _create_test_retrieval_item(
            memory_id="mem-db-v1",
            content="Project uses PostgreSQL 14",
            root_memory_id="root-db",
            version=1,
            score=0.80,
            status="SUPERSEDED",
        )
        v2 = _create_test_retrieval_item(
            memory_id="mem-db-v2",
            content="Project uses PostgreSQL 16",
            root_memory_id="root-db",
            version=2,
            score=0.92,
            status="ACTIVE",
        )

        criteria = SelectionCriteria(include_superseded=True, deduplicate_versions=True)
        sel = MemorySelector(criteria=criteria)
        selected, excluded = sel.select([v1, v2], project_id="proj-alpha")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].memory_id, "mem-db-v2")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].memory_id, "mem-db-v1")
        self.assertEqual(excluded[0].reason, "duplicate_version")

    def test_superseded_and_archived_excluded_by_default(self) -> None:
        """Superseded and archived memories are excluded by default."""
        sup = _create_test_retrieval_item(
            memory_id="mem-sup",
            content="Old fact",
            status="SUPERSEDED",
        )
        arch = _create_test_retrieval_item(
            memory_id="mem-arch",
            content="Archived task",
            status="ARCHIVED",
        )
        active = _create_test_retrieval_item(
            memory_id="mem-act",
            content="Active fact",
            status="ACTIVE",
        )

        selected, excluded = self.selector.select([sup, arch, active], project_id="proj-alpha")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].memory_id, "mem-act")
        excluded_reasons = {ex.memory_id: ex.reason for ex in excluded}
        self.assertEqual(excluded_reasons["mem-sup"], "excluded_status")
        self.assertEqual(excluded_reasons["mem-arch"], "excluded_status")

    def test_disputed_memories_preserved(self) -> None:
        """Disputed memories are retained and marked as disputed."""
        disp = _create_test_retrieval_item(
            memory_id="mem-disp",
            content="Contradictory schema design",
            status="DISPUTED",
            is_disputed=True,
        )
        selected, excluded = self.selector.select([disp], project_id="proj-alpha")
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0].is_disputed)

    def test_min_relevance_score_filter(self) -> None:
        """Candidates below min_relevance_score threshold are excluded."""
        low_item = _create_test_retrieval_item(
            memory_id="mem-low",
            content="Irrelevant trivia",
            score=0.25,
        )
        high_item = _create_test_retrieval_item(
            memory_id="mem-high",
            content="Core architecture",
            score=0.88,
        )

        criteria = SelectionCriteria(min_relevance_score=0.50)
        sel = MemorySelector(criteria=criteria)
        selected, excluded = sel.select([low_item, high_item], project_id="proj-alpha")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].memory_id, "mem-high")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].memory_id, "mem-low")
        self.assertEqual(excluded[0].reason, "low_relevance")


class TestContextPrioritization(unittest.TestCase):
    """Tests for multi-factor deterministic prioritization and scoring."""

    def setUp(self) -> None:
        self.prioritizer = MemoryPrioritizer()

    def test_relevance_primary_gate_zero_score(self) -> None:
        """A memory with 0.0 relevance yields 0.0 priority even if importance is 1.0."""
        item = _create_test_retrieval_item(
            memory_id="mem-irrelevant",
            content="Irrelevant highly important fact",
            score=0.0,
            importance=1.0,
            confidence=1.0,
        )
        priority, breakdown = self.prioritizer.compute_priority(item)
        self.assertEqual(priority, 0.0)
        self.assertEqual(breakdown.final_priority, 0.0)
        self.assertEqual(breakdown.relevance_score, 0.0)

    def test_decisions_and_tasks_boosted(self) -> None:
        """Decisions and active tasks receive type bonuses over standard facts."""
        fact = _create_test_retrieval_item(
            memory_id="mem-fact",
            content="Database is Postgres",
            memory_type="FACT",
            score=0.75,
            importance=0.5,
            confidence=1.0,
        )
        decision = _create_test_retrieval_item(
            memory_id="mem-dec",
            content="Migrate database to Postgres",
            memory_type="DECISION",
            score=0.75,
            importance=0.5,
            confidence=1.0,
        )
        task = _create_test_retrieval_item(
            memory_id="mem-task",
            content="Run database migrations",
            memory_type="TASK",
            score=0.75,
            importance=0.5,
            confidence=1.0,
        )

        p_fact, b_fact = self.prioritizer.compute_priority(fact)
        p_dec, b_dec = self.prioritizer.compute_priority(decision)
        p_task, b_task = self.prioritizer.compute_priority(task)

        self.assertGreater(p_dec, p_fact)
        self.assertGreater(p_task, p_fact)
        self.assertEqual(b_dec.type_bonus, 1.15)
        self.assertEqual(b_task.type_bonus, 1.10)

    def test_disputed_memories_penalized(self) -> None:
        """Disputed memories receive status discount so they rank below solid facts."""
        solid = _create_test_retrieval_item(
            memory_id="mem-solid",
            content="Uses Python 3.11",
            status="ACTIVE",
            score=0.80,
            importance=0.8,
            confidence=1.0,
        )
        disputed = _create_test_retrieval_item(
            memory_id="mem-disp",
            content="Uses Node 18",
            status="DISPUTED",
            is_disputed=True,
            score=0.80,
            importance=0.8,
            confidence=1.0,
        )

        p_solid, b_solid = self.prioritizer.compute_priority(solid)
        p_disp, b_disp = self.prioritizer.compute_priority(disputed)

        self.assertGreater(p_solid, p_disp)
        self.assertEqual(b_disp.status_multiplier, 0.65)

    def test_inspectable_priority_breakdown(self) -> None:
        """PriorityBreakdown serializes cleanly and exposes transparent components."""
        item = _create_test_retrieval_item(
            memory_id="mem-inspect",
            content="Architecture spec",
            score=0.82,
            importance=0.7,
            confidence=0.9,
        )
        _, breakdown = self.prioritizer.compute_priority(item)
        d = breakdown.to_dict()
        self.assertIn("relevance_score", d)
        self.assertIn("importance_score", d)
        self.assertIn("confidence_score", d)
        self.assertIn("type_bonus", d)
        self.assertIn("recency_factor", d)
        self.assertIn("status_multiplier", d)
        self.assertIn("final_priority", d)


class TestContextCompression(unittest.TestCase):
    """Tests for semantic compression, claim merging, and contradiction preservation."""

    def setUp(self) -> None:
        self.builder = ContextBuilder()

    def test_whitespace_normalization(self) -> None:
        """Excessive whitespace and tabs are cleaned up in context."""
        item = _create_test_retrieval_item(
            memory_id="mem-ws",
            content="Project   uses   Python    3.11.\n\n\n\nRuns on Docker.   ",
        )
        pkg = self.builder.build_context([item], project_id="proj-alpha")
        self.assertEqual(len(pkg.items), 1)
        self.assertEqual(pkg.items[0].content, "Project uses Python 3.11.\n\nRuns on Docker.")

    def test_compatible_claim_merging_consolidates_provenance(self) -> None:
        """Compatible identical facts from different agents are merged and provenance consolidated."""
        claim1 = _create_test_retrieval_item(
            memory_id="mem-c1",
            content="The project uses PostgreSQL 16.",
            source_agent_ids=("agent-gemini",),
            source_event_ids=("ev-1",),
            score=0.85,
        )
        claim2 = _create_test_retrieval_item(
            memory_id="mem-c2",
            content="The project uses PostgreSQL 16.",
            source_agent_ids=("agent-claude",),
            source_event_ids=("ev-2",),
            score=0.88,
        )

        pkg = self.builder.build_context([claim1, claim2], project_id="proj-alpha")
        self.assertEqual(len(pkg.items), 1)
        merged_item = pkg.items[0]
        self.assertEqual(set(merged_item.source_agent_ids), {"agent-gemini", "agent-claude"})
        self.assertEqual(set(merged_event_ids := merged_item.source_event_ids), {"ev-1", "ev-2"})

    def test_contradictory_and_disputed_claims_never_merged(self) -> None:
        """Contradictory and disputed claims remain distinct."""
        claim1 = _create_test_retrieval_item(
            memory_id="mem-mysql",
            content="The database is MySQL",
            status="DISPUTED",
            is_disputed=True,
            score=0.75,
        )
        claim2 = _create_test_retrieval_item(
            memory_id="mem-postgres",
            content="The database is PostgreSQL",
            status="DISPUTED",
            is_disputed=True,
            score=0.75,
        )

        pkg = self.builder.build_context([claim1, claim2], project_id="proj-alpha")
        self.assertEqual(len(pkg.items), 2)
        memory_ids = {it.memory_id for it in pkg.items}
        self.assertIn("mem-mysql", memory_ids)
        self.assertIn("mem-postgres", memory_ids)

    def test_original_retrieval_items_not_mutated(self) -> None:
        """Compression and packing must never mutate original objects."""
        original_content = "Raw   content   with    spaces"
        item = _create_test_retrieval_item(
            memory_id="mem-orig",
            content=original_content,
        )
        self.builder.build_context([item], project_id="proj-alpha")
        self.assertEqual(item.content, original_content)


class TestTokenBudgetManagement(unittest.TestCase):
    """Tests for token budget allocation, greedy packing, and accounting."""

    def setUp(self) -> None:
        self.builder = ContextBuilder(default_max_tokens=150)

    def test_budget_enforced_greedily(self) -> None:
        """Items are packed greedily until the token budget is exhausted; remainder excluded."""
        items = [
            _create_test_retrieval_item(
                memory_id=f"mem-{i}",
                content=f"Important factual statement number {i} detailing system configuration and architecture.",
                score=0.90 - i * 0.05,
            )
            for i in range(10)
        ]

        pkg = self.builder.build_context(items, project_id="proj-alpha", max_tokens=250)
        self.assertLessEqual(pkg.tokens_used, 250)
        self.assertGreater(len(pkg.items), 0)
        self.assertGreater(len(pkg.excluded_items), 0)

        budget_excluded = [ex for ex in pkg.excluded_items if ex.reason == "token_budget_exceeded"]
        self.assertGreater(len(budget_excluded), 0)

    def test_comprehensive_token_accounting(self) -> None:
        """Token accounting correctly reports used, remaining, and budget."""
        item = _create_test_retrieval_item(
            memory_id="mem-acc",
            content="Short fact.",
            score=0.85,
        )
        pkg = self.builder.build_context([item], project_id="proj-alpha", max_tokens=200)
        self.assertEqual(pkg.token_budget, 200)
        self.assertEqual(pkg.tokens_used + pkg.remaining_tokens, 200)
        self.assertGreater(pkg.tokens_used, 0)

    def test_strict_budget_overflow_raises_error(self) -> None:
        """Strict budget raises TokenBudgetExceededError if base header cannot fit."""
        item = _create_test_retrieval_item(
            memory_id="mem-overflow",
            content="Fact",
        )
        with self.assertRaises(TokenBudgetExceededError):
            self.builder.build_context(
                [item],
                project_id="proj-alpha",
                max_tokens=5,  # Too small even for baseline header
                strict_budget=True,
            )


class TestAgentReadyContextPackage(unittest.TestCase):
    """Tests for structured sections, markdown formatting, and serialization."""

    def setUp(self) -> None:
        self.builder = ContextBuilder(default_max_tokens=1000)

    def test_context_sections_and_helpers(self) -> None:
        """Package correctly groups decisions, tasks, facts, and provides helper lookups."""
        dec = _create_test_retrieval_item(
            memory_id="dec-1",
            content="Adopt GraphQL for client API",
            memory_type="DECISION",
            score=0.95,
        )
        task = _create_test_retrieval_item(
            memory_id="task-1",
            content="Write GraphQL schema definitions",
            memory_type="TASK",
            score=0.90,
        )
        fact = _create_test_retrieval_item(
            memory_id="fact-1",
            content="Server is implemented in TypeScript",
            memory_type="FACT",
            score=0.85,
        )

        pkg = self.builder.build_context([dec, task, fact], project_id="proj-alpha", query="api architecture")

        self.assertEqual(len(pkg.get_active_decisions()), 1)
        self.assertEqual(len(pkg.get_active_tasks()), 1)
        self.assertFalse(pkg.has_disputes)

        sec_dec = pkg.get_section("Active Decisions & Constraints")
        self.assertIsNotNone(sec_dec)
        self.assertEqual(len(sec_dec.items), 1)

    def test_disputed_memories_prominently_warned_in_markdown(self) -> None:
        """Disputed items include prominent warning banners in rendered markdown."""
        disp = _create_test_retrieval_item(
            memory_id="disp-1",
            content="API runs on port 8080 according to Agent A, but 3000 according to Agent B",
            status="DISPUTED",
            is_disputed=True,
            score=0.80,
        )

        pkg = self.builder.build_context([disp], project_id="proj-alpha")
        self.assertTrue(pkg.has_disputes)

        md = pkg.to_markdown()
        self.assertIn("⚠️", md)
        self.assertIn("DISPUTED", md)
        self.assertIn("Do NOT treat them as established ground truth", md)

    def test_provenance_attribution_summary_included(self) -> None:
        """Markdown output includes provenance attribution table."""
        item = _create_test_retrieval_item(
            memory_id="fact-prov",
            content="PostgreSQL database hosted on AWS RDS",
            source_agent_ids=("agent-gemini", "agent-codex"),
            source_event_ids=("ev-999",),
        )
        pkg = self.builder.build_context([item], project_id="proj-alpha")
        md = pkg.to_markdown()
        self.assertIn("## Provenance & Attribution Summary", md)
        self.assertIn("agent-gemini, agent-codex", md)
        self.assertIn("ev-999", md)

    def test_package_serialization_roundtrip(self) -> None:
        """Package serializes to dict and json, and reconstructs identically."""
        item = _create_test_retrieval_item(
            memory_id="fact-ser",
            content="Cache layer uses Redis Cluster",
            score=0.88,
        )
        pkg = self.builder.build_context([item], project_id="proj-alpha", query="caching")

        d = pkg.to_dict()
        reconstructed = ContextPackage.from_dict(d)
        self.assertEqual(reconstructed.project_id, pkg.project_id)
        self.assertEqual(reconstructed.query, pkg.query)
        self.assertEqual(reconstructed.tokens_used, pkg.tokens_used)
        self.assertEqual(len(reconstructed.items), len(pkg.items))
        self.assertEqual(reconstructed.items[0].memory_id, "fact-ser")

        json_str = pkg.to_json()
        reconstructed_json = ContextPackage.from_json(json_str)
        self.assertEqual(reconstructed_json.tokens_used, pkg.tokens_used)


class TestDisputedMemoryAndBudgetPolicy(unittest.TestCase):
    """
    Regression test suite resolving the conflict between disputed-memory preservation
    and strict token-budget enforcement.
    """

    def setUp(self) -> None:
        self.tokenizer = DeterministicTokenizer()
        self.builder = ContextBuilder(tokenizer=self.tokenizer)

    def test_1_disputed_memory_included_when_sufficient_budget(self) -> None:
        """A disputed memory is included in context when sufficient token budget exists."""
        fact = _create_test_retrieval_item(
            memory_id="fact-1",
            content="Primary database is PostgreSQL 16.",
            score=0.90,
        )
        disp = _create_test_retrieval_item(
            memory_id="disp-1",
            content="Cache is Redis on 6379, but Agent B reports Memcached.",
            status="DISPUTED",
            is_disputed=True,
            score=0.85,
        )

        pkg = self.builder.build_context([fact, disp], project_id="proj-alpha", max_tokens=600)
        self.assertTrue(pkg.has_disputes)
        self.assertFalse(pkg.has_omitted_disputes)
        included_ids = {it.memory_id for it in pkg.items}
        self.assertIn("disp-1", included_ids)
        self.assertIn("fact-1", included_ids)

        md = pkg.to_markdown()
        self.assertIn("## Disputed Knowledge", md)
        self.assertIn("disp-1", md)
        self.assertNotIn("DISPUTED KNOWLEDGE OMITTED", md)

    def test_2_disputed_memory_excluded_when_budget_insufficient(self) -> None:
        """A disputed memory is excluded when the token budget is insufficient to hold it."""
        fact = _create_test_retrieval_item(
            memory_id="fact-1",
            content="The system persistence layer runs PostgreSQL 16 with Citus sharding and automated backups.",
            score=0.95,
        )
        disp = _create_test_retrieval_item(
            memory_id="disp-1",
            content="Unverified conflicting statement regarding memory cache clustering configuration.",
            status="DISPUTED",
            is_disputed=True,
            score=0.70,
        )

        # Budget large enough for fact + header + omission notice, but too small for both items
        pkg = self.builder.build_context([fact, disp], project_id="proj-alpha", max_tokens=250)
        included_ids = {it.memory_id for it in pkg.items}
        self.assertIn("fact-1", included_ids)
        self.assertNotIn("disp-1", included_ids)
        self.assertTrue(pkg.has_omitted_disputes)

    def test_3_excluded_disputed_memory_retains_identity_and_explicit_reason(self) -> None:
        """Excluded disputed memories appear in excluded_items with identity, disputed status, and explicit reason."""
        fact = _create_test_retrieval_item(
            memory_id="fact-1",
            content="The system persistence layer runs PostgreSQL 16 with Citus sharding and automated backups.",
            score=0.95,
        )
        disp = _create_test_retrieval_item(
            memory_id="disp-1",
            content="Unverified conflicting statement regarding memory cache clustering configuration.",
            status="DISPUTED",
            is_disputed=True,
            score=0.70,
        )

        pkg = self.builder.build_context([fact, disp], project_id="proj-alpha", max_tokens=250)
        excluded_map = {ex.memory_id: ex for ex in pkg.excluded_items}
        self.assertIn("disp-1", excluded_map)
        ex_disp = excluded_map["disp-1"]
        self.assertEqual(ex_disp.memory_id, "disp-1")
        self.assertTrue(ex_disp.is_disputed)
        self.assertEqual(ex_disp.status, "DISPUTED")
        self.assertEqual(ex_disp.reason, "token_budget_exceeded")
        self.assertGreater(ex_disp.token_count, 0)

    def test_4_final_context_signals_disputed_omission(self) -> None:
        """The final context signals when disputed information was omitted due to budget."""
        fact = _create_test_retrieval_item(
            memory_id="fact-1",
            content="The system persistence layer runs PostgreSQL 16 with Citus sharding and automated backups.",
            score=0.95,
        )
        disp = _create_test_retrieval_item(
            memory_id="disp-1",
            content="Unverified conflicting statement regarding memory cache clustering configuration.",
            status="DISPUTED",
            is_disputed=True,
            score=0.70,
        )

        pkg = self.builder.build_context([fact, disp], project_id="proj-alpha", max_tokens=250)
        md = pkg.to_markdown()
        txt = pkg.to_text()

        self.assertIn("DISPUTED KNOWLEDGE OMITTED", md)
        self.assertIn("disp-1", md)
        self.assertIn("omitted due to token budget constraints", txt)
        self.assertTrue(pkg.has_omitted_disputes)
        self.assertEqual(len(pkg.get_excluded_disputes()), 1)

    def test_5_disputed_claims_never_merged_into_established_claims(self) -> None:
        """Disputed claims are never merged into established facts during compression."""
        fact = _create_test_retrieval_item(
            memory_id="fact-1",
            content="The project uses Python 3.11 runtime.",
            status="ACTIVE",
            is_disputed=False,
            score=0.85,
        )
        disp = _create_test_retrieval_item(
            memory_id="disp-1",
            content="The project uses Python 3.11 runtime.",
            status="DISPUTED",
            is_disputed=True,
            score=0.85,
        )

        pkg = self.builder.build_context([fact, disp], project_id="proj-alpha", max_tokens=600)
        # Must retain 2 distinct items; disputed claim must NOT be swallowed into established fact
        self.assertEqual(len(pkg.items), 2)
        item_statuses = {it.memory_id: it.is_disputed for it in pkg.items}
        self.assertFalse(item_statuses["fact-1"])
        self.assertTrue(item_statuses["disp-1"])

    def test_6_contradictory_claims_remain_separate(self) -> None:
        """Contradictory claims remain separate and are never merged."""
        claim_a = _create_test_retrieval_item(
            memory_id="mem-a",
            content="The database is PostgreSQL.",
            score=0.85,
        )
        claim_b = _create_test_retrieval_item(
            memory_id="mem-b",
            content="The database is MySQL.",
            score=0.85,
        )

        pkg = self.builder.build_context([claim_a, claim_b], project_id="proj-alpha", max_tokens=600)
        self.assertEqual(len(pkg.items), 2)
        contents = {it.content for it in pkg.items}
        self.assertIn("The database is PostgreSQL.", contents)
        self.assertIn("The database is MySQL.", contents)

    def test_7_compression_preserves_provenance(self) -> None:
        """Semantic compression of identical claims preserves and unifies provenance."""
        claim_1 = _create_test_retrieval_item(
            memory_id="mem-1",
            content="The system runs on Debian Linux.",
            source_agent_ids=("agent-gemini",),
            source_event_ids=("ev-01",),
            score=0.88,
        )
        claim_2 = _create_test_retrieval_item(
            memory_id="mem-2",
            content="The system runs on Debian Linux.",
            source_agent_ids=("agent-claude",),
            source_event_ids=("ev-02",),
            score=0.86,
        )

        pkg = self.builder.build_context([claim_1, claim_2], project_id="proj-alpha", max_tokens=600)
        self.assertEqual(len(pkg.items), 1)
        merged = pkg.items[0]
        self.assertEqual(set(merged.source_agent_ids), {"agent-gemini", "agent-claude"})
        self.assertEqual(set(merged.source_event_ids), {"ev-01", "ev-02"})

    def test_8_rendered_output_strictly_within_token_budget(self) -> None:
        """Final rendered markdown, including warnings and omission notices, strictly respects budget."""
        items = [
            _create_test_retrieval_item(
                memory_id=f"item-{i}",
                content=f"Detailed specification regarding system component {i} in the architecture.",
                score=0.95 - (i * 0.05),
                is_disputed=(i == 3),
                status="DISPUTED" if i == 3 else "ACTIVE",
            )
            for i in range(8)
        ]

        budgets = [160, 180, 220, 300, 500]
        for b in budgets:
            pkg = self.builder.build_context(items, project_id="proj-alpha", max_tokens=b)
            rendered_tokens = self.tokenizer.count_tokens(pkg.to_markdown())
            self.assertLessEqual(
                rendered_tokens,
                b,
                f"Rendered markdown ({rendered_tokens} tokens) exceeded budget limit ({b} tokens)",
            )
            self.assertEqual(rendered_tokens, pkg.tokens_used)
            self.assertEqual(pkg.token_budget, b)
            self.assertEqual(pkg.tokens_used + pkg.remaining_tokens, b)

    def test_9_original_memory_and_retrieval_records_remain_unchanged(self) -> None:
        """Original retrieval items and memory records are never mutated by context building."""
        raw_content = "Raw   content   with    extra    spacing.  "
        raw_meta = {"custom_field": "immutable_value", "importance": 0.85, "confidence": 0.9}
        raw_sources = ("agent-orig",)
        raw_events = ("ev-orig-1",)

        original_item = _create_test_retrieval_item(
            memory_id="mem-immutable",
            content=raw_content,
            status="DISPUTED",
            is_disputed=True,
            score=0.88,
            source_agent_ids=raw_sources,
            source_event_ids=raw_events,
        )
        object.__setattr__(original_item, "metadata", raw_meta)

        self.builder.build_context([original_item], project_id="proj-alpha", max_tokens=400)

        # Verify all attributes on original object remain unchanged
        self.assertEqual(original_item.content, raw_content)
        self.assertEqual(original_item.status, "DISPUTED")
        self.assertTrue(original_item.is_disputed)
        self.assertEqual(original_item.source_agent_ids, raw_sources)
        self.assertEqual(original_item.source_event_ids, raw_events)
        self.assertEqual(original_item.metadata["custom_field"], "immutable_value")


if __name__ == "__main__":
    unittest.main()
