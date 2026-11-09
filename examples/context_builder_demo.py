"""
Demonstration of the Context Builder subsystem in AI Brain.

Shows:
1. Converting retrieval results into prioritized, agent-ready context packages.
2. Token budget enforcement and greedy knapsack packing.
3. Transparent priority breakdowns and relevance gating.
4. Safe claim compression and provenance consolidation.
5. High-visibility handling of disputed knowledge.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ai_brain.context import ContextBuilder
from ai_brain.retrieval.models import RetrievalItem, RetrievalResponse, ScoreBreakdown


def main() -> None:
    print("=" * 70)
    print("AI Brain — Context Builder Subsystem Demonstration")
    print("=" * 70)

    project_id = "proj-ecommerce"

    # 1. Synthesize realistic retrieval items from multiple agents
    now = datetime.now(timezone.utc)

    items = [
        RetrievalItem(
            memory_id="dec-db",
            root_memory_id="root-dec-db",
            version=2,
            project_id=project_id,
            content="Use PostgreSQL 16 as the primary database with Citus extension for sharding.",
            memory_type="DECISION",
            status="ACTIVE",
            score=0.96,
            score_breakdown=ScoreBreakdown(
                base_score=0.96,
                vector_score=0.95,
                graph_score=0.98,
                importance_factor=1.0,
                confidence_factor=1.0,
                recency_factor=1.0,
                status_multiplier=1.0,
                provenance_bonus=0.05,
                final_score=0.96,
            ),
            sources=("vector", "graph"),
            source_agent_ids=("agent-architect",),
            source_event_ids=("ev-arch-01",),
            created_at=now,
            metadata={"importance": 0.9, "confidence": 1.0},
        ),
        RetrievalItem(
            memory_id="task-migration",
            root_memory_id="root-task-migration",
            version=1,
            project_id=project_id,
            content="Execute Flyway migrations for schema v2.4 before starting payment service.",
            memory_type="TASK",
            status="ACTIVE",
            score=0.91,
            score_breakdown=ScoreBreakdown(
                base_score=0.91,
                vector_score=0.91,
                graph_score=None,
                importance_factor=0.95,
                confidence_factor=1.0,
                recency_factor=1.0,
                status_multiplier=1.0,
                provenance_bonus=0.0,
                final_score=0.91,
            ),
            sources=("vector",),
            source_agent_ids=("agent-devops",),
            source_event_ids=("ev-ops-02",),
            created_at=now,
            metadata={"importance": 0.85, "confidence": 0.95},
        ),
        RetrievalItem(
            memory_id="fact-auth-1",
            root_memory_id="root-fact-auth-1",
            version=1,
            project_id=project_id,
            content="User authentication is handled via OAuth 2.0 with JWT tokens.",
            memory_type="FACT",
            status="ACTIVE",
            score=0.88,
            score_breakdown=ScoreBreakdown(
                base_score=0.88,
                vector_score=0.88,
                graph_score=None,
                importance_factor=0.8,
                confidence_factor=1.0,
                recency_factor=1.0,
                status_multiplier=1.0,
                provenance_bonus=0.0,
                final_score=0.88,
            ),
            sources=("vector",),
            source_agent_ids=("agent-backend",),
            source_event_ids=("ev-be-03",),
            created_at=now,
            metadata={"importance": 0.7, "confidence": 1.0},
        ),
        RetrievalItem(
            memory_id="fact-auth-2",
            root_memory_id="root-fact-auth-2",
            version=1,
            project_id=project_id,
            content="User authentication is handled via OAuth 2.0 with JWT tokens.",
            memory_type="FACT",
            status="ACTIVE",
            score=0.86,
            score_breakdown=ScoreBreakdown(
                base_score=0.86,
                vector_score=0.86,
                graph_score=None,
                importance_factor=0.8,
                confidence_factor=1.0,
                recency_factor=1.0,
                status_multiplier=1.0,
                provenance_bonus=0.0,
                final_score=0.86,
            ),
            sources=("vector",),
            source_agent_ids=("agent-security",),
            source_event_ids=("ev-sec-04",),
            created_at=now,
            metadata={"importance": 0.7, "confidence": 1.0},
        ),
        RetrievalItem(
            memory_id="disp-cache",
            root_memory_id="root-disp-cache",
            version=1,
            project_id=project_id,
            content="Session cache uses Redis on port 6379, but Agent Codex states Memcached is deployed.",
            memory_type="FACT",
            status="DISPUTED",
            is_disputed=True,
            score=0.82,
            score_breakdown=ScoreBreakdown(
                base_score=0.82,
                vector_score=0.82,
                graph_score=None,
                importance_factor=0.8,
                confidence_factor=0.8,
                recency_factor=1.0,
                status_multiplier=0.7,
                provenance_bonus=0.0,
                final_score=0.82,
            ),
            sources=("vector",),
            source_agent_ids=("agent-gemini", "agent-codex"),
            source_event_ids=("ev-cache-05",),
            created_at=now,
            metadata={"importance": 0.6, "confidence": 0.5, "is_disputed": True},
        ),
    ]

    retrieval_response = RetrievalResponse(
        query="database migration and authentication architecture",
        normalized_query="database migration and authentication architecture",
        project_id=project_id,
        method="hybrid",
        results=tuple(items),
    )

    # 2. Build Agent Context with standard budget
    builder = ContextBuilder(default_max_tokens=600)
    package = builder.build_context(retrieval_response, project_id=project_id)

    print(f"\n1. Context Package Overview:")
    print(f"   Project ID:       {package.project_id}")
    print(f"   Query:            {package.query}")
    print(f"   Token Budget:     {package.token_budget}")
    print(f"   Tokens Used:      {package.tokens_used}")
    print(f"   Remaining Tokens: {package.remaining_tokens}")
    print(f"   Items Included:   {len(package.items)}")
    print(f"   Items Excluded:   {len(package.excluded_items)}")
    print(f"   Disputes Present: {package.has_disputes}")

    print("\n2. Sections Formatted:")
    for sec in package.sections:
        print(f"   - {sec.name} ({len(sec.items)} items, ~{sec.section_tokens} tokens)")

    print("\n3. Priority Breakdown of Top Item:")
    top_item = package.items[0]
    bd = top_item.priority_breakdown.to_dict()
    print(f"   Item ID:        {top_item.memory_id} ({top_item.memory_type})")
    print(f"   Final Priority: {bd['final_priority']}")
    print(f"   Components:     Relevance={bd['relevance_score']}, Importance={bd['importance_score']}, TypeBonus={bd['type_bonus']}, StatusMultiplier={bd['status_multiplier']}")

    print("\n4. Rendered Agent-Ready Markdown Prompt:")
    print("-" * 50)
    print(package.to_markdown())
    print("-" * 50)

    # 3. Demonstration of strict budget constraint and disputed omission signal
    print("\n5. Testing Constrained Budget (max_tokens=320, Disputed Omission Signal):")
    budgeted_pkg = builder.build_context(retrieval_response, project_id=project_id, max_tokens=320)
    print(f"   Tokens Used:        {budgeted_pkg.tokens_used} / {budgeted_pkg.token_budget}")
    print(f"   Items Retained:     {len(budgeted_pkg.items)} ({[it.memory_id for it in budgeted_pkg.items]})")
    print(f"   Items Dropped:      {len(budgeted_pkg.excluded_items)}")
    print(f"   Disputes Omitted:   {budgeted_pkg.has_omitted_disputes}")
    for ex in budgeted_pkg.excluded_items:
        disp_tag = " [DISPUTED]" if ex.is_disputed else ""
        print(f"     * Excluded: `{ex.memory_id}`{disp_tag} — Reason: {ex.reason} (tokens: {ex.token_count})")
    print("\n   Rendered Constrained Markdown (Note Disputed Omission Notice):")
    print("   " + "-" * 46)
    for line in budgeted_pkg.to_markdown().splitlines()[:12]:
        print("   " + line)
    print("   ...")

    print("\nContext Builder demonstration completed successfully!")


if __name__ == "__main__":
    main()
