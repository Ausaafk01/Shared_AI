"""
Benchmark dataset containing realistic multi-session scenarios for AI Brain evaluation.
Includes 10 structured scenario categories with explicit provenance, expected outcomes, and distractors.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from ai_brain.benchmarks.models import (
    BenchmarkScenario,
    ScenarioCategory,
    ScenarioDifficulty,
    ScenarioProvenance,
)
from ai_brain.events.models import Event, EventType


def _make_event(
    event_id: str,
    event_type: EventType,
    content: str,
    project_id: str,
    agent_id: str,
    session_id: str,
    timestamp: datetime,
    metadata: Optional[Dict[str, str]] = None,
) -> Event:
    type_val = event_type.value if hasattr(event_type, "value") else str(event_type)
    return Event(
        event_id=event_id,
        event_type=type_val,
        timestamp=timestamp,
        project_id=project_id,
        session_id=session_id,
        actor=agent_id,
        agent_id=agent_id,
        output={"text": content},
        metadata=metadata or {},
    )


class BenchmarkDataset:
    """Pre-built suite and loader for AI Brain evaluation scenarios."""

    def __init__(self) -> None:
        self._scenarios: Dict[str, BenchmarkScenario] = {}
        self._load_standard_scenarios()

    def add_scenario(self, scenario: BenchmarkScenario) -> None:
        self._scenarios[scenario.scenario_id] = scenario

    def get_scenario(self, scenario_id: str) -> Optional[BenchmarkScenario]:
        return self._scenarios.get(scenario_id)

    def list_scenarios(
        self,
        category: Optional[ScenarioCategory] = None,
        difficulty: Optional[ScenarioDifficulty] = None,
        provenance: Optional[ScenarioProvenance] = None,
    ) -> List[BenchmarkScenario]:
        results = list(self._scenarios.values())
        if category:
            results = [s for s in results if s.category == category]
        if difficulty:
            results = [s for s in results if s.difficulty == difficulty]
        if provenance:
            results = [s for s in results if s.provenance == provenance]
        return results

    def _load_standard_scenarios(self) -> None:
        base_time = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)

        # ---------------------------------------------------------------------
        # 1. TASK_CONTINUATION: Continuing a multi-step service implementation
        # ---------------------------------------------------------------------
        p1 = "proj-bench-auth"
        s1_events = (
            _make_event("ev-tc-1", EventType.USER_MESSAGE, "Implement user authentication with JWT and bcrypt.", p1, "user", "sess-1", base_time),
            _make_event("ev-tc-2", EventType.DECISION, "DECISION: Use RS256 algorithm for JWT signing with RSA 4096-bit keypair.", p1, "agent-coder", "sess-1", base_time + timedelta(minutes=5)),
            _make_event("ev-tc-3", EventType.CODE_CHANGE, "Added user password hashing using bcrypt with work factor 12.", p1, "agent-coder", "sess-1", base_time + timedelta(minutes=15)),
            _make_event("ev-tc-4", EventType.TASK_COMPLETED, "Completed password hashing implementation.", p1, "agent-coder", "sess-1", base_time + timedelta(minutes=20)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_01_TASK_CONTINUATION",
                name="Continue Authentication Service Implementation",
                category=ScenarioCategory.TASK_CONTINUATION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Continue implementing the token verification endpoint for the previously started auth service.",
                historical_events=s1_events,
                task_prompt="Now implement the token verification endpoint for our authentication service.",
                project_id=p1,
                expected_facts=("bcrypt with work factor 12",),
                expected_decisions=("RS256 algorithm for JWT signing",),
                expected_constraints=("RSA 4096-bit keypair",),
                expected_conflicts=(),
                notes="Agent with Brain recalls RS256 and keypair; Baseline agent must ask or invent HS256/symmetric.",
            )
        )

        # ---------------------------------------------------------------------
        # 2. ARCHITECTURAL_DECISION: Recalling prior architectural decisions
        # ---------------------------------------------------------------------
        p2 = "proj-bench-cache"
        s2_events = (
            _make_event("ev-ad-1", EventType.USER_MESSAGE, "Should we use Memcached or Redis for caching?", p2, "user", "sess-2", base_time),
            _make_event("ev-ad-2", EventType.DECISION, "DECISION: Adopt Redis Sentinel for session caching because Memcached lacks replication and persistence.", p2, "agent-arch", "sess-2", base_time + timedelta(minutes=10)),
            _make_event("ev-ad-3", EventType.AGENT_MESSAGE, "The project uses Redis Sentinel on port 26379 with a 3-node replica set.", p2, "agent-arch", "sess-2", base_time + timedelta(minutes=12)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_02_ARCHITECTURAL_DECISION",
                name="Recall Caching Infrastructure Decision",
                category=ScenarioCategory.ARCHITECTURAL_DECISION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must configure cache client without re-asking what caching layer was chosen.",
                historical_events=s2_events,
                task_prompt="Configure the cache connection client for the web application.",
                project_id=p2,
                expected_facts=("Redis Sentinel", "port 26379"),
                expected_decisions=("Adopt Redis Sentinel for session caching", "Memcached lacks replication"),
                expected_constraints=("3-node replica set",),
                expected_conflicts=(),
            )
        )

        # ---------------------------------------------------------------------
        # 3. TECHNICAL_CONSTRAINTS: Adhering to strict historical technical boundaries
        # ---------------------------------------------------------------------
        p3 = "proj-bench-db"
        s3_events = (
            _make_event("ev-tc3-1", EventType.USER_MESSAGE, "Set up PostgreSQL database connection pooling.", p3, "user", "sess-3", base_time),
            _make_event("ev-tc3-2", EventType.DECISION, "DECISION: Database is PostgreSQL 16 on host db.internal port 5432.", p3, "agent-dba", "sess-3", base_time + timedelta(minutes=5)),
            _make_event("ev-tc3-3", EventType.AGENT_MESSAGE, "CONSTRAINT: Maximum connection pool size is 20, statement timeout 5000ms. Raw SQL strings are strictly prohibited; all queries must use SQLAlchemy Core.", p3, "agent-dba", "sess-3", base_time + timedelta(minutes=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_03_TECHNICAL_CONSTRAINTS",
                name="Enforce Database Connection Constraints",
                category=ScenarioCategory.TECHNICAL_CONSTRAINTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must construct a database query module respecting pool and timeout constraints.",
                historical_events=s3_events,
                task_prompt="Write the database query function to fetch active user accounts.",
                project_id=p3,
                expected_facts=("PostgreSQL 16", "db.internal port 5432"),
                expected_decisions=("PostgreSQL 16",),
                expected_constraints=("pool size is 20", "timeout 5000ms", "SQLAlchemy Core", "Raw SQL strings are strictly prohibited"),
                expected_conflicts=(),
            )
        )

        # ---------------------------------------------------------------------
        # 4. UPDATED_REQUIREMENTS: Superseding earlier requirements cleanly
        # ---------------------------------------------------------------------
        p4 = "proj-bench-api"
        s4_events = (
            _make_event("ev-ur-1", EventType.DECISION, "DECISION: The API protocol is REST JSON on port 8080.", p4, "agent-arch", "sess-4a", base_time),
            _make_event("ev-ur-2", EventType.USER_MESSAGE, "Update: We are migrating the inter-service API protocol from REST JSON to gRPC on port 9090.", p4, "user", "sess-4b", base_time + timedelta(days=2)),
            _make_event("ev-ur-3", EventType.DECISION, "DECISION: Supersede REST JSON with gRPC protobuf on port 9090 for inter-service communication.", p4, "agent-arch", "sess-4b", base_time + timedelta(days=2, minutes=5)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_04_UPDATED_REQUIREMENTS",
                name="Respect Requirement Supersession",
                category=ScenarioCategory.UPDATED_REQUIREMENTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Verify that agent adopts the updated gRPC protocol and rejects the superseded REST protocol.",
                historical_events=s4_events,
                task_prompt="Generate the client configuration for internal service calls.",
                project_id=p4,
                expected_facts=("gRPC", "port 9090"),
                expected_decisions=("gRPC protobuf on port 9090",),
                expected_constraints=("port 9090",),
                expected_conflicts=(),
                notes="Brain correctly supersedes port 8080 REST and delivers port 9090 gRPC.",
            )
        )

        # ---------------------------------------------------------------------
        # 5. CONTRADICTORY_CLAIMS: Identifying and preserving disputed claims
        # ---------------------------------------------------------------------
        p5 = "proj-bench-dispute"
        s5_events = (
            _make_event("ev-cc-1", EventType.AGENT_MESSAGE, "Agent-Alpha: Production database is MariaDB 10.11.", p5, "agent-alpha", "sess-5a", base_time),
            _make_event("ev-cc-2", EventType.AGENT_MESSAGE, "Agent-Beta: Production database is PostgreSQL 16.", p5, "agent-beta", "sess-5b", base_time + timedelta(hours=1)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_05_CONTRADICTORY_CLAIMS",
                name="Detect and Preserve Contradictory Database Claims",
                category=ScenarioCategory.CONTRADICTORY_CLAIMS,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Two agents made conflicting assertions about the primary database engine. The system must preserve the dispute warning.",
                historical_events=s5_events,
                task_prompt="What database driver should we install for production deployment?",
                project_id=p5,
                expected_facts=(),
                expected_decisions=(),
                expected_constraints=(),
                expected_conflicts=("MariaDB 10.11", "PostgreSQL 16"),
                notes="Brain marks claims as DISPUTED and emits dispute warnings in context.",
            )
        )

        # ---------------------------------------------------------------------
        # 6. AGENT_SWITCHING: Cross-agent context continuity
        # ---------------------------------------------------------------------
        p6 = "proj-bench-handoff"
        s6_events = (
            _make_event("ev-as-1", EventType.USER_MESSAGE, "Research Kafka versus RabbitMQ for order event processing.", p6, "user", "sess-6", base_time),
            _make_event("ev-as-2", EventType.AGENT_MESSAGE, "Research findings: Kafka is required because we need at-least-once message retention for 7 days and replay capability.", p6, "researcher", "sess-6", base_time + timedelta(minutes=10)),
            _make_event("ev-as-3", EventType.DECISION, "DECISION: Select Apache Kafka with 3 partitions per topic.", p6, "researcher", "sess-6", base_time + timedelta(minutes=15)),
            _make_event("ev-as-4", EventType.AGENT_SWITCH, "Handover from researcher to coder agent for producer implementation.", p6, "agent-switcher", "sess-6", base_time + timedelta(minutes=20), metadata={"from_agent": "researcher", "to_agent": "coder"}),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_06_AGENT_SWITCHING",
                name="Cross-Agent Handoff Continuity",
                category=ScenarioCategory.AGENT_SWITCHING,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="CodingAgent must immediately continue from ResearchAgent's findings without re-researching.",
                historical_events=s6_events,
                task_prompt="As the coder agent, implement the message publisher for order events.",
                project_id=p6,
                expected_facts=("Apache Kafka", "7 days", "replay capability"),
                expected_decisions=("Select Apache Kafka with 3 partitions per topic",),
                expected_constraints=("3 partitions per topic",),
                expected_conflicts=(),
            )
        )

        # ---------------------------------------------------------------------
        # 7. SESSION_INTERRUPTION: Resuming interrupted task without losing state
        # ---------------------------------------------------------------------
        p7 = "proj-bench-resume"
        s7_events = (
            _make_event("ev-si-1", EventType.USER_MESSAGE, "Begin migrating legacy user records to the new UUID v7 format.", p7, "user", "sess-7a", base_time),
            _make_event("ev-si-2", EventType.AGENT_MESSAGE, "Processed batches 1 through 5 (IDs 1-50,000). Batch 6 failed at record ID 52,143 due to null email.", p7, "agent-migrator", "sess-7a", base_time + timedelta(minutes=30)),
            _make_event("ev-si-3", EventType.ERROR, "NullEmailError: record 52143 in batch 6 missing required email address.", p7, "agent-migrator", "sess-7a", base_time + timedelta(minutes=31)),
            _make_event("ev-si-4", EventType.SESSION_END, "Session 7a terminated due to unexpected error.", p7, "system", "sess-7a", base_time + timedelta(minutes=32)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_07_SESSION_INTERRUPTION",
                name="Resume Interrupted Data Migration",
                category=ScenarioCategory.SESSION_INTERRUPTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="Agent starts in a fresh session (sess-7b) and must resume from batch 6 record 52143.",
                historical_events=s7_events,
                task_prompt="Resume the data migration where we left off.",
                project_id=p7,
                expected_facts=("Batches 1 through 5 completed", "record 52,143", "null email"),
                expected_decisions=(),
                expected_constraints=("UUID v7 format",),
                expected_conflicts=(),
            )
        )

        # ---------------------------------------------------------------------
        # 8. REPEATED_EXPLANATION: Avoiding redundant user explanations
        # ---------------------------------------------------------------------
        p8 = "proj-bench-repexp"
        s8_events = (
            _make_event("ev-re-1", EventType.AGENT_MESSAGE, "The project uses Stripe API version 2023-10-16 with webhook secret whsec_prod_xyz.", p8, "agent-coder", "sess-8a", base_time),
            _make_event("ev-re-2", EventType.DECISION, "DECISION: All Stripe webhooks must verify HMAC-SHA256 signature using whsec_prod_xyz.", p8, "agent-coder", "sess-8a", base_time + timedelta(minutes=5)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_08_REPEATED_EXPLANATION",
                name="Avoid Repeating Webhook Secret and Gateway Info",
                category=ScenarioCategory.REPEATED_EXPLANATION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="User simply asks: 'Write the webhook handler endpoint.' Without Brain, user must re-explain Stripe credentials.",
                historical_events=s8_events,
                task_prompt="Write the webhook handler endpoint to handle payment success events.",
                project_id=p8,
                expected_facts=("Stripe", "API version 2023-10-16"),
                expected_decisions=("verify HMAC-SHA256 signature",),
                expected_constraints=("whsec_prod_xyz",),
                expected_conflicts=(),
            )
        )

        # ---------------------------------------------------------------------
        # 9. NEEDLE_IN_HAYSTACK: Retrieving relevant fact among 15 distractors
        # ---------------------------------------------------------------------
        p9 = "proj-bench-needle"
        needle_events = [
            _make_event("ev-nih-target", EventType.DECISION, "DECISION: The rate limiter threshold for public search endpoints is exactly 42 requests per minute per IP.", p9, "agent-arch", "sess-9", base_time + timedelta(minutes=2)),
        ]
        distractors = []
        for i in range(15):
            distractors.append(
                _make_event(f"ev-dist-{i}", EventType.AGENT_MESSAGE, f"System log note {i}: Background sync worker {i} checked status OK at port {9000 + i}.", p9, "agent-sync", "sess-9", base_time + timedelta(minutes=3 + i))
            )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_09_NEEDLE_IN_HAYSTACK",
                name="Retrieve Search Rate Limiter Threshold Among Noise",
                category=ScenarioCategory.NEEDLE_IN_HAYSTACK,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.SYNTHETIC,
                description="Retrieve the exact 42 req/min rate limit from a noisy event history with 15 background sync distractors.",
                historical_events=tuple(needle_events),
                task_prompt="What is the configured rate limit for the public search endpoint?",
                project_id=p9,
                expected_facts=("42 requests per minute per IP",),
                expected_decisions=("rate limiter threshold for public search",),
                expected_constraints=("42 requests per minute",),
                expected_conflicts=(),
                distractor_events=tuple(distractors),
            )
        )

        # ---------------------------------------------------------------------
        # 10. STALE_KNOWLEDGE_REJECTION: Rejecting outdated superseded auth token
        # ---------------------------------------------------------------------
        p10 = "proj-bench-stale"
        s10_events = (
            _make_event("ev-sk-1", EventType.DECISION, "DECISION: Temporary token format is MD5 hex string.", p10, "agent-dev", "sess-10a", base_time),
            _make_event("ev-sk-2", EventType.DECISION, "DECISION: Supersede MD5 tokens. Permanent production token format is OAuth2 Bearer with Ed25519 signatures.", p10, "agent-sec", "sess-10b", base_time + timedelta(days=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_10_STALE_KNOWLEDGE_REJECTION",
                name="Reject Outdated MD5 Token in Favor of Ed25519",
                category=ScenarioCategory.STALE_KNOWLEDGE_REJECTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must use the current OAuth2 Bearer Ed25519 token format and reject the stale MD5 format.",
                historical_events=s10_events,
                task_prompt="Generate the token parsing middleware for incoming API requests.",
                project_id=p10,
                expected_facts=("OAuth2 Bearer", "Ed25519"),
                expected_decisions=("OAuth2 Bearer with Ed25519 signatures",),
                expected_constraints=("Ed25519",),
                expected_conflicts=(),
                notes="Brain marks MD5 as SUPERSEDED; only Ed25519 is included in active context.",
            )
        )
