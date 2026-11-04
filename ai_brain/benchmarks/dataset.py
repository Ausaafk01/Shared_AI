"""
Benchmark dataset containing realistic multi-session scenarios for AI Brain evaluation.
Includes 30 structured scenarios (3 per category) across 10 categories with explicit provenance,
ground-truth expected outcomes, and distractor noise.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from ai_brain.benchmarks.models import (
    BenchmarkScenario,
    ConflictGroundTruth,
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
        if scenario_id in self._scenarios:
            return self._scenarios[scenario_id]
        # Support aliases without letter (e.g. SCENARIO_01_TASK_CONTINUATION -> SCENARIO_01A_TASK_CONTINUATION)
        for sid, s in self._scenarios.items():
            normalized = sid.replace("01A_", "01_").replace("02A_", "02_").replace("03A_", "03_") \
                            .replace("04A_", "04_").replace("05A_", "05_").replace("06A_", "06_") \
                            .replace("07A_", "07_").replace("08A_", "08_").replace("09A_", "09_") \
                            .replace("10A_", "10_")
            if normalized == scenario_id:
                return s
        return None

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
        base_time = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)

        # =====================================================================
        # 1. TASK_CONTINUATION (3 Scenarios)
        # =====================================================================
        p1a = "proj-bench-auth"
        s1a_events = (
            _make_event("ev-tc-1", EventType.USER_MESSAGE, "Implement user authentication with JWT and bcrypt.", p1a, "user", "sess-1a", base_time),
            _make_event("ev-tc-2", EventType.DECISION, "DECISION: Use RS256 algorithm for JWT signing with RSA 4096-bit keypair.", p1a, "agent-coder", "sess-1a", base_time + timedelta(minutes=5)),
            _make_event("ev-tc-3", EventType.CODE_CHANGE, "Added user password hashing using bcrypt with work factor 12.", p1a, "agent-coder", "sess-1a", base_time + timedelta(minutes=15)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_01A_TASK_CONTINUATION",
                name="Continue Authentication Service Implementation",
                category=ScenarioCategory.TASK_CONTINUATION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Continue implementing token verification endpoint using previously established RS256 signing.",
                historical_events=s1a_events,
                task_prompt="Now implement the token verification endpoint for our authentication service.",
                project_id=p1a,
                expected_facts=("bcrypt with work factor 12",),
                expected_decisions=("RS256 algorithm for JWT signing",),
                expected_constraints=("RSA 4096-bit keypair",),
                expected_conflicts=(),
            )
        )

        p1b = "proj-bench-payment"
        s1b_events = (
            _make_event("ev-tc-b1", EventType.USER_MESSAGE, "Integrate Stripe Checkout for subscription billing.", p1b, "user", "sess-1b", base_time),
            _make_event("ev-tc-b2", EventType.DECISION, "DECISION: Use Stripe Checkout Sessions in subscription mode with currency USD.", p1b, "agent-coder", "sess-1b", base_time + timedelta(minutes=5)),
            _make_event("ev-tc-b3", EventType.AGENT_MESSAGE, "The project uses Stripe Checkout API version 2023-10-16.", p1b, "agent-coder", "sess-1b", base_time + timedelta(minutes=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_01B_TASK_CONTINUATION",
                name="Continue Stripe Payment Workflow to Refund Processing",
                category=ScenarioCategory.TASK_CONTINUATION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Continue billing implementation by adding the customer refund processing endpoint.",
                historical_events=s1b_events,
                task_prompt="Implement the refund handling endpoint for our Stripe billing system.",
                project_id=p1b,
                expected_facts=("Stripe Checkout API version 2023-10-16",),
                expected_decisions=("Stripe Checkout Sessions in subscription mode",),
                expected_constraints=("currency USD",),
                expected_conflicts=(),
            )
        )

        p1c = "proj-bench-cicd"
        s1c_events = (
            _make_event("ev-tc-c1", EventType.USER_MESSAGE, "Configure our GitHub Actions CI pipeline for Python.", p1c, "user", "sess-1c", base_time),
            _make_event("ev-tc-c2", EventType.DECISION, "DECISION: CI pipeline runs on ubuntu-22.04 with Python 3.12 and poetry package manager.", p1c, "agent-devops", "sess-1c", base_time + timedelta(minutes=5)),
            _make_event("ev-tc-c3", EventType.CODE_CHANGE, "Configured ruff linter and pytest steps in .github/workflows/ci.yml.", p1c, "agent-devops", "sess-1c", base_time + timedelta(minutes=15)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_01C_TASK_CONTINUATION",
                name="Continue CI Pipeline Setup to Docker Packaging",
                category=ScenarioCategory.TASK_CONTINUATION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Add the automated Docker build and container registry publish step to the existing CI pipeline.",
                historical_events=s1c_events,
                task_prompt="Add the container image build and publishing step to our existing CI workflow.",
                project_id=p1c,
                expected_facts=("ubuntu-22.04", "poetry package manager"),
                expected_decisions=("ubuntu-22.04 with Python 3.12",),
                expected_constraints=("Python 3.12", "poetry"),
                expected_conflicts=(),
            )
        )

        # =====================================================================
        # 2. ARCHITECTURAL_DECISION (3 Scenarios)
        # =====================================================================
        p2a = "proj-bench-cache"
        s2a_events = (
            _make_event("ev-ad-1", EventType.USER_MESSAGE, "Should we use Memcached or Redis for caching?", p2a, "user", "sess-2a", base_time),
            _make_event("ev-ad-2", EventType.DECISION, "DECISION: Adopt Redis Sentinel for session caching because Memcached lacks replication and persistence.", p2a, "agent-arch", "sess-2a", base_time + timedelta(minutes=10)),
            _make_event("ev-ad-3", EventType.AGENT_MESSAGE, "The project uses Redis Sentinel on port 26379 with a 3-node replica set.", p2a, "agent-arch", "sess-2a", base_time + timedelta(minutes=12)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_02A_ARCHITECTURAL_DECISION",
                name="Recall Caching Infrastructure Decision",
                category=ScenarioCategory.ARCHITECTURAL_DECISION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must configure cache client without re-asking what caching layer was chosen.",
                historical_events=s2a_events,
                task_prompt="Configure the cache connection client for the web application.",
                project_id=p2a,
                expected_facts=("Redis Sentinel", "port 26379"),
                expected_decisions=("Adopt Redis Sentinel for session caching", "Memcached lacks replication"),
                expected_constraints=("3-node replica set",),
                expected_conflicts=(),
            )
        )

        p2b = "proj-bench-dbengine"
        s2b_events = (
            _make_event("ev-ad-b1", EventType.USER_MESSAGE, "Select primary database for e-commerce transactions: MongoDB vs PostgreSQL.", p2b, "user", "sess-2b", base_time),
            _make_event("ev-ad-b2", EventType.DECISION, "DECISION: Use PostgreSQL 16 because ACID transactions and foreign key guarantees are mandatory for ledger entries.", p2b, "agent-arch", "sess-2b", base_time + timedelta(minutes=15)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_02B_ARCHITECTURAL_DECISION",
                name="Recall Transaction Database Engine Decision",
                category=ScenarioCategory.ARCHITECTURAL_DECISION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Recall PostgreSQL selection for the order ledger without proposing MongoDB or Document stores.",
                historical_events=s2b_events,
                task_prompt="Define the data access schema for our new order processing module.",
                project_id=p2b,
                expected_facts=("PostgreSQL 16", "ACID transactions"),
                expected_decisions=("Use PostgreSQL 16 because ACID transactions",),
                expected_constraints=("foreign key guarantees",),
                expected_conflicts=(),
            )
        )

        p2c = "proj-bench-msgformat"
        s2c_events = (
            _make_event("ev-ad-c1", EventType.USER_MESSAGE, "What message format should services use over the internal queue?", p2c, "user", "sess-2c", base_time),
            _make_event("ev-ad-c2", EventType.DECISION, "DECISION: Standardize on Protocol Buffers v3 with buf schema registry for internal service serialization.", p2c, "agent-arch", "sess-2c", base_time + timedelta(minutes=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_02C_ARCHITECTURAL_DECISION",
                name="Recall Inter-Service Serialization Decision",
                category=ScenarioCategory.ARCHITECTURAL_DECISION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Recall Protobuf v3 selection instead of proposing raw JSON payloads.",
                historical_events=s2c_events,
                task_prompt="Define the event serialization payload for user registration events.",
                project_id=p2c,
                expected_facts=("Protocol Buffers v3", "buf schema registry"),
                expected_decisions=("Standardize on Protocol Buffers v3",),
                expected_constraints=("Protocol Buffers v3",),
                expected_conflicts=(),
            )
        )

        # =====================================================================
        # 3. TECHNICAL_CONSTRAINTS (3 Scenarios)
        # =====================================================================
        p3a = "proj-bench-db"
        s3a_events = (
            _make_event("ev-tc3-1", EventType.USER_MESSAGE, "Set up PostgreSQL database connection pooling.", p3a, "user", "sess-3a", base_time),
            _make_event("ev-tc3-2", EventType.DECISION, "DECISION: Database is PostgreSQL 16 on host db.internal port 5432.", p3a, "agent-dba", "sess-3a", base_time + timedelta(minutes=5)),
            _make_event("ev-tc3-3", EventType.AGENT_MESSAGE, "The project uses connection pool size 20 and statement timeout 5000ms. Raw SQL strings are strictly prohibited; all queries must use SQLAlchemy Core.", p3a, "agent-dba", "sess-3a", base_time + timedelta(minutes=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_03A_TECHNICAL_CONSTRAINTS",
                name="Enforce Database Connection Constraints",
                category=ScenarioCategory.TECHNICAL_CONSTRAINTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must construct a database query module respecting pool size 20, 5000ms timeout, and SQLAlchemy Core.",
                historical_events=s3a_events,
                task_prompt="Write the database query function to fetch active user accounts.",
                project_id=p3a,
                expected_facts=("PostgreSQL 16", "db.internal port 5432"),
                expected_decisions=("PostgreSQL 16",),
                expected_constraints=("pool size 20", "timeout 5000ms", "SQLAlchemy Core"),
                expected_conflicts=(),
            )
        )

        p3b = "proj-bench-ratelimit"
        s3b_events = (
            _make_event("ev-tc-b1", EventType.DECISION, "DECISION: Maximum HTTP request payload is 10MB and global rate limit is 100 requests per minute per IP.", p3b, "agent-sec", "sess-3b", base_time),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_03B_TECHNICAL_CONSTRAINTS",
                name="Enforce Payload Size and Rate Limits",
                category=ScenarioCategory.TECHNICAL_CONSTRAINTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must configure FastAPI middleware enforcing 10MB payload and 100 req/min limits.",
                historical_events=s3b_events,
                task_prompt="Configure the request rate limiter and body size validator middleware.",
                project_id=p3b,
                expected_facts=("10MB", "100 requests per minute"),
                expected_decisions=("Maximum HTTP request payload is 10MB",),
                expected_constraints=("10MB", "100 requests per minute"),
                expected_conflicts=(),
            )
        )

        p3c = "proj-bench-crypto"
        s3c_events = (
            _make_event("ev-tc-c1", EventType.DECISION, "DECISION: All encrypted fields must use AES-256-GCM authenticated encryption. CBC mode is strictly prohibited.", p3c, "agent-sec", "sess-3c", base_time),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_03C_TECHNICAL_CONSTRAINTS",
                name="Enforce Cryptographic Cipher Standards",
                category=ScenarioCategory.TECHNICAL_CONSTRAINTS,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must configure column-level encryption using AES-256-GCM, rejecting legacy CBC.",
                historical_events=s3c_events,
                task_prompt="Write the database encryption helper function for sensitive user tax IDs.",
                project_id=p3c,
                expected_facts=("AES-256-GCM",),
                expected_decisions=("AES-256-GCM authenticated encryption",),
                expected_constraints=("AES-256-GCM", "CBC mode is strictly prohibited"),
                expected_conflicts=(),
            )
        )

        # =====================================================================
        # 4. UPDATED_REQUIREMENTS (3 Scenarios)
        # =====================================================================
        p4a = "proj-bench-api"
        s4a_events = (
            _make_event("ev-ur-1", EventType.DECISION, "DECISION: The API protocol is REST JSON on port 8080.", p4a, "agent-arch", "sess-4a", base_time),
            _make_event("ev-ur-2", EventType.DECISION, "DECISION: Supersede REST JSON with gRPC protobuf on port 9090 for inter-service communication.", p4a, "agent-arch", "sess-4b", base_time + timedelta(days=2)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_04A_UPDATED_REQUIREMENTS",
                name="Respect Requirement Supersession (REST to gRPC)",
                category=ScenarioCategory.UPDATED_REQUIREMENTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Verify that agent adopts the updated gRPC protocol and rejects the superseded REST protocol.",
                historical_events=s4a_events,
                task_prompt="Generate the client configuration for internal service calls.",
                project_id=p4a,
                expected_facts=("gRPC", "port 9090"),
                expected_decisions=("gRPC protobuf on port 9090",),
                expected_constraints=("port 9090",),
                expected_conflicts=(),
                conflict_ground_truth=ConflictGroundTruth.TEMPORAL_UPDATE,
            )
        )

        p4b = "proj-bench-sessiontime"
        s4b_events = (
            _make_event("ev-ur-b1", EventType.DECISION, "DECISION: User login session expiry is fixed at 24 hours.", p4b, "agent-lead", "sess-4b1", base_time),
            _make_event("ev-ur-b2", EventType.DECISION, "DECISION: Supersede 24-hour fixed expiry. Security policy mandates a 15-minute sliding window with absolute 8-hour cap.", p4b, "agent-sec", "sess-4b2", base_time + timedelta(days=5)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_04B_UPDATED_REQUIREMENTS",
                name="Respect Session Timeout Policy Update",
                category=ScenarioCategory.UPDATED_REQUIREMENTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must configure 15-minute sliding expiry and reject the superseded 24-hour rule.",
                historical_events=s4b_events,
                task_prompt="Configure the user session cookie expiration parameters.",
                project_id=p4b,
                expected_facts=("15-minute sliding window", "8-hour cap"),
                expected_decisions=("15-minute sliding window",),
                expected_constraints=("15-minute", "8-hour"),
                expected_conflicts=(),
                conflict_ground_truth=ConflictGroundTruth.TEMPORAL_UPDATE,
            )
        )

        p4c = "proj-bench-logging"
        s4c_events = (
            _make_event("ev-ur-c1", EventType.DECISION, "DECISION: Application logs are written to local file /var/log/app.log.", p4c, "agent-coder", "sess-4c1", base_time),
            _make_event("ev-ur-c2", EventType.DECISION, "DECISION: Supersede local file logging. All application logs must stream via Fluentd socket on port 24224 in JSON format.", p4c, "agent-devops", "sess-4c2", base_time + timedelta(days=4)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_04C_UPDATED_REQUIREMENTS",
                name="Respect Logging Destination Migration",
                category=ScenarioCategory.UPDATED_REQUIREMENTS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Configure log handler to stream via Fluentd port 24224, avoiding superseded file logging.",
                historical_events=s4c_events,
                task_prompt="Configure the production logging handler for the backend service.",
                project_id=p4c,
                expected_facts=("Fluentd socket", "port 24224"),
                expected_decisions=("stream via Fluentd socket on port 24224",),
                expected_constraints=("port 24224",),
                expected_conflicts=(),
                conflict_ground_truth=ConflictGroundTruth.TEMPORAL_UPDATE,
            )
        )

        # =====================================================================
        # 5. CONTRADICTORY_CLAIMS (3 Scenarios)
        # =====================================================================
        p5a = "proj-bench-dispute"
        s5a_events = (
            _make_event("ev-cc-1", EventType.AGENT_MESSAGE, "The project uses MariaDB 10.11 for production database.", p5a, "agent-alpha", "sess-5a", base_time),
            _make_event("ev-cc-2", EventType.AGENT_MESSAGE, "The project uses PostgreSQL 16 for production database.", p5a, "agent-beta", "sess-5b", base_time + timedelta(hours=1)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_05A_CONTRADICTORY_CLAIMS",
                name="Detect Contradictory Database Engine Claims",
                category=ScenarioCategory.CONTRADICTORY_CLAIMS,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Two agents made conflicting assertions about the primary database engine. The system must preserve the dispute warning.",
                historical_events=s5a_events,
                task_prompt="What database driver should we install for production deployment?",
                project_id=p5a,
                expected_facts=(),
                expected_decisions=(),
                expected_constraints=(),
                expected_conflicts=("MariaDB 10.11", "PostgreSQL 16"),
                conflict_ground_truth=ConflictGroundTruth.GENUINE_CONTRADICTION,
            )
        )

        p5b = "proj-bench-apiversion"
        s5b_events = (
            _make_event("ev-cc-b1", EventType.AGENT_MESSAGE, "The project uses API version v2.1 which is deprecated and scheduled for shutdown.", p5b, "agent-1", "sess-5b1", base_time),
            _make_event("ev-cc-b2", EventType.AGENT_MESSAGE, "The project uses API version v2.1 which is the active long-term stable release.", p5b, "agent-2", "sess-5b2", base_time + timedelta(hours=2)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_05B_CONTRADICTORY_CLAIMS",
                name="Detect Contradictory API Deprecation Assertions",
                category=ScenarioCategory.CONTRADICTORY_CLAIMS,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Conflicting assertions on whether API v2.1 is deprecated or active LTS. Preserve dispute.",
                historical_events=s5b_events,
                task_prompt="Determine if we can continue using API v2.1 for our mobile clients.",
                project_id=p5b,
                expected_facts=(),
                expected_decisions=(),
                expected_constraints=(),
                expected_conflicts=("deprecated", "active long-term stable"),
                conflict_ground_truth=ConflictGroundTruth.GENUINE_CONTRADICTION,
            )
        )

        p5c = "proj-bench-compatible"
        s5c_events = (
            _make_event("ev-cc-c1", EventType.AGENT_MESSAGE, "The project uses Redis on port 6379 for fast temporary cache.", p5c, "agent-arch", "sess-5c", base_time),
            _make_event("ev-cc-c2", EventType.AGENT_MESSAGE, "The project uses PostgreSQL 16 on port 5432 for primary durable storage.", p5c, "agent-dba", "sess-5c", base_time + timedelta(minutes=5)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_05C_CONTRADICTORY_CLAIMS",
                name="Compatible Complementary Claims (Negative Conflict Test)",
                category=ScenarioCategory.CONTRADICTORY_CLAIMS,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Redis cache and PostgreSQL database are complementary. Verify NO false-positive conflict is detected.",
                historical_events=s5c_events,
                task_prompt="Summarize our storage infrastructure stack.",
                project_id=p5c,
                expected_facts=("Redis on port 6379", "PostgreSQL 16 on port 5432"),
                expected_decisions=(),
                expected_constraints=(),
                expected_conflicts=(),  # Zero conflicts expected
                conflict_ground_truth=ConflictGroundTruth.COMPATIBLE_CLAIMS,
            )
        )

        # =====================================================================
        # 6. AGENT_SWITCHING (3 Scenarios)
        # =====================================================================
        p6a = "proj-bench-handoff"
        s6a_events = (
            _make_event("ev-as-1", EventType.USER_MESSAGE, "Research Kafka versus RabbitMQ for order event processing.", p6a, "user", "sess-6a", base_time),
            _make_event("ev-as-2", EventType.AGENT_MESSAGE, "Research findings: The project uses Apache Kafka with message retention for 7 days and replay capability.", p6a, "researcher", "sess-6a", base_time + timedelta(minutes=10)),
            _make_event("ev-as-3", EventType.DECISION, "DECISION: Select Apache Kafka with 3 partitions per topic.", p6a, "researcher", "sess-6a", base_time + timedelta(minutes=15)),
            _make_event("ev-as-4", EventType.AGENT_SWITCH, "Handover from researcher to coder agent for producer implementation.", p6a, "agent-switcher", "sess-6a", base_time + timedelta(minutes=20)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_06A_AGENT_SWITCHING",
                name="Cross-Agent Handoff Continuity (Researcher to Coder)",
                category=ScenarioCategory.AGENT_SWITCHING,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="CodingAgent must immediately continue from ResearchAgent's findings without re-researching.",
                historical_events=s6a_events,
                task_prompt="As the coder agent, implement the message publisher for order events.",
                project_id=p6a,
                expected_facts=("Apache Kafka", "7 days"),
                expected_decisions=("Select Apache Kafka with 3 partitions per topic",),
                expected_constraints=("3 partitions per topic",),
                expected_conflicts=(),
            )
        )

        p6b = "proj-bench-coder-reviewer"
        s6b_events = (
            _make_event("ev-as-b1", EventType.CODE_CHANGE, "Implemented user query using parameterized queries with SQLAlchemy text bindparams.", p6b, "agent-coder", "sess-6b", base_time),
            _make_event("ev-as-b2", EventType.DECISION, "DECISION: All user queries must use bind parameters to avoid SQL injection.", p6b, "agent-coder", "sess-6b", base_time + timedelta(minutes=5)),
            _make_event("ev-as-b3", EventType.AGENT_SWITCH, "Switch to review agent for security validation.", p6b, "agent-switcher", "sess-6b", base_time + timedelta(minutes=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_06B_AGENT_SWITCHING",
                name="Cross-Agent Security Review Handoff",
                category=ScenarioCategory.AGENT_SWITCHING,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="ReviewAgent evaluates security posture using context from CoderAgent.",
                historical_events=s6b_events,
                task_prompt="Review the database query implementation for SQL injection vulnerabilities.",
                project_id=p6b,
                expected_facts=("parameterized queries",),
                expected_decisions=("use bind parameters to avoid SQL injection",),
                expected_constraints=("bind parameters",),
                expected_conflicts=(),
            )
        )

        p6c = "proj-bench-devops-qa"
        s6c_events = (
            _make_event("ev-as-c1", EventType.DECISION, "DECISION: Staging cluster URL is https://staging.internal:8443 with TLS 1.3.", p6c, "agent-devops", "sess-6c", base_time),
            _make_event("ev-as-c2", EventType.AGENT_SWITCH, "Handover to QA agent for automated smoke test suite.", p6c, "agent-switcher", "sess-6c", base_time + timedelta(minutes=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_06C_AGENT_SWITCHING",
                name="DevOps to QA Smoke Testing Handoff",
                category=ScenarioCategory.AGENT_SWITCHING,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="QA agent receives staging cluster endpoint and TLS requirements from DevOps.",
                historical_events=s6c_events,
                task_prompt="Execute the automated health and smoke test script against staging.",
                project_id=p6c,
                expected_facts=("https://staging.internal:8443", "TLS 1.3"),
                expected_decisions=("Staging cluster URL is https://staging.internal:8443",),
                expected_constraints=("TLS 1.3",),
                expected_conflicts=(),
            )
        )

        # =====================================================================
        # 7. SESSION_INTERRUPTION (3 Scenarios)
        # =====================================================================
        p7a = "proj-bench-resume"
        s7a_events = (
            _make_event("ev-si-1", EventType.USER_MESSAGE, "Begin migrating legacy user records to the new UUID v7 format.", p7a, "user", "sess-7a", base_time),
            _make_event("ev-si-2", EventType.AGENT_MESSAGE, "The project uses UUID v7 migration. Processed batches 1 through 5 (IDs 1-50,000). Batch 6 failed at record ID 52,143 due to null email.", p7a, "agent-migrator", "sess-7a", base_time + timedelta(minutes=30)),
            _make_event("ev-si-3", EventType.SESSION_END, "Session 7a terminated unexpectedly.", p7a, "system", "sess-7a", base_time + timedelta(minutes=32)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_07A_SESSION_INTERRUPTION",
                name="Resume Interrupted Data Migration",
                category=ScenarioCategory.SESSION_INTERRUPTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="Agent starts in a fresh session (sess-7b) and must resume from batch 6 record 52143.",
                historical_events=s7a_events,
                task_prompt="Resume the data migration where we left off.",
                project_id=p7a,
                expected_facts=("Batches 1 through 5", "record ID 52,143"),
                expected_decisions=(),
                expected_constraints=("UUID v7",),
                expected_conflicts=(),
            )
        )

        p7b = "proj-bench-fileresume"
        s7b_events = (
            _make_event("ev-si-b1", EventType.AGENT_MESSAGE, "The project uses batch document processing. Processed documents 1 through 141 out of 500. Document 142 timed out on OCR conversion.", p7b, "agent-worker", "sess-7b1", base_time),
            _make_event("ev-si-b2", EventType.SESSION_END, "Session crashed on timeout.", p7b, "system", "sess-7b1", base_time + timedelta(minutes=15)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_07B_SESSION_INTERRUPTION",
                name="Resume Interrupted Document OCR Processing",
                category=ScenarioCategory.SESSION_INTERRUPTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="Agent resumes document batch processing starting at document 142.",
                historical_events=s7b_events,
                task_prompt="Resume the document OCR processing pipeline.",
                project_id=p7b,
                expected_facts=("documents 1 through 141", "Document 142 timed out"),
                expected_decisions=(),
                expected_constraints=(),
                expected_conflicts=(),
            )
        )

        p7c = "proj-bench-schemaresume"
        s7c_events = (
            _make_event("ev-si-c1", EventType.AGENT_MESSAGE, "The project uses Alembic migrations. Applied migrations 001_initial, 002_users, and 003_orders. Migration 004_add_indexes failed with lock wait timeout.", p7c, "agent-dba", "sess-7c1", base_time),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_07C_SESSION_INTERRUPTION",
                name="Resume Interrupted Database Schema Migration",
                category=ScenarioCategory.SESSION_INTERRUPTION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.REAL_WORLD_DERIVED,
                description="Agent resumes migration sequence starting with 004_add_indexes.",
                historical_events=s7c_events,
                task_prompt="Continue the database schema migration sequence.",
                project_id=p7c,
                expected_facts=("004_add_indexes failed",),
                expected_decisions=(),
                expected_constraints=("Alembic migrations",),
                expected_conflicts=(),
            )
        )

        # =====================================================================
        # 8. REPEATED_EXPLANATION (3 Scenarios)
        # =====================================================================
        p8a = "proj-bench-repexp"
        s8a_events = (
            _make_event("ev-re-1", EventType.AGENT_MESSAGE, "The project uses Stripe API version 2023-10-16 with webhook secret whsec_prod_xyz.", p8a, "agent-coder", "sess-8a", base_time),
            _make_event("ev-re-2", EventType.DECISION, "DECISION: All Stripe webhooks must verify HMAC-SHA256 signature using whsec_prod_xyz.", p8a, "agent-coder", "sess-8a", base_time + timedelta(minutes=5)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_08A_REPEATED_EXPLANATION",
                name="Avoid Repeating Webhook Secret and Gateway Info",
                category=ScenarioCategory.REPEATED_EXPLANATION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="User asks to write webhook endpoint. Brain recalls Stripe and webhook secret without re-explanation.",
                historical_events=s8a_events,
                task_prompt="Write the webhook handler endpoint to handle payment success events.",
                project_id=p8a,
                expected_facts=("Stripe", "API version 2023-10-16"),
                expected_decisions=("verify HMAC-SHA256 signature",),
                expected_constraints=("whsec_prod_xyz",),
                expected_conflicts=(),
            )
        )

        p8b = "proj-bench-mtls"
        s8b_events = (
            _make_event("ev-re-b1", EventType.DECISION, "DECISION: Service mesh mTLS client certificate path is /etc/ssl/certs/mesh-client.pem with key /etc/ssl/private/mesh-client.key.", p8b, "agent-sec", "sess-8b", base_time),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_08B_REPEATED_EXPLANATION",
                name="Avoid Repeating Service Mesh mTLS Certificate Paths",
                category=ScenarioCategory.REPEATED_EXPLANATION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent configures HTTP transport without forcing user to re-provide certificate file paths.",
                historical_events=s8b_events,
                task_prompt="Configure the mTLS secure transport for the internal service client.",
                project_id=p8b,
                expected_facts=("/etc/ssl/certs/mesh-client.pem",),
                expected_decisions=("mTLS client certificate path",),
                expected_constraints=("/etc/ssl/private/mesh-client.key",),
                expected_conflicts=(),
            )
        )

        p8c = "proj-bench-analytics"
        s8c_events = (
            _make_event("ev-re-c1", EventType.DECISION, "DECISION: Analytics tracking uses PostHog host https://posthog.internal with project api key ph_prod_8829.", p8c, "agent-lead", "sess-8c", base_time),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_08C_REPEATED_EXPLANATION",
                name="Avoid Repeating Product Analytics Configuration",
                category=ScenarioCategory.REPEATED_EXPLANATION,
                difficulty=ScenarioDifficulty.EASY,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent implements telemetry tracking without re-asking for PostHog endpoint and project token.",
                historical_events=s8c_events,
                task_prompt="Implement the telemetry event tracker for checkout button clicks.",
                project_id=p8c,
                expected_facts=("https://posthog.internal", "ph_prod_8829"),
                expected_decisions=("PostHog host https://posthog.internal",),
                expected_constraints=("ph_prod_8829",),
                expected_conflicts=(),
            )
        )

        # =====================================================================
        # 9. NEEDLE_IN_HAYSTACK (3 Scenarios)
        # =====================================================================
        p9a = "proj-bench-needle"
        needle_a = [
            _make_event("ev-nih-target", EventType.DECISION, "DECISION: The rate limiter threshold for public search endpoints is exactly 42 requests per minute per IP.", p9a, "agent-arch", "sess-9a", base_time + timedelta(minutes=2)),
        ]
        distractors_a = [
            _make_event(f"ev-dist-a{i}", EventType.AGENT_MESSAGE, f"The project uses background sync worker {i} on port {9000 + i}.", p9a, "agent-sync", "sess-9a", base_time + timedelta(minutes=3 + i))
            for i in range(15)
        ]
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_09A_NEEDLE_IN_HAYSTACK",
                name="Retrieve Search Rate Limiter Threshold Among Noise",
                category=ScenarioCategory.NEEDLE_IN_HAYSTACK,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.SYNTHETIC,
                description="Retrieve the exact 42 req/min rate limit from an event history containing 15 background sync distractors.",
                historical_events=tuple(needle_a),
                task_prompt="What is the configured rate limit for the public search endpoint?",
                project_id=p9a,
                expected_facts=("42 requests per minute per IP",),
                expected_decisions=("rate limiter threshold for public search",),
                expected_constraints=("42 requests per minute",),
                expected_conflicts=(),
                distractor_events=tuple(distractors_a),
            )
        )

        p9b = "proj-bench-needle-salt"
        needle_b = [
            _make_event("ev-nih-b", EventType.DECISION, "DECISION: Password hashing salt length is configured to 32 bytes with Argon2id variant.", p9b, "agent-sec", "sess-9b", base_time),
        ]
        distractors_b = [
            _make_event(f"ev-dist-b{i}", EventType.AGENT_MESSAGE, f"The project uses build artifact number {1000 + i} stored in S3 bucket {i}.", p9b, "agent-ci", "sess-9b", base_time + timedelta(minutes=1 + i))
            for i in range(15)
        ]
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_09B_NEEDLE_IN_HAYSTACK",
                name="Retrieve Password Salt Length Among Build Logs",
                category=ScenarioCategory.NEEDLE_IN_HAYSTACK,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.SYNTHETIC,
                description="Locate the 32-byte Argon2id salt specification amid 15 build artifact logs.",
                historical_events=tuple(needle_b),
                task_prompt="What is the required password hashing salt size and algorithm variant?",
                project_id=p9b,
                expected_facts=("32 bytes", "Argon2id"),
                expected_decisions=("Password hashing salt length is configured to 32 bytes",),
                expected_constraints=("Argon2id",),
                expected_conflicts=(),
                distractor_events=tuple(distractors_b),
            )
        )

        p9c = "proj-bench-needle-port"
        needle_c = [
            _make_event("ev-nih-c", EventType.DECISION, "DECISION: Prometheus metrics scraping endpoint runs on private internal port 9876.", p9c, "agent-devops", "sess-9c", base_time),
        ]
        distractors_c = [
            _make_event(f"ev-dist-c{i}", EventType.AGENT_MESSAGE, f"The project uses database connection metric count {i * 10} on test instance {i}.", p9c, "agent-metric", "sess-9c", base_time + timedelta(minutes=1 + i))
            for i in range(15)
        ]
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_09C_NEEDLE_IN_HAYSTACK",
                name="Retrieve Prometheus Metrics Port Among Telemetry Noise",
                category=ScenarioCategory.NEEDLE_IN_HAYSTACK,
                difficulty=ScenarioDifficulty.HARD,
                provenance=ScenarioProvenance.SYNTHETIC,
                description="Find the private internal port 9876 among 15 database metric records.",
                historical_events=tuple(needle_c),
                task_prompt="What internal port must the Prometheus metrics server bind to?",
                project_id=p9c,
                expected_facts=("port 9876",),
                expected_decisions=("endpoint runs on private internal port 9876",),
                expected_constraints=("port 9876",),
                expected_conflicts=(),
                distractor_events=tuple(distractors_c),
            )
        )

        # =====================================================================
        # 10. STALE_KNOWLEDGE_REJECTION (3 Scenarios)
        # =====================================================================
        p10a = "proj-bench-stale"
        s10a_events = (
            _make_event("ev-sk-1", EventType.DECISION, "DECISION: Temporary token format is MD5 hex string.", p10a, "agent-dev", "sess-10a", base_time),
            _make_event("ev-sk-2", EventType.DECISION, "DECISION: Supersede MD5 tokens. Permanent production token format is OAuth2 Bearer with Ed25519 signatures.", p10a, "agent-sec", "sess-10b", base_time + timedelta(days=10)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_10A_STALE_KNOWLEDGE_REJECTION",
                name="Reject Outdated MD5 Token in Favor of Ed25519",
                category=ScenarioCategory.STALE_KNOWLEDGE_REJECTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must use the current OAuth2 Bearer Ed25519 token format and reject the stale MD5 format.",
                historical_events=s10a_events,
                task_prompt="Generate the token parsing middleware for incoming API requests.",
                project_id=p10a,
                expected_facts=("OAuth2 Bearer", "Ed25519"),
                expected_decisions=("OAuth2 Bearer with Ed25519 signatures",),
                expected_constraints=("Ed25519",),
                expected_conflicts=(),
                conflict_ground_truth=ConflictGroundTruth.TEMPORAL_UPDATE,
            )
        )

        p10b = "proj-bench-stale-python"
        s10b_events = (
            _make_event("ev-sk-b1", EventType.DECISION, "DECISION: Service runtime is Python 3.9 on Debian bullseye.", p10b, "agent-dev", "sess-10b1", base_time),
            _make_event("ev-sk-b2", EventType.DECISION, "DECISION: Supersede Python 3.9 runtime. Upgraded production runtime to Python 3.12 on Debian bookworm.", p10b, "agent-devops", "sess-10b2", base_time + timedelta(days=20)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_10B_STALE_KNOWLEDGE_REJECTION",
                name="Reject Deprecated Python 3.9 Runtime in Favor of 3.12",
                category=ScenarioCategory.STALE_KNOWLEDGE_REJECTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent builds Dockerfile targeting Python 3.12, rejecting superseded Python 3.9 base image.",
                historical_events=s10b_events,
                task_prompt="Write the production Dockerfile base image definition.",
                project_id=p10b,
                expected_facts=("Python 3.12", "Debian bookworm"),
                expected_decisions=("Python 3.12 on Debian bookworm",),
                expected_constraints=("Python 3.12",),
                expected_conflicts=(),
                conflict_ground_truth=ConflictGroundTruth.TEMPORAL_UPDATE,
            )
        )

        p10c = "proj-bench-stale-cdn"
        s10c_events = (
            _make_event("ev-sk-c1", EventType.DECISION, "DECISION: User avatar uploads are fetched directly from S3 bucket asset-storage.s3.amazonaws.com.", p10c, "agent-dev", "sess-10c1", base_time),
            _make_event("ev-sk-c2", EventType.DECISION, "DECISION: Supersede direct S3 URLs. All avatar images must be served via CloudFront CDN distribution https://cdn.example.com.", p10c, "agent-arch", "sess-10c2", base_time + timedelta(days=15)),
        )
        self.add_scenario(
            BenchmarkScenario(
                scenario_id="SCENARIO_10C_STALE_KNOWLEDGE_REJECTION",
                name="Reject Legacy S3 Direct Endpoint in Favor of CloudFront CDN",
                category=ScenarioCategory.STALE_KNOWLEDGE_REJECTION,
                difficulty=ScenarioDifficulty.MEDIUM,
                provenance=ScenarioProvenance.MANUALLY_AUTHORED,
                description="Agent must use CloudFront CDN URL https://cdn.example.com and reject legacy S3 direct endpoint.",
                historical_events=s10c_events,
                task_prompt="Generate the avatar image URL generator function.",
                project_id=p10c,
                expected_facts=("https://cdn.example.com", "CloudFront CDN"),
                expected_decisions=("served via CloudFront CDN distribution",),
                expected_constraints=("https://cdn.example.com",),
                expected_conflicts=(),
                conflict_ground_truth=ConflictGroundTruth.TEMPORAL_UPDATE,
            )
        )
