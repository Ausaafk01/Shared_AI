# AI Brain — Phase 5: Production Backend & Infrastructure Implementation Plan

**Document Version:** 2.0.0  
**Status:** Architecture Approved; Milestone 5.1 Authorized for Implementation  
**Baseline Verification:** Phase 1–4 Stable Baseline (`4283ede`, 343 tests collected: 339 passed, 4 skipped)  

---

## 1. Execution Principles & Milestone Roadmap

Phase 5 execution is broken down into **seven modular, independently testable milestones**. Each milestone introduces a discrete layer of production infrastructure, maintains backward compatibility with the existing Phase 1–4 SQLite baseline, and delivers complete unit and integration tests.

```
                         PHASE 5 MILESTONE ROADMAP
                         
   +---------------------------------------------------------------+
   | M5.1: Relational Schema & Migration Infrastructure (Alembic)   |
   |       (14 Tables, Partitioning, Write-Once, RLS Foundations)  |
   +-------------------------------+-------------------------------+
                                   |
                                   v
   +---------------------------------------------------------------+
   | M5.2: Persistence Abstractions & Storage Adapters             |
   |       (MemoryStore & GraphStore ABCs, PostgreSQL Adapters)    |
   +-------------------------------+-------------------------------+
                                   |
                                   v
   +---------------------------------------------------------------+
   | M5.3: Multi-Model Vector Storage & HNSW Hybrid Retrieval      |
   |       (384 / 768 / 1536 Dim Vector Tables, RRF Fusion)        |
   +-------------------------------+-------------------------------+
                                   |
                                   v
   +---------------------------------------------------------------+
   | M5.4: Authentication, RBAC & Multi-Tenant Gateway             |
   |       (JWT, Argon2id/SHA-256 API Keys, Untrusted Memory Guard)|
   +-------------------------------+-------------------------------+
                                   |
                                   v
   +---------------------------------------------------------------+
   | M5.5: FastAPI REST Service Layer & Streaming Endpoints        |
   |       (OpenAPI 3.1, SSE Streams, Tenant Middleware)           |
   +-------------------------------+-------------------------------+
                                   |
                                   v
   +---------------------------------------------------------------+
   | M5.6: Durable Asynchronous Background Job Worker Fleet        |
   |       (Leases, Retries, Exponential Backoff, Dead-Lettering)  |
   +-------------------------------+-------------------------------+
                                   |
                                   v
   +---------------------------------------------------------------+
   | M5.7: Docker Packaging, Observability & Baseline Verification |
   |       (Prometheus, Structured Logs, 343/343 Regression Pass)  |
   +-------------------------------+-------------------------------+
```

---

## 2. Milestone Breakdown & Specification

### Milestone 5.1: Relational Schema & Migration Infrastructure (Alembic) [CURRENT]
- **Scope & Objectives**:
  - Establish central production configuration management (`ai_brain/config.py`) using environment variables with validation and safe SQLite defaults for offline execution.
  - Establish PostgreSQL connection manager and pool abstraction (`ai_brain/storage/database.py`) supporting connection checkout sanitization (`RESET ALL`) and transaction-scoped tenant isolation (`SET LOCAL app.current_project_id = ...`).
  - Set up Alembic migration framework (`alembic/`, `alembic.ini`) configured for PostgreSQL 16+.
  - Create initial declarative schema migration creating all **14 authoritative core tables**:
    1. `projects`
    2. `project_members`
    3. `events` (Range-partitioned by `timestamp`)
    4. `event_lookup` (Global primary key uniqueness and partition routing)
    5. `memories`
    6. `graph_nodes`
    7. `graph_relationships`
    8. `memory_vectors_384` (Default 384-dim vector table with HNSW index)
    9. `agents`
    10. `agent_credentials` (Hashed API keys with `revoked_at` and scope)
    11. `sessions`
    12. `interactions`
    13. `feedback_records`
    14. `background_tasks` (Worker queue with leases and dead-letter status)
  - Implement engine-level append-only rules for `events` (`no_update_events`, `no_delete_events`).
  - Implement Row-Level Security (RLS) policies on all tenant-owned tables (`USING (project_id = current_setting('app.current_project_id'))`).
- **Dependencies**: None.
- **Files Created / Modified**:
  - `ai_brain/config.py` (Production configuration dataclass & settings loader)
  - `ai_brain/storage/database.py` (PostgreSQL connection manager & pool helper)
  - `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`, `alembic/versions/001_m5_1_initial_schema.py`
  - `scripts/init_postgres_schema.sql` (Standalone DDL script for direct psql initialization)
  - `tests/test_postgres_m5_1.py` (M5.1 migration, schema, rule, RLS, and transaction rollback tests)
- **Database Changes**: Creates core relational tables with foreign keys, indexes, partitioning, and RLS policies.
- **Acceptance Criteria & Tests**:
  - `alembic upgrade head` runs cleanly on fresh PostgreSQL.
  - `alembic downgrade base` tears down cleanly, and re-upgrade succeeds.
  - All 14 tables, constraints, foreign keys, and indexes exist.
  - Engine-level write-once rules reject any `UPDATE` or `DELETE` on `events`.
  - Global `event_lookup` enforces primary key uniqueness and idempotency across partitions.
  - Tenant isolation via RLS and `SET LOCAL app.current_project_id = ...` isolates project records.
  - Reused connection cannot retain prior tenant context (pool sanitization clears context).
  - Failed transactions roll back cleanly without partial commits.
  - If live PostgreSQL is unreachable, PostgreSQL tests skip cleanly with clear explanation; SQLite tests and all 8 offline demos continue to pass without regression.
- **Security Implications**: Verifies database credential isolation via environment variables (`DATABASE_URL`), safe tenant isolation via `SET LOCAL`, and append-only event immutability.
- **Rollback Consideration**: Revert migration via `alembic downgrade base`.

---

### Milestone 5.2: Persistence Abstractions & Storage Adapters
- **Scope & Objectives**:
  - Formally define abstract base classes in the core domain:
    - `MemoryStore` ABC (`ai_brain/memories/storage/base.py`)
    - `GraphStore` ABC (`ai_brain/graph/storage/base.py`)
  - Refactor existing in-memory/SQLite memory and graph implementations to conform to these ABCs without breaking existing tests.
  - Implement concrete PostgreSQL adapters for all storage interfaces:
    - `PostgreSQLEventStore` (implements `EventStore` ABC)
    - `PostgreSQLMemoryStore` (implements `MemoryStore` ABC)
    - `PostgreSQLGraphStore` (implements `GraphStore` ABC)
    - `PostgreSQLInteractionStore` (implements `InteractionStore` ABC)
    - `PostgreSQLAgentRegistryStore` (implements `AgentRegistryStore` ABC)
    - `PostgreSQLSessionStore` (implements `SessionStore` ABC)
    - `PostgreSQLFeedbackStore` (implements `FeedbackStore` ABC)
  - Maintain strict project tenant isolation in every SQL query.
  - Implement PostgreSQL session/transaction advisory locks (`pg_advisory_xact_lock`) for race-free memory version chain increments.
- **Dependencies**: Milestone 5.1.
- **Files Expected to Create / Modify**:
  - `ai_brain/memories/storage/base.py`, `ai_brain/memories/storage/postgres.py`
  - `ai_brain/graph/storage/base.py`, `ai_brain/graph/storage/postgres.py`
  - `ai_brain/events/storage/postgres.py`
  - `ai_brain/loop/storage/postgres.py`
  - `ai_brain/agents/registry/postgres.py`
  - `ai_brain/agents/sessions/storage/postgres.py`
  - `ai_brain/ml/feedback/postgres.py`
  - `tests/test_postgres_storage.py`
- **Database Changes**: None (uses M5.1 schema).
- **Acceptance Criteria & Tests**:
  - Complete parity with SQLite test suites: All operations (CRUD, filtering, chronological ordering, idempotency, dispute preservation) pass against PostgreSQL.
  - Concurrent version increment test: Two parallel threads attempting to update the same memory resolve deterministically without version chain corruption.
- **Security Implications**: All queries use parameterized statements (`$1, $2`) to eliminate SQL injection vulnerabilities.
- **Rollback Consideration**: Switch configuration `STORAGE_BACKEND=sqlite` to revert to local embedded operation.

---

### Milestone 5.3: Multi-Model Vector Storage & HNSW Hybrid Retrieval
- **Scope & Objectives**:
  - Implement `PgVectorStore` (implements `VectorStore` ABC) using `pgvector` with HNSW cosine distance indexing (`vector_cosine_ops`).
  - Support multi-model vector tables (`memory_vectors_384`, `memory_vectors_768`, `memory_vectors_1536`) with dynamic model routing.
  - Enable relational pre-filtering in vector queries (`WHERE project_id = $1 AND status = 'ACTIVE'`).
  - Integrate `PgVectorStore` seamlessly into `VectorIndex` and `RetrievalEngine`.
- **Dependencies**: Milestone 5.2.
- **Files Expected to Create / Modify**:
  - `ai_brain/vectors/storage/pgvector.py`
  - `alembic/versions/002_m5_3_vector_tables.py` (Add 768 and 1536 dimension tables if needed)
  - `tests/test_pgvector.py`
- **Database Changes**: Builds HNSW index on `memory_vectors_384`.
- **Acceptance Criteria & Tests**:
  - Vector similarity rankings in `PgVectorStore` match existing `MockEmbeddingProvider` cosine calculations.
  - Superseded and archived memories are filtered at the database level before computing cosine top-$k$.
  - Hybrid RRF Fusion ($k=60$) correctly merges vector results from `PgVectorStore` and graph traversals.
- **Security Implications**: Verifies cross-project vectors are excluded at the SQL index level.
- **Rollback Consideration**: VectorStore interface permits switching back to `SQLiteVectorStore`.

---

### Milestone 5.4: Authentication, RBAC & Multi-Tenant Gateway
- **Scope & Objectives**:
  - Implement cryptographic JWT token issuer and verification middleware (RS256 / Ed25519).
  - Implement agent API key generation (`agy_sk_...`) with irreversible Argon2id / SHA-256 storage, `revoked_at` revocation checking, and timing-safe comparison.
  - Implement project-level RBAC authorization middleware validating incoming caller identities against `project_members`.
  - Implement defense-in-depth secret scrubbing to sanitize sensitive credentials before persistence.
  - Implement untrusted memory prompt injection mitigations: structural XML isolation, explicit framing instructions, control-tag escaping, and strict tool argument validation.
- **Dependencies**: Milestone 5.1, Milestone 5.2.
- **Files Expected to Create / Modify**:
  - `ai_brain/auth/jwt.py` (Token verification & public key decoding)
  - `ai_brain/auth/api_keys.py` (Key generator, hasher, validator, revocations)
  - `ai_brain/auth/rbac.py` (Permission matrix & role checkers)
  - `ai_brain/auth/sanitizer.py` (Credential regex scrubbers)
  - `ai_brain/context/sanitizer.py` (Memory prompt injection escaper)
  - `tests/test_auth_rbac.py`
- **Database Changes**: Uses `project_members` and `agent_credentials`.
- **Acceptance Criteria & Tests**:
  - Unauthenticated requests receive HTTP 401 Unauthorized.
  - An agent belonging to Project A attempting to query Project B receives HTTP 403 Forbidden.
  - Secret scrubber correctly strips mock API keys and bearer tokens from event payloads.
  - Prompt injection payload inside retrieved memory is escaped and neutral in prompt assembly.
- **Security Implications**: Enforces hard cryptographic tenancy boundaries and prompt injection defenses.
- **Rollback Consideration**: Disable auth middleware via test flag in local mode.

---

### Milestone 5.5: FastAPI REST Service Layer & Streaming Endpoints
- **Scope & Objectives**:
  - Build FastAPI application (`ai_brain/server/app.py`) providing complete RESTful endpoints:
    - `/api/v1/projects/{project_id}/interactions/prepare`
    - `/api/v1/projects/{project_id}/interactions/{interaction_id}/complete`
    - `/api/v1/projects/{project_id}/memories` (Query, history, version chain)
    - `/api/v1/projects/{project_id}/retrieval/query`
    - `/api/v1/projects/{project_id}/sessions` (Lifecycle & handover)
    - `/api/v1/projects/{project_id}/workflows` (Multi-stage collaboration)
  - Implement Server-Sent Events (SSE) streaming endpoint for streaming prompt context and interaction events.
  - Provide auto-generated OpenAPI 3.1 schema and interactive Swagger UI (`/docs`).
  - Map domain exceptions to standard HTTP status codes (400, 403, 404, 409, 500).
- **Dependencies**: Milestone 5.2, Milestone 5.3, Milestone 5.4.
- **Files Expected to Create / Modify**:
  - `ai_brain/server/app.py`, `ai_brain/server/dependencies.py`
  - `ai_brain/server/routers/interactions.py`
  - `ai_brain/server/routers/memories.py`
  - `ai_brain/server/routers/retrieval.py`
  - `ai_brain/server/routers/sessions.py`
  - `ai_brain/server/routers/workflows.py`
  - `ai_brain/server/schemas.py` (Pydantic DTO request/response schemas)
  - `tests/test_api_endpoints.py` (TestClient integration tests)
- **Database Changes**: None.
- **Acceptance Criteria & Tests**:
  - Full end-to-end multi-agent interaction executed via HTTP requests through `prepare` and `complete`.
  - HTTP 409 Conflict returned on duplicate idempotency key with conflicting payload.
  - Swagger UI rendered and validated against OpenAPI 3.1 specification.
- **Security Implications**: Rate-limiting middleware and CORS policies enforced.
- **Rollback Consideration**: Service layer is decoupled from domain logic; can be disabled without impacting core library.

---

### Milestone 5.6: Durable Asynchronous Background Job Worker Fleet
- **Scope & Objectives**:
  - Implement a resilient PostgreSQL-backed task queue using `background_tasks` with `FOR UPDATE SKIP LOCKED`.
  - Enforce durable worker semantics:
    - Leases via `locked_until` (heartbeat renewal every 30s).
    - Exponential backoff retries (`retry_count < max_retries`).
    - Poison-task dead-lettering (`status = 'DEAD_LETTER'`) on exceeding `max_retries`.
    - Crash recovery sweeper reclaiming expired leases (`locked_until < NOW()`).
    - Graceful shutdown draining in-flight tasks.
  - Move heavy background workloads off critical request paths:
    - Asynchronous memory extraction for batch event ingestion.
    - Asynchronous vector re-indexing.
- **Dependencies**: Milestone 5.2, Milestone 5.3.
- **Files Expected to Create / Modify**:
  - `ai_brain/workers/queue.py` (Transactional task enqueueing & dequeuing)
  - `ai_brain/workers/runner.py` (Asyncio task worker execution loop & lease renewal)
  - `ai_brain/workers/handlers/extraction.py`
  - `ai_brain/workers/handlers/indexing.py`
  - `tests/test_background_workers.py`
- **Database Changes**: Uses `background_tasks`.
- **Acceptance Criteria & Tests**:
  - Worker picks up pending extraction job, completes processing, and marks `COMPLETED`.
  - Abandoned worker task with expired `locked_until` is safely reclaimed by sweeper and retried.
  - Poison task exceeding `max_retries` transitions to `DEAD_LETTER` with recorded error.
- **Security Implications**: Worker jobs inherit project tenant scoping from task payload.
- **Rollback Consideration**: Tasks can be executed synchronously in-process via fallback setting.

---

### Milestone 5.7: Docker Packaging, Observability & Baseline Verification
- **Scope & Objectives**:
  - Create multi-stage production `Dockerfile` (distroless/slim Python 3.13 image).
  - Create `docker-compose.yml` orchestrating `api-service`, `worker-daemon`, and `postgres` (with `pgvector`).
  - Implement Prometheus metrics exporter (`/metrics`) and structured JSON logging formatter with request correlation IDs.
  - Implement health checks (`/healthz`, `/readyz`).
  - Execute full regression verification: run all 343 existing Phase 1–4 unit tests and all 8 demonstrations against the baseline.
- **Dependencies**: Milestones 5.1 through 5.6.
- **Files Expected to Create / Modify**:
  - `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  - `ai_brain/observability/logging.py`, `ai_brain/observability/metrics.py`
  - `tests/test_production_e2e.py`
- **Database Changes**: None.
- **Acceptance Criteria & Tests**:
  - `docker compose up -d` boots API, worker, and PostgreSQL successfully.
  - `curl http://localhost:8000/healthz` returns HTTP 200 `{"status": "ok"}`.
  - Full Phase 1–4 test suite (`python -W error::ResourceWarning -m unittest discover -s tests -v`) passes cleanly with 343 tests collected (339 passed, 4 skipped).
  - All 8 offline examples execute and complete without errors.
- **Security Implications**: Container runs under a dedicated unprivileged user (`nonroot:nonroot`).
- **Rollback Consideration**: Standard container image rollback via Docker tags.

---

## 3. Milestone Dependency Graph

```
                      +---------------------------------------+
                      | M5.1: Relational Schema & Migrations  |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
+---------------------------------------+   +------------------------------------+
| M5.2: Persistence ABCs & PG Adapters  |   | M5.4: Authentication & RBAC        |
+---------------------+-----------------+   +-----------------+------------------+
                      |                                       |
                      v                                       |
+---------------------------------------+                     |
| M5.3: pgvector Storage & HNSW Hybrid  |                     |
+---------------------+-----------------+                     |
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | M5.5: FastAPI REST Service Layer      |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
+---------------------------------------+   +------------------------------------+
| M5.6: Background Task Worker Fleet    |   | M5.7: Docker, Observability & Regr.|
+---------------------+-----------------+   +-----------------+------------------+
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Phase 5 Production Release Ready      |
                      +---------------------------------------+
```

---

## 4. Comprehensive Risk Register

| Risk ID | Risk Description | Severity | Likelihood | Technical Mitigation Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **R-01** | **Regression of Phase 1–4 Baselines**: Introducing PostgreSQL breaks existing offline SQLite tests or benchmark reproducibility. | **High** | **Low** | **Dual-Backend Architectural Isolation**: Abstract storage interfaces (`EventStore`, `MemoryStore`, `GraphStore`, `VectorStore`, etc.) are preserved. SQLite remains the default in tests and offline demos. CI runs the complete 343-test suite against SQLite before running PostgreSQL integration tests. |
| **R-02** | **Lock Contention on Memory Updating**: Multiple agents attempting concurrent updates to the same conceptual memory cause deadlocks. | **Medium** | **Medium** | **Optimistic Concurrency Control (OCC) + Advisory Locks**: Updates check version numbers (`WHERE memory_id = $id AND version = $v`). Transaction-scoped advisory locks on `hashtext(project_id \|\| root_memory_id)` serialize competing updates safely without deadlocks. |
| **R-03** | **pgvector Memory Exhaustion**: HNSW vector graph builds consume excessive RAM on high-dimensional vectors. | **Medium** | **Low** | **Configurable HNSW Index Parameters**: Explicitly set `m = 16`, `ef_construction = 64`. Multi-model tables segregate 384, 768, and 1536 dimensional vectors. Database memory capped in Docker configuration. |
| **R-04** | **Cross-Tenant Data Leakage**: Software defect in SQL query omits `project_id` filter, returning another project's data. | **Critical** | **Low** | **PostgreSQL Row-Level Security (RLS) + SET LOCAL**: Enforce RLS policies on all project tables (`USING (project_id = current_setting('app.current_project_id'))`). The application connection pool enforces `SET LOCAL app.current_project_id = ...` and sanitizes connections with `RESET ALL` upon return to pool. |
| **R-05** | **Worker Starvation or Poison Tasks**: Background worker crashes or encounters unrecoverable task exceptions. | **Medium** | **Medium** | **Lease Heartbeat, Exponential Backoff & Dead-Lettering**: Tasks locked with `locked_until`. Stale tasks exceeding lease timeout are reclaimed by sweeper. Tasks exceeding `max_retries` move to `DEAD_LETTER` with recorded error diagnostics. |
| **R-06** | **Credential Exposure in Logs & Storage**: Authentication tokens or agent secrets leaked into application logs or database plain text. | **High** | **Low** | **Argon2id/SHA-256 Hashing & Structured Log Redactor**: API keys stored as salted hashes; active keys track `revoked_at`. Observability middleware automatically masks `Authorization` headers, `agy_sk_...` keys, and token payloads using regex filtering before log output. |
| **R-07** | **Prompt Injection via Retrieved Memories**: Malicious instructions embedded in memory records hijacking agent execution. | **High** | **Medium** | **Structural Tagging & Instruction Boundary Escaping**: Retrieved memories wrapped in structural XML blocks with untrusted source markers and escaped delimiters; system prompts instruct LLM to treat memories strictly as factual context, not executable instructions. |

---

## 5. Proposed Test & Acceptance Matrix

| Milestone | Test Suite | Test Type | Acceptance Target |
| :--- | :--- | :---: | :--- |
| **M5.1** | `alembic upgrade / downgrade` & `tests/test_postgres_m5_1.py` | Migration & Schema Test | Clean forward and backward schema execution; all 14 tables, constraints, indexes verified; write-once rules reject event UPDATE/DELETE; RLS enforces tenant isolation; connection pool resets tenant context. Skips gracefully if PG daemon offline. |
| **M5.2** | `tests/test_postgres_storage.py` | Integration Test | 100% functional parity with SQLiteEventStore, SQLiteInteractionStore, SQLiteSessionStore, SQLiteRegistryStore; MemoryStore and GraphStore ABCs implemented. |
| **M5.3** | `tests/test_pgvector.py` | Integration Test | Exact cosine ranking agreement with MockEmbeddingProvider; status pre-filtering validated across multi-model vector tables. |
| **M5.4** | `tests/test_auth_rbac.py` | Security Test | 401 on missing token; 403 on cross-project access; timing-safe API key validation verified; prompt injection escaping confirmed. |
| **M5.5** | `tests/test_api_endpoints.py` | API Test | End-to-end `prepare` $\to$ `complete` interaction lifecycle executed over HTTP; OpenAPI schema valid. |
| **M5.6** | `tests/test_background_workers.py` | Concurrency Test | Task queue processes batch extractions; crashed worker task auto-reclaimed; poison task moved to DEAD_LETTER. |
| **M5.7** | Full Regression Suite | Regression Test | **343/343 tests collected (339 passed, 4 skipped)**; all 8 offline examples pass cleanly; Docker boot verified. |

---

## 6. Implementation Authorizations & Next Steps

1. **Approved for Implementation**: **Milestone 5.1** (Relational Schema & Migration Infrastructure).
2. **Current Constraints**:
   - Zero modifications to Phase 1–4 domain models, SQLite storage engines, or benchmarks.
   - Offline SQLite operation remains default for all existing tests and demonstrations.
   - Pure standard library implementation for core configuration and database utilities; Alembic and PostgreSQL drivers dynamically imported or executed via migrations.
   - PostgreSQL integration tests must connect to real instance if available, or skip cleanly with clear reporting when offline.
3. **Subsequent Milestones (M5.2–M5.7)**: Unimplemented and awaiting completion and review of M5.1.
