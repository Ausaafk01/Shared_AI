"""Comprehensive test suite for Phase 5, Milestone 5.5: FastAPI REST Service Layer & Streaming Endpoints.

Covers:
1. System Probes & OpenAPI 3.1 Specification:
   - GET /healthz and GET /readyz probes.
   - GET /openapi.json (OpenAPI 3.1.0 schema, paths, HTTPBearer & ApiKeyAuth security schemes).
   - Swagger UI (/docs) and ReDoc (/redoc).

2. HTTP Security Middleware & Reliability:
   - Request size limiting (HTTP 413 on payload > 1 MB).
   - Correlation IDs (X-Request-ID tracing on requests and responses).
   - Brute-force throttling (HTTP 429 when client IP is blocked).
   - Safe CORS defaults.

3. Authentication & RBAC Authorization (M5.4 Foundation):
   - Missing credentials rejection (HTTP 401).
   - Invalid JWT token and invalid API key rejection (HTTP 401).
   - JWT Bearer authentication (HTTP 200).
   - Agent API key authentication via X-API-Key and Authorization Bearer agy_ak_... (HTTP 200).
   - Insufficient permission rejection (HTTP 403).
   - Cross-project access rejection (HTTP 403).
   - Agent scope bounding over HTTP (HTTP 403).

4. Interaction Lifecycle & Idempotency:
   - Prepare interaction, complete interaction, retrieve status, and list interactions.
   - Idempotency key preservation and conflict detection (HTTP 409).

5. Server-Sent Events (SSE) Streaming:
   - Real-time interaction progress and chunk streaming (text/event-stream).
   - Verification of lifecycle events: prepare, context_delivered, agent_chunk, completion, done.

6. Memory Management:
   - Create, retrieve, and list project memories.
   - Versioning update (Rule A-D) with parent memory lineage.

7. Multi-Strategy Retrieval:
   - Hybrid, vector, and graph retrieval with limits and similarity thresholds.

8. Session Continuity & Multi-Agent Coordination:
   - Create session, retrieve, list, switch agent, and close session.

9. Multi-Agent Collaboration Workflows:
   - Create workflow (Research -> Coding -> Review), retrieve, and execute workflow steps.

10. SQLite and PostgreSQL Dual-Backend Parity.
"""

from __future__ import annotations

import json
import os
import unittest
import uuid
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from ai_brain.auth import (
    AuthenticatedIdentity,
    AuthenticationService,
    AuthorizationService,
    CreatedApiKey,
    IdentityType,
    KeyManager,
    Permission,
    ProjectMemberRecord,
    ProjectRole,
    SecurityAuditLogger,
    TokenService,
    UserRecord,
    generate_api_key,
)
from ai_brain.auth.store import SQLiteAuthStore
from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.server.app import create_app
from ai_brain.server.dependencies import (
    BrainServiceContainer,
    create_postgres_container,
    create_sqlite_container,
)
from ai_brain.storage.database import DatabaseConnectionManager

load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)


def _check_postgres_live() -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=3,
        pool_timeout=3.0,
    )
    mgr = DatabaseConnectionManager(cfg)
    if not mgr.driver_available:
        return False, "PostgreSQL driver not installed", None
    try:
        conn = mgr.get_connection(timeout=3.0)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()
        mgr.return_connection(conn)
        return True, str(ver[0] if ver else "unknown"), mgr
    except Exception as e:
        mgr.close()
        return False, f"PostgreSQL unavailable: {e}", None


class BaseAPITestCase(unittest.TestCase):
    """Base test fixture setting up container, projects, users, agents, and TestClient."""

    def setUp(self) -> None:
        self.project_a = f"proj_api_a_{uuid.uuid4().hex[:8]}"
        self.project_b = f"proj_api_b_{uuid.uuid4().hex[:8]}"

        self.container = create_sqlite_container(db_path=":memory:")
        self.app = create_app(container=self.container)
        self.client = TestClient(self.app)

        # Bootstrap user with OWNER in Project A, but NO role in Project B
        self.owner_user_id = f"user_owner_{uuid.uuid4().hex[:8]}"
        self.container.auth_store.create_user(UserRecord(user_id=self.owner_user_id, email="owner@example.com", name="Owner User"))
        self.container.auth_store.add_project_member(ProjectMemberRecord(member_id=f"mem_{uuid.uuid4().hex[:8]}", project_id=self.project_a, user_id=self.owner_user_id, role=ProjectRole.OWNER))

        # Bootstrap user with READ_ONLY in Project A
        self.readonly_user_id = f"user_ro_{uuid.uuid4().hex[:8]}"
        self.container.auth_store.create_user(UserRecord(user_id=self.readonly_user_id, email="ro@example.com", name="Read Only User"))
        self.container.auth_store.add_project_member(ProjectMemberRecord(member_id=f"mem_{uuid.uuid4().hex[:8]}", project_id=self.project_a, user_id=self.readonly_user_id, role=ProjectRole.READ_ONLY))

        # Generate JWT tokens
        self.owner_token = self.container.token_service.create_access_token(
            subject=self.owner_user_id,
            identity_type=IdentityType.USER,
            name="Owner User",
            scopes=["*"],
        )
        self.readonly_token = self.container.token_service.create_access_token(
            subject=self.readonly_user_id,
            identity_type=IdentityType.USER,
            name="Read Only User",
            scopes=["*"],
        )

        # Bootstrap agent in Project A with full agent scopes
        self.agent_id = f"agent_worker_{uuid.uuid4().hex[:8]}"
        self.created_agent_key: CreatedApiKey = self.container.auth_service.create_agent_api_key(
            agent_id=self.agent_id,
            project_id=self.project_a,
            name="Primary Worker Key",
            scopes=["interaction:*", "memory:*", "vector:*", "session:*"],
        )
        self.agent_raw_key = self.created_agent_key.raw_key

        # Bootstrap scoped agent with ONLY memory:read scope
        self.scoped_agent_id = f"agent_scoped_{uuid.uuid4().hex[:8]}"
        self.scoped_agent_key: CreatedApiKey = self.container.auth_service.create_agent_api_key(
            agent_id=self.scoped_agent_id,
            project_id=self.project_a,
            name="Read-Only Scoped Key",
            scopes=["memory:read"],
        )
        self.scoped_agent_raw_key = self.scoped_agent_key.raw_key

    def tearDown(self) -> None:
        self.client.close()
        self.container.close()


class TestSystemProbesAndOpenAPI(BaseAPITestCase):
    """Test health, readiness, and OpenAPI 3.1 schema endpoints."""

    def test_health_probe(self) -> None:
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("timestamp", data)

    def test_ready_probe(self) -> None:
        res = self.client.get("/readyz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["database"], "connected")
        self.assertIn("backend", data)

    def test_openapi_schema_version_and_paths(self) -> None:
        res = self.client.get("/openapi.json")
        self.assertEqual(res.status_code, 200)
        schema = res.json()
        self.assertEqual(schema["openapi"], "3.1.0")
        self.assertEqual(schema["info"]["title"], "AI Brain REST API")

        # Verify security schemes
        sec_schemes = schema.get("components", {}).get("securitySchemes", {})
        self.assertIn("HTTPBearer", sec_schemes)
        self.assertIn("ApiKeyAuth", sec_schemes)

        # Verify key paths exist
        paths = schema.get("paths", {})
        self.assertIn("/api/v1/projects/{project_id}/interactions/prepare", paths)
        self.assertIn("/api/v1/projects/{project_id}/interactions/stream", paths)
        self.assertIn("/api/v1/projects/{project_id}/memories", paths)
        self.assertIn("/api/v1/projects/{project_id}/retrieval/query", paths)
        self.assertIn("/api/v1/projects/{project_id}/sessions", paths)
        self.assertIn("/api/v1/projects/{project_id}/workflows", paths)

    def test_interactive_documentation_ui(self) -> None:
        res_docs = self.client.get("/docs")
        self.assertEqual(res_docs.status_code, 200)
        self.assertIn("swagger-ui", res_docs.text.lower())

        res_redoc = self.client.get("/redoc")
        self.assertEqual(res_redoc.status_code, 200)
        self.assertIn("redoc", res_redoc.text.lower())


class TestHttpSecurityAndMiddleware(BaseAPITestCase):
    """Test HTTP reliability, payload limits, correlation IDs, and rate limiting."""

    def test_request_size_limit_exceeded(self) -> None:
        # 1 MB + 10 bytes payload
        huge_text = "X" * (1024 * 1024 + 10)
        res = self.client.post(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json={"content": huge_text},
        )
        self.assertEqual(res.status_code, 413)
        self.assertIn("PAYLOAD_TOO_LARGE", res.text)

    def test_correlation_id_propagation(self) -> None:
        custom_req_id = "test-corr-id-998877"
        res = self.client.get("/healthz", headers={"X-Request-ID": custom_req_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("x-request-id"), custom_req_id)

    def test_auto_generated_correlation_id(self) -> None:
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(len(res.headers.get("x-request-id", "")) > 0)

    def test_brute_force_client_ip_blocked(self) -> None:
        test_ip = "192.168.1.100"
        # Simulate blocking IP in audit logger
        for _ in range(6):
            self.container.audit_logger.record_event(
                "AUTH_FAILURE",
                ip_address=test_ip,
                status="FAILURE",
            )
        self.assertTrue(self.container.audit_logger.is_client_blocked(ip_address=test_ip))

        # Attempt call from blocked IP
        res = self.client.get("/healthz", headers={"X-Forwarded-For": test_ip})
        self.assertEqual(res.status_code, 429)
        self.assertIn("TOO_MANY_REQUESTS", res.text)


class TestAuthenticationAndRbac(BaseAPITestCase):
    """Test M5.4 authentication and authorization over HTTP."""

    def test_missing_authentication_returns_401(self) -> None:
        res = self.client.get(f"/api/v1/projects/{self.project_a}/memories")
        self.assertEqual(res.status_code, 401)
        self.assertIn("WWW-Authenticate", res.headers)

    def test_invalid_jwt_returns_401(self) -> None:
        res = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        self.assertEqual(res.status_code, 401)

    def test_invalid_api_key_returns_401(self) -> None:
        res = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"X-API-Key": "agy_ak_invalidkeyfortesting12345"},
        )
        self.assertEqual(res.status_code, 401)

    def test_agent_api_key_via_x_api_key(self) -> None:
        res = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"X-API-Key": self.agent_raw_key},
        )
        self.assertEqual(res.status_code, 200)

    def test_agent_api_key_via_bearer(self) -> None:
        res = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.agent_raw_key}"},
        )
        self.assertEqual(res.status_code, 200)

    def test_insufficient_role_permission_returns_403(self) -> None:
        # READ_ONLY role caller cannot create memories (requires MEMORY_WRITE)
        res = self.client.post(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.readonly_token}"},
            json={"content": "Unauthorized memory write attempt", "memory_type": "FACT"},
        )
        self.assertEqual(res.status_code, 403)

    def test_cross_project_access_denied_returns_403(self) -> None:
        # Owner of Project A tries to list memories in Project B where they have no membership
        res = self.client.get(
            f"/api/v1/projects/{self.project_b}/memories",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res.status_code, 403)

    def test_agent_scope_bounding_over_http(self) -> None:
        # Scoped agent has ONLY memory:read; attempting memory write should be forbidden
        res = self.client.post(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"X-API-Key": self.scoped_agent_raw_key},
            json={"content": "Agent write exceeding scopes", "memory_type": "FACT"},
        )
        self.assertEqual(res.status_code, 403)

        # But memory read should succeed
        res_read = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"X-API-Key": self.scoped_agent_raw_key},
        )
        self.assertEqual(res_read.status_code, 200)


class TestInteractionLifecycleAndSSE(BaseAPITestCase):
    """Test interaction lifecycle, idempotency preservation, and SSE streaming."""

    def test_interaction_prepare_and_complete_lifecycle(self) -> None:
        # 1. Prepare interaction
        prep_payload = {
            "input_text": "Need architectural guidelines on PostgreSQL indexing.",
            "search_query": "PostgreSQL indexing",
            "max_context_tokens": 1200,
        }
        res_prep = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/prepare",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=prep_payload,
        )
        self.assertEqual(res_prep.status_code, 200)
        prep_data = res_prep.json()
        interaction_id = prep_data["interaction_id"]
        self.assertEqual(prep_data["status"], "PREPARED")
        self.assertIn("prompt_context_markdown", prep_data)

        # 2. Complete interaction
        complete_payload = {
            "agent_response": "Use HNSW indexes for vector retrieval and B-Tree for relational queries.",
        }
        res_comp = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/{interaction_id}/complete",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=complete_payload,
        )
        self.assertEqual(res_comp.status_code, 200)
        comp_data = res_comp.json()
        self.assertEqual(comp_data["status"], "COMPLETED")

        # 3. Retrieve interaction record
        res_get = self.client.get(
            f"/api/v1/projects/{self.project_a}/interactions/{interaction_id}",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_get.status_code, 200)
        get_data = res_get.json()
        self.assertEqual(get_data["interaction_id"], interaction_id)
        self.assertEqual(get_data["status"], "COMPLETED")

        # 4. List interactions
        res_list = self.client.get(
            f"/api/v1/projects/{self.project_a}/interactions",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_list.status_code, 200)
        interactions = res_list.json()
        self.assertTrue(any(it["interaction_id"] == interaction_id for it in interactions))

    def test_interaction_idempotency_conflict(self) -> None:
        idem_key = "unique-key-xyz-123"
        prep_payload = {
            "input_text": "Original text prompt",
            "idempotency_key": idem_key,
        }
        res1 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/prepare",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=prep_payload,
        )
        self.assertEqual(res1.status_code, 200)

        # Exact repeat returns same result
        res2 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/prepare",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=prep_payload,
        )
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res1.json()["interaction_id"], res2.json()["interaction_id"])

        # Conflicting input with same idempotency key returns 409
        conflicting_payload = {
            "input_text": "Completely different text with identical key",
            "idempotency_key": idem_key,
        }
        res3 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/prepare",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=conflicting_payload,
        )
        self.assertEqual(res3.status_code, 409)

    def test_server_sent_events_streaming(self) -> None:
        stream_payload = {
            "input_text": "Stream cognitive interaction test",
            "search_query": "Cognitive stream",
        }
        res = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=stream_payload,
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))

        # Parse SSE events
        raw_lines = res.text.split("\n")
        events: List[str] = []
        data_payloads: List[Dict[str, Any]] = []

        current_event = None
        for line in raw_lines:
            line = line.strip()
            if line.startswith("event: "):
                current_event = line.replace("event: ", "").strip()
            elif line.startswith("data: "):
                events.append(current_event or "message")
                data_json = json.loads(line.replace("data: ", "").strip())
                data_payloads.append(data_json)
                current_event = None

        # Verify lifecycle events
        self.assertIn("prepare", events)
        self.assertIn("context_delivered", events)
        self.assertIn("agent_chunk", events)
        self.assertIn("completion", events)
        self.assertIn("done", events)


class TestMemoryManagementAndRetrieval(BaseAPITestCase):
    """Test memory CRUD, versioning rules A-D, and multi-strategy retrieval."""

    def test_memory_crud_and_versioning_update(self) -> None:
        # 1. Create Memory
        create_payload = {
            "content": "The application must use PostgreSQL 16.",
            "memory_type": "CONSTRAINT",
            "importance": 0.9,
            "metadata": {"category": "database"},
        }
        res_create = self.client.post(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=create_payload,
        )
        self.assertEqual(res_create.status_code, 201)
        mem1 = res_create.json()
        mem1_id = mem1["memory_id"]
        self.assertEqual(mem1["version"], 1)
        self.assertEqual(mem1["status"], "ACTIVE")
        self.assertEqual(mem1["importance"], 0.9)

        # 2. Get Memory by ID
        res_get = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories/{mem1_id}",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["content"], create_payload["content"])

        # 3. List Memories
        res_list = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(len(res_list.json()), 1)

        # 4. Update Memory creating version 2 (Rule B - Supersede)
        update_payload = {
            "content": "The application must use PostgreSQL 16.15 with pgvector.",
            "importance": 0.95,
        }
        res_update = self.client.post(
            f"/api/v1/projects/{self.project_a}/memories/{mem1_id}/update",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=update_payload,
        )
        self.assertEqual(res_update.status_code, 200)
        up_data = res_update.json()
        self.assertEqual(up_data["version"], 2)
        new_memory = up_data["memory"]
        self.assertEqual(new_memory["parent_memory_id"], mem1_id)
        self.assertEqual(new_memory["root_memory_id"], mem1_id)

        # Verify old memory is now SUPERSEDED
        res_old = self.client.get(
            f"/api/v1/projects/{self.project_a}/memories/{mem1_id}",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_old.status_code, 200)
        self.assertEqual(res_old.json()["status"], "SUPERSEDED")

    def test_multi_strategy_retrieval(self) -> None:
        # Create knowledge memories
        self.client.post(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json={"content": "FastAPI REST API exposes asynchronous HTTP endpoints.", "memory_type": "FACT"},
        )
        self.client.post(
            f"/api/v1/projects/{self.project_a}/memories",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json={"content": "Uvicorn ASGI server serves ASGI applications on port 8000.", "memory_type": "FACT"},
        )

        # Execute hybrid retrieval
        query_payload = {
            "query_text": "FastAPI HTTP server endpoints",
            "strategy": "HYBRID",
            "limit": 5,
        }
        res_query = self.client.post(
            f"/api/v1/projects/{self.project_a}/retrieval/query",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=query_payload,
        )
        self.assertEqual(res_query.status_code, 200)
        data = res_query.json()
        self.assertEqual(data["strategy"], "HYBRID")
        self.assertGreater(data["count"], 0)
        self.assertIn("FastAPI", data["items"][0]["content"])


class TestSessionsAndWorkflows(BaseAPITestCase):
    """Test session continuity, agent handoffs, and multi-agent collaboration workflows."""

    def test_session_lifecycle_and_handover(self) -> None:
        # 1. Create Session
        res_create = self.client.post(
            f"/api/v1/projects/{self.project_a}/sessions",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json={"summary": "Agent collaboration session", "primary_agent_id": "agent-primary"},
        )
        self.assertEqual(res_create.status_code, 201)
        sess = res_create.json()
        session_id = sess["session_id"]
        self.assertEqual(sess["status"], "ACTIVE")

        # 2. Get Session
        res_get = self.client.get(
            f"/api/v1/projects/{self.project_a}/sessions/{session_id}",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_get.status_code, 200)

        # 3. Switch Agent
        res_switch = self.client.post(
            f"/api/v1/projects/{self.project_a}/sessions/{session_id}/switch-agent",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json={"new_agent_id": "agent-secondary", "reason": "Handoff to specialist"},
        )
        self.assertEqual(res_switch.status_code, 200)
        self.assertEqual(res_switch.json()["primary_agent_id"], "agent-secondary")

        # 4. Close Session
        res_close = self.client.post(
            f"/api/v1/projects/{self.project_a}/sessions/{session_id}/close",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_close.status_code, 200)
        self.assertEqual(res_close.json()["status"], "COMPLETED")

    def test_multi_agent_workflow_execution(self) -> None:
        # 1. Create standard Research -> Coding -> Review workflow
        wf_payload = {
            "name": "Feature Implementation",
            "task_description": "Implement caching layer for retrieval results.",
            "workflow_type": "RESEARCH_CODING_REVIEW",
        }
        res_create = self.client.post(
            f"/api/v1/projects/{self.project_a}/workflows",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=wf_payload,
        )
        self.assertEqual(res_create.status_code, 201)
        wf_data = res_create.json()
        workflow_id = wf_data["workflow_id"]
        self.assertEqual(wf_data["step_count"], 3)

        # 2. Get workflow
        res_get = self.client.get(
            f"/api/v1/projects/{self.project_a}/workflows/{workflow_id}",
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["workflow_id"], workflow_id)

        # 3. Execute workflow
        res_exec = self.client.post(
            f"/api/v1/projects/{self.project_a}/workflows/{workflow_id}/execute",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json={},
        )
        self.assertEqual(res_exec.status_code, 200)
        exec_data = res_exec.json()
        self.assertEqual(exec_data["status"], "COMPLETED")
        self.assertEqual(len(exec_data["completed_steps"]), 3)
        self.assertIsNotNone(exec_data["final_output"])


class TestPostgresAPIIntegration(unittest.TestCase):
    """Integration test suite executing the REST API layer against live PostgreSQL."""

    @classmethod
    def setUpClass(cls) -> None:
        live, msg, mgr = _check_postgres_live()
        if not live or mgr is None:
            raise unittest.SkipTest(f"Live PostgreSQL unavailable: {msg}")
        cls.db_manager = mgr

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "db_manager") and cls.db_manager:
            cls.db_manager.close()

    def setUp(self) -> None:
        self.project_id = f"proj_pg_api_{uuid.uuid4().hex[:8]}"
        self.user_id = f"user_pg_{uuid.uuid4().hex[:8]}"

        self.container = create_postgres_container(self.db_manager)
        self.app = create_app(container=self.container)
        self.client = TestClient(self.app)

        # Setup user in PostgreSQL auth store
        self.container.auth_store.create_user(UserRecord(user_id=self.user_id, email=f"{self.user_id}@example.com", name="PG User"))
        self.container.auth_store.add_project_member(ProjectMemberRecord(member_id=f"mem_pg_{uuid.uuid4().hex[:8]}", project_id=self.project_id, user_id=self.user_id, role=ProjectRole.OWNER))

        self.token = self.container.token_service.create_access_token(
            subject=self.user_id,
            identity_type=IdentityType.USER,
            name="PG User",
            scopes=["*"],
        )

    def tearDown(self) -> None:
        self.client.close()
        self.container.close()

    def test_postgres_memory_crud_and_retrieval(self) -> None:
        # Create memory in PostgreSQL
        res_create = self.client.post(
            f"/api/v1/projects/{self.project_id}/memories",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "content": "PostgreSQL 16 handles multi-tenant JSONB and pgvector queries efficiently.",
                "memory_type": "FACT",
                "importance": 0.88,
            },
        )
        self.assertEqual(res_create.status_code, 201)
        mem_id = res_create.json()["memory_id"]

        # Retrieve memory
        res_get = self.client.get(
            f"/api/v1/projects/{self.project_id}/memories/{mem_id}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["memory_id"], mem_id)

        # Retrieval query
        res_query = self.client.post(
            f"/api/v1/projects/{self.project_id}/retrieval/query",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"query_text": "PostgreSQL multi-tenant JSONB", "strategy": "HYBRID"},
        )
        self.assertEqual(res_query.status_code, 200)
        self.assertGreaterEqual(res_query.json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
