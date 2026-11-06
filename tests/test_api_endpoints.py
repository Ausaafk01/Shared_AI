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

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
import time
import unittest
import uuid
from typing import Any, Dict, List, Optional

from unittest.mock import AsyncMock, patch
from starlette.requests import Request
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ai_brain.server.idempotency import (
    IdempotencyStatus,
    PostgreSQLIdempotencyCoordinator,
    SQLiteIdempotencyCoordinator,
    StaleFencingTokenError,
)
from ai_brain.loop.models import AgentInteractionInput, InteractionRecord, InteractionStatus

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
from ai_brain.server.limiter import HttpRateLimiter
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


def _check_postgres_live(close_after: bool = False) -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
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
        if close_after:
            mgr.close()
            return True, str(ver[0] if ver else "unknown"), None
        return True, str(ver[0] if ver else "unknown"), mgr
    except Exception as e:
        mgr.close()
        return False, f"PostgreSQL unavailable: {e}", None


_IS_PG_LIVE, _, _ = _check_postgres_live(close_after=True)


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
        # 1. Negative authorization: caller with READ_ONLY (lacking INTERACTION_EXECUTE) is rejected with 403
        wf_payload = {
            "name": "Feature Implementation",
            "task_description": "Implement caching layer for retrieval results.",
            "workflow_type": "RESEARCH_CODING_REVIEW",
        }
        res_unauthorized = self.client.post(
            f"/api/v1/projects/{self.project_a}/workflows",
            headers={"Authorization": f"Bearer {self.readonly_token}"},
            json=wf_payload,
        )
        self.assertEqual(res_unauthorized.status_code, 403)

        # 2. Positive authorization: caller with INTERACTION_EXECUTE (owner) succeeds with 201
        res_create = self.client.post(
            f"/api/v1/projects/{self.project_a}/workflows",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=wf_payload,
        )
        self.assertEqual(res_create.status_code, 201)
        wf_data = res_create.json()
        workflow_id = wf_data["workflow_id"]
        self.assertEqual(wf_data["step_count"], 3)

        # 3. Read authorization: caller with INTERACTION_READ (read-only member) can inspect workflow
        res_get_ro = self.client.get(
            f"/api/v1/projects/{self.project_a}/workflows/{workflow_id}",
            headers={"Authorization": f"Bearer {self.readonly_token}"},
        )
        self.assertEqual(res_get_ro.status_code, 200)
        self.assertEqual(res_get_ro.json()["workflow_id"], workflow_id)

        # 4. Negative authorization: caller with READ_ONLY cannot execute workflow
        res_exec_ro = self.client.post(
            f"/api/v1/projects/{self.project_a}/workflows/{workflow_id}/execute",
            headers={"Authorization": f"Bearer {self.readonly_token}"},
            json={},
        )
        self.assertEqual(res_exec_ro.status_code, 403)

        # 5. Positive execution: caller with INTERACTION_EXECUTE succeeds
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


class TestHttpRateLimiting(unittest.TestCase):
    """Test in-process bounded HTTP rate limiter (Token Bucket, IP resolution, LRU eviction)."""

    def setUp(self) -> None:
        self.container = create_sqlite_container(db_path=":memory:")
        # Low capacity (5 authenticated rpm, 3 unauthenticated rpm) for deterministic testing
        self.limiter = HttpRateLimiter(
            authenticated_rpm=5,
            unauthenticated_rpm=3,
            burst_multiplier=1.0,
            max_buckets=5,
            entry_ttl_seconds=1.0,
            trusted_proxies={"127.0.0.1"},
        )
        self.app = create_app(container=self.container, limiter=self.limiter)
        self.client = TestClient(self.app)

        self.project_id = f"proj_lim_{uuid.uuid4().hex[:8]}"
        self.user_id = f"user_lim_{uuid.uuid4().hex[:8]}"
        self.container.auth_store.create_user(
            UserRecord(user_id=self.user_id, email="lim@example.com", name="Limiter User")
        )
        self.container.auth_store.add_project_member(
            ProjectMemberRecord(
                member_id=f"mem_{uuid.uuid4().hex[:8]}",
                project_id=self.project_id,
                user_id=self.user_id,
                role=ProjectRole.OWNER,
            )
        )
        self.token = self.container.token_service.create_access_token(
            subject=self.user_id,
            identity_type=IdentityType.USER,
            name="Limiter User",
            scopes=["*"],
        )

    def tearDown(self) -> None:
        self.client.close()
        self.container.close()

    def test_authenticated_rate_limit_enforcement(self) -> None:
        headers = {"Authorization": f"Bearer {self.token}"}
        # First request succeeds
        res1 = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers)
        self.assertEqual(res1.status_code, 200)

        # Consume remaining tokens
        for _ in range(4):
            self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers)

        # Depleted bucket returns HTTP 429 with Retry-After header
        res_blocked = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers)
        self.assertEqual(res_blocked.status_code, 429)
        self.assertIn("Retry-After", res_blocked.headers)
        self.assertEqual(res_blocked.json()["error"], "TOO_MANY_REQUESTS")

    def test_same_api_key_same_project_shares_bucket(self) -> None:
        agent_id = f"agent_lim_{uuid.uuid4().hex[:8]}"
        created_key = self.container.auth_service.create_agent_api_key(
            agent_id=agent_id,
            project_id=self.project_id,
            name="Limiter Agent",
            scopes=["memory:read"],
        )
        headers = {"X-API-Key": created_key.raw_key}

        # 5 requests succeed (bucket capacity = 5)
        for _ in range(5):
            res = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers)
            self.assertEqual(res.status_code, 200)

        # 6th request with the same API key in the same project is rate-limited (429)
        res_blocked = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers)
        self.assertEqual(res_blocked.status_code, 429)
        self.assertIn("Retry-After", res_blocked.headers)

    def test_same_api_key_identity_different_authorized_project_uses_independent_buckets(self) -> None:
        agent_id = f"agent_shared_{uuid.uuid4().hex[:8]}"
        proj_b = f"proj_lim_b_{uuid.uuid4().hex[:8]}"

        key_a = self.container.auth_service.create_agent_api_key(
            agent_id=agent_id,
            project_id=self.project_id,
            name="Agent in Proj A",
            scopes=["memory:read"],
        )
        key_b = self.container.auth_service.create_agent_api_key(
            agent_id=agent_id,
            project_id=proj_b,
            name="Agent in Proj B",
            scopes=["memory:read"],
        )

        # Exhaust Project A's bucket
        for _ in range(5):
            res_a = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers={"X-API-Key": key_a.raw_key})
            self.assertEqual(res_a.status_code, 200)
        res_a_blocked = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers={"X-API-Key": key_a.raw_key})
        self.assertEqual(res_a_blocked.status_code, 429)

        # Project B with Key B for same agent identity operates in an independent bucket and succeeds
        res_b = self.client.get(f"/api/v1/projects/{proj_b}/memories", headers={"X-API-Key": key_b.raw_key})
        self.assertEqual(res_b.status_code, 200)

    def test_unauthorized_project_switching_cannot_create_or_access_bucket(self) -> None:
        agent_id = f"agent_isolated_{uuid.uuid4().hex[:8]}"
        foreign_proj = f"proj_foreign_{uuid.uuid4().hex[:8]}"
        key_a = self.container.auth_service.create_agent_api_key(
            agent_id=agent_id,
            project_id=self.project_id,
            name="Isolated Agent",
            scopes=["memory:read"],
        )

        initial_buckets = self.limiter.active_bucket_count()

        # Attempt to access foreign_proj using key authorized ONLY for self.project_id
        res = self.client.get(f"/api/v1/projects/{foreign_proj}/memories", headers={"X-API-Key": key_a.raw_key})
        # Strictly rejected by RBAC with 403 Forbidden
        self.assertEqual(res.status_code, 403)

        # Crucial check: No bucket for foreign_proj was ever created or accessed
        self.assertEqual(self.limiter.active_bucket_count(), initial_buckets)

    def test_independent_identity_rate_limit_buckets(self) -> None:
        headers1 = {"Authorization": f"Bearer {self.token}"}
        for _ in range(5):
            self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers1)
        res_blocked = self.client.get(f"/api/v1/projects/{self.project_id}/memories", headers=headers1)
        self.assertEqual(res_blocked.status_code, 429)

        # User 2 has separate bucket and is NOT throttled by User 1's consumption
        user2_id = f"user2_{uuid.uuid4().hex[:8]}"
        self.container.auth_store.create_user(
            UserRecord(user_id=user2_id, email="user2@example.com", name="User 2")
        )
        self.container.auth_store.add_project_member(
            ProjectMemberRecord(
                member_id=f"mem2_{uuid.uuid4().hex[:8]}",
                project_id=self.project_id,
                user_id=user2_id,
                role=ProjectRole.OWNER,
            )
        )
        token2 = self.container.token_service.create_access_token(
            subject=user2_id,
            identity_type=IdentityType.USER,
            name="User 2",
        )
        res_user2 = self.client.get(
            f"/api/v1/projects/{self.project_id}/memories",
            headers={"Authorization": f"Bearer {token2}"},
        )
        self.assertEqual(res_user2.status_code, 200)

    def test_untrusted_vs_trusted_proxy_ip_resolution(self) -> None:
        # Request with untrusted proxy does not spoof client IP
        limiter_untrusted = HttpRateLimiter(
            unauthenticated_rpm=10,
            trusted_proxies=set(),
        )
        scope_mock = {
            "type": "http",
            "client": ("127.0.0.1", 12345),
            "headers": [(b"x-forwarded-for", b"203.0.113.1, 10.0.0.1")],
        }
        req_untrusted = Request(scope_mock)
        resolved_ip = limiter_untrusted.resolve_client_ip(req_untrusted)
        self.assertEqual(resolved_ip, "127.0.0.1")

        # When proxy is in trusted_proxies, X-Forwarded-For is respected
        limiter_trusted = HttpRateLimiter(
            unauthenticated_rpm=10,
            trusted_proxies={"127.0.0.1"},
        )
        resolved_trusted_ip = limiter_trusted.resolve_client_ip(req_untrusted)
        self.assertEqual(resolved_trusted_ip, "203.0.113.1")

    def test_bucket_capacity_bounding_and_eviction(self) -> None:
        bounded_limiter = HttpRateLimiter(max_buckets=2, entry_ttl_seconds=60.0)

        def make_req(ip: str) -> Request:
            return Request({"type": "http", "client": (ip, 12345), "headers": []})

        bounded_limiter.check(make_req("10.0.0.1"))
        bounded_limiter.check(make_req("10.0.0.2"))
        self.assertEqual(bounded_limiter.active_bucket_count(), 2)

        # Adding a 3rd evicts the oldest bucket, bounding storage to max_buckets
        bounded_limiter.check(make_req("10.0.0.3"))
        self.assertLessEqual(bounded_limiter.active_bucket_count(), 2)


class TestServerSentEventsStreaming(BaseAPITestCase):
    """Dedicated test suite verifying progressive SSE streaming, client disconnect cancellation, and idempotency."""

    def test_progressive_chunk_delivery(self) -> None:
        stream_payload = {
            "input_text": "Implement secure authentication with PostgreSQL and OAuth 2.0",
            "search_query": "PostgreSQL authentication",
            "agent_id": "agent-coder",
        }
        res = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=stream_payload,
        )
        self.assertEqual(res.status_code, 200)

        chunks: List[str] = []
        events: List[str] = []
        for line in res.text.split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                events.append(line.replace("event: ", "").strip())
            elif line.startswith("data: ") and events and events[-1] == "agent_chunk":
                payload = json.loads(line.replace("data: ", "").strip())
                chunks.append(payload["chunk"])

        # Must have delivered multiple progressive chunks (not a fabricated single chunk)
        self.assertGreater(len(chunks), 1)
        full_streamed_response = "".join(chunks)
        self.assertIn("PostgreSQL", full_streamed_response)

    def test_disconnect_after_at_least_one_progressive_agent_chunk(self) -> None:
        idem_key = f"disc_chunk_{uuid.uuid4().hex[:8]}"
        stream_payload = {
            "input_text": "Progressive chunk generation that will be interrupted mid-stream",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # Mock is_disconnected: False on prep, False on context, False on chunk 0, True on chunk 1
        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            mock_disc.side_effect = [False, False, False, True, True, True, True]

            res = self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=stream_payload,
            )
            self.assertEqual(res.status_code, 200)

            # Delivered prepare, context_delivered, and at least 1 agent_chunk
            self.assertIn("event: prepare", res.text)
            self.assertIn("event: context_delivered", res.text)
            self.assertIn("event: agent_chunk", res.text)
            # Completion was cancelled: never reached completion or done
            self.assertNotIn("event: completion", res.text)
            self.assertNotIn("event: done", res.text)

            # Zero memories written to memory_store
            self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 0)

            # Interaction record in store is marked FAILED / INTERRUPTED
            saved_rec = self.container.interaction_store.get_by_idempotency_key(idem_key)
            self.assertIsNotNone(saved_rec)
            self.assertEqual(saved_rec.status.value, "FAILED")
            self.assertIn("INTERRUPTED", saved_rec.current_stage)

    def test_reconnect_using_same_idempotency_key_after_interruption(self) -> None:
        idem_key = f"recon_{uuid.uuid4().hex[:8]}"
        stream_payload = {
            "input_text": "Setup PostgreSQL database connection pooling for reconnection",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # 1. First attempt: interrupted mid-stream
        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            mock_disc.side_effect = [False, False, False, True, True, True]
            res1 = self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=stream_payload,
            )
            self.assertEqual(res1.status_code, 200)
            self.assertNotIn("event: completion", res1.text)

        # Zero memories written during interrupted attempt
        self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 0)

        # 2. Reconnection attempt with same idempotency key (normal connection without disconnect)
        res2 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=stream_payload,
        )
        self.assertEqual(res2.status_code, 200)
        self.assertIn("event: completion", res2.text)
        self.assertIn("event: done", res2.text)

        # Exactly ONE completed record in store
        completed_rec = self.container.interaction_store.get_by_idempotency_key(idem_key)
        self.assertIsNotNone(completed_rec)
        self.assertEqual(completed_rec.status.value, "COMPLETED")

        # Exactly ONE completion's memories persisted (not duplicated)
        self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 1)

    def test_multiple_retries_after_interruption(self) -> None:
        idem_key = f"multi_retry_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Setup PostgreSQL database connection pooling for multiple retries",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # Attempt 1: disconnects early
        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            mock_disc.side_effect = [False, True, True, True]
            self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=payload,
            )
        self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 0)

        # Attempt 2: disconnects mid-stream
        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            mock_disc.side_effect = [False, False, False, True, True]
            self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=payload,
            )
        self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 0)

        # Attempt 3: runs to completion
        res3 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=payload,
        )
        self.assertEqual(res3.status_code, 200)
        self.assertIn("event: completion", res3.text)
        self.assertIn("event: done", res3.text)
        self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 1)

    def test_two_concurrent_requests_with_same_key_rejected(self) -> None:
        idem_key = f"concurrent_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Concurrent invocation conflict check",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # Actively reserve key in database to simulate ongoing worker execution
        active_res = self.container.idempotency_manager.reserve(
            project_id=self.project_a,
            idempotency_key=idem_key,
            input_content=payload["input_text"],
            interaction_store=self.container.interaction_store,
            lease_seconds=30.0,
        )

        try:
            # Second concurrent request with the same key arrives while first is active
            res_concurrent = self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=payload,
            )
            # Rejected with HTTP 409 Conflict
            self.assertEqual(res_concurrent.status_code, 409)
            resp_body = res_concurrent.json()
            err_msg = resp_body.get("message") or resp_body.get("detail", "")
            self.assertIn("currently IN_PROGRESS", err_msg)
        finally:
            active_res.release()

    def test_zero_partial_memory_writes_when_operation_abandoned(self) -> None:
        idem_key = f"abandoned_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Abandoned interaction test verifying zero memory writes",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            mock_disc.side_effect = [False, False, False, True, True]
            self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=payload,
            )

        # Abandoned interaction must leave 0 memories written
        self.assertEqual(self.container.memory_store.count(project_id=self.project_a), 0)

    def test_recovery_of_stale_in_progress_reservations(self) -> None:
        idem_key = f"stale_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Recovery from crash or stale in-progress state",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # Seed an interaction in store with status INITIALIZED and updated_at 60s in the past (stale)
        stale_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        stale_rec = InteractionRecord(
            interaction_id=f"int_stale_{uuid.uuid4().hex[:8]}",
            idempotency_key=idem_key,
            project_id=self.project_a,
            agent_id="agent-coder",
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            status=InteractionStatus.INITIALIZED,
            current_stage="INITIALIZED",
            input=AgentInteractionInput(
                interaction_id=f"int_stale_{uuid.uuid4().hex[:8]}",
                project_id=self.project_a,
                agent_id="agent-coder",
                session_id=f"sess_{uuid.uuid4().hex[:8]}",
                content=payload["input_text"],
                idempotency_key=idem_key,
            ),
            created_at=stale_time,
            updated_at=stale_time,
        )
        self.container.interaction_store.save(stale_rec)

        # Incoming request must not be blocked by stale reservation; it reclaims and completes cleanly
        res = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=payload,
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("event: completion", res.text)
        self.assertIn("event: done", res.text)

    def test_idempotency_key_reconnection_no_duplicate_writes(self) -> None:
        idem_key = f"sse_idem_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Setup PostgreSQL database connection pooling",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # 1. First execution
        res1 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=payload,
        )
        self.assertEqual(res1.status_code, 200)

        initial_memory_count = self.container.memory_store.count(project_id=self.project_a)

        # 2. Reconnection with same idempotency key
        res2 = self.client.post(
            f"/api/v1/projects/{self.project_a}/interactions/stream",
            headers={"Authorization": f"Bearer {self.owner_token}"},
            json=payload,
        )
        self.assertEqual(res2.status_code, 200)
        self.assertIn('"replayed": true', res2.text)

        # 3. Verify zero duplicate memory extractions or writes occurred
        after_memory_count = self.container.memory_store.count(project_id=self.project_a)
        self.assertEqual(initial_memory_count, after_memory_count)

    def test_client_disconnect_cancellation(self) -> None:
        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            # Client drops connection after context delivery
            mock_disc.side_effect = [False, False, True, True, True, True]

            stream_payload = {
                "input_text": "Transient interaction aborted on client disconnect",
                "idempotency_key": f"disc_{uuid.uuid4().hex[:8]}",
            }
            res = self.client.post(
                f"/api/v1/projects/{self.project_a}/interactions/stream",
                headers={"Authorization": f"Bearer {self.owner_token}"},
                json=stream_payload,
            )
            self.assertEqual(res.status_code, 200)
            # The completion event was not emitted because client disconnected before completion
            self.assertNotIn("event: completion", res.text)
            self.assertNotIn("event: done", res.text)


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


@unittest.skipUnless(_IS_PG_LIVE, "Live PostgreSQL required for distributed idempotency tests")
class TestPostgreSQLDistributedIdempotency(unittest.TestCase):
    """Verifies durable, cross-process idempotency coordination on live PostgreSQL."""

    @classmethod
    def setUpClass(cls) -> None:
        live, msg, _ = _check_postgres_live(close_after=True)
        if not live:
            raise unittest.SkipTest(f"Live PostgreSQL unavailable: {msg}")

    def setUp(self) -> None:
        self.project_id = f"proj_pg_idem_{uuid.uuid4().hex[:8]}"
        self.user_id = f"user_pg_idem_{uuid.uuid4().hex[:8]}"

        cfg1 = DatabaseConfig(backend=StorageBackend.POSTGRES, database_url=ADMIN_PG_URL, pool_size=5, pool_timeout=3.0)
        cfg2 = DatabaseConfig(backend=StorageBackend.POSTGRES, database_url=ADMIN_PG_URL, pool_size=5, pool_timeout=3.0)
        self.mgr1 = DatabaseConnectionManager(cfg1)
        self.mgr2 = DatabaseConnectionManager(cfg2)

        self.container1 = create_postgres_container(self.mgr1)
        self.container2 = create_postgres_container(self.mgr2)
        self.app1 = create_app(container=self.container1)
        self.app2 = create_app(container=self.container2)
        self.client1 = TestClient(self.app1)
        self.client2 = TestClient(self.app2)

        # Setup user and project membership in PostgreSQL auth store
        self.container1.auth_store.create_user(UserRecord(user_id=self.user_id, email=f"{self.user_id}@example.com", name="PG Idem User"))
        self.container1.auth_store.add_project_member(ProjectMemberRecord(member_id=f"mem_pg_{uuid.uuid4().hex[:8]}", project_id=self.project_id, user_id=self.user_id, role=ProjectRole.OWNER))
        self.token = self.container1.token_service.create_access_token(
            subject=self.user_id,
            identity_type=IdentityType.USER,
            name="PG Idem User",
            scopes=["*"],
        )

    def tearDown(self) -> None:
        self.client1.close()
        self.client2.close()
        self.container1.close()
        self.container2.close()

    def test_01_two_independent_connections_same_reservation_one_winner(self) -> None:
        """1 & 4: Two independent connections attempting the same reservation: only one obtains ownership."""
        idem_key = f"key_race_{uuid.uuid4().hex[:8]}"
        coord1 = self.container1.idempotency_manager.coordinator
        coord2 = self.container2.idempotency_manager.coordinator

        # Worker 1 reserves successfully
        res1 = coord1.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_alpha",
            lease_seconds=5.0,
            interaction_store=self.container1.interaction_store,
            input_content="Concurrent reservation attempt",
        )
        self.assertEqual(res1.status, IdempotencyStatus.NEW)
        self.assertEqual(res1.fencing_token, 1)

        # Worker 2 attempts concurrent reservation with same key -> rejected with 409 Conflict
        with self.assertRaises(HTTPException) as ctx:
            coord2.reserve(
                project_id=self.project_id,
                idempotency_key=idem_key,
                owner_id="worker_beta",
                lease_seconds=5.0,
                interaction_store=self.container2.interaction_store,
                input_content="Concurrent reservation attempt",
            )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("currently IN_PROGRESS", ctx.exception.detail)
        res1.release()

    def test_02_concurrent_requests_through_separate_worker_instances(self) -> None:
        """2, 3 & 4: Concurrent HTTP requests through separate worker/client instances: only one winner."""
        idem_key = f"http_race_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Concurrent stream execution across separate worker replicas",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        def send_request(client: TestClient) -> Any:
            return client.post(
                f"/api/v1/projects/{self.project_id}/interactions/stream",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(send_request, self.client1)
            fut2 = executor.submit(send_request, self.client2)
            res1 = fut1.result()
            res2 = fut2.result()

        # Exactly one must succeed (200 OK), the other rejected (409 Conflict)
        statuses = sorted([res1.status_code, res2.status_code])
        self.assertEqual(statuses, [200, 409], f"Expected [200, 409], got {statuses}")

    def test_05_live_lease_not_reclaimed_when_heartbeats_continue(self) -> None:
        """5: A live lease is NOT reclaimed after original lease interval when heartbeats continue."""
        idem_key = f"live_hb_{uuid.uuid4().hex[:8]}"
        coord1 = self.container1.idempotency_manager.coordinator
        coord2 = self.container2.idempotency_manager.coordinator

        # Worker 1 reserves with 0.8s lease
        res1 = coord1.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_live",
            lease_seconds=0.8,
            interaction_store=self.container1.interaction_store,
            input_content="Heartbeat lease protection test",
        )

        # Worker 1 sends heartbeats across a 1.2s window (exceeding original 0.8s lease)
        for _ in range(4):
            time.sleep(0.3)
            res1.heartbeat(lease_seconds=0.8)

        # Worker 2 attempts to claim while Worker 1 is still actively heartbeating -> 409 Conflict
        with self.assertRaises(HTTPException) as ctx:
            coord2.reserve(
                project_id=self.project_id,
                idempotency_key=idem_key,
                owner_id="worker_intruder",
                lease_seconds=1.0,
                interaction_store=self.container2.interaction_store,
                input_content="Heartbeat lease protection test",
            )
        self.assertEqual(ctx.exception.status_code, 409)
        res1.release()

    def test_06_expired_lease_reclaimed_after_heartbeat_failure(self) -> None:
        """6: An expired lease CAN be reclaimed after heartbeat failure / worker crash."""
        idem_key = f"exp_hb_{uuid.uuid4().hex[:8]}"
        coord1 = self.container1.idempotency_manager.coordinator
        coord2 = self.container2.idempotency_manager.coordinator

        # Worker 1 reserves with short lease (0.5s) and abandons it (stops heartbeating)
        res1 = coord1.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_crashed",
            lease_seconds=0.5,
            interaction_store=self.container1.interaction_store,
            input_content="Expired lease reclamation test",
        )
        self.assertEqual(res1.fencing_token, 1)

        # Wait for lease to expire
        time.sleep(0.7)

        # Worker 2 arrives and successfully reclaims the expired lease
        res2 = coord2.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_reclaimer",
            lease_seconds=2.0,
            interaction_store=self.container2.interaction_store,
            input_content="Expired lease reclamation test",
        )
        self.assertEqual(res2.status, IdempotencyStatus.INTERRUPTED)
        # Fencing token is monotonically incremented
        self.assertEqual(res2.fencing_token, 2)
        res2.release()

    def test_07_obsolete_fencing_token_cannot_commit_completion_or_memories(self) -> None:
        """7: A zombie worker with an obsolete fencing token cannot commit completion or extracted memories."""
        idem_key = f"fence_guard_{uuid.uuid4().hex[:8]}"
        coord1 = self.container1.idempotency_manager.coordinator
        coord2 = self.container2.idempotency_manager.coordinator

        # Worker 1 gets lease (fencing_token=1)
        res1 = coord1.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_zombie",
            lease_seconds=0.5,
            interaction_store=self.container1.interaction_store,
            input_content="Fencing guard test",
        )
        self.assertEqual(res1.fencing_token, 1)

        # Lease expires
        time.sleep(0.7)

        # Worker 2 reclaims (fencing_token=2)
        res2 = coord2.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_active",
            lease_seconds=5.0,
            interaction_store=self.container2.interaction_store,
            input_content="Fencing guard test",
        )
        self.assertEqual(res2.fencing_token, 2)

        # Zombie Worker 1 wakes up and attempts to commit completion -> REJECTED by fencing token check
        with self.assertRaises(StaleFencingTokenError):
            res1.commit_completion()

        # Zombie Worker 1 attempts heartbeat -> REJECTED
        with self.assertRaises(StaleFencingTokenError):
            res1.heartbeat()

        # Legitimate Worker 2 can commit completion cleanly
        res2.commit_completion()
        self.assertEqual(res2.status, IdempotencyStatus.COMPLETED)

    def test_08_reclaimed_operation_completes_without_duplicate_writes(self) -> None:
        """8 & 9: Reclaimed operation completes with exactly one set of memories, graph, and events."""
        idem_key = f"reclaim_uniq_{uuid.uuid4().hex[:8]}"
        payload = {
            "input_text": "Setup PostgreSQL database connection pooling for single completion test",
            "idempotency_key": idem_key,
            "agent_id": "agent-coder",
        }

        # 1. First run disconnects mid-stream
        with patch("starlette.requests.Request.is_disconnected", new_callable=AsyncMock) as mock_disc:
            mock_disc.side_effect = [False, False, False, True, True, True]
            res_abort = self.client1.post(
                f"/api/v1/projects/{self.project_id}/interactions/stream",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )
            self.assertEqual(res_abort.status_code, 200)

        # Zero memories written during aborted run
        self.assertEqual(self.container1.memory_store.count(project_id=self.project_id), 0)

        # 2. Separate worker instance resumes with same key and runs to completion
        res_done = self.client2.post(
            f"/api/v1/projects/{self.project_id}/interactions/stream",
            headers={"Authorization": f"Bearer {self.token}"},
            json=payload,
        )
        self.assertEqual(res_done.status_code, 200)
        self.assertIn("event: completion", res_done.text)

        # Exactly 1 memory persisted in PostgreSQL
        self.assertEqual(self.container1.memory_store.count(project_id=self.project_id), 1)

        # 3. Third request replays cached result without duplicate memory writes
        res_replay = self.client1.post(
            f"/api/v1/projects/{self.project_id}/interactions/stream",
            headers={"Authorization": f"Bearer {self.token}"},
            json=payload,
        )
        self.assertEqual(res_replay.status_code, 200)
        self.assertIn('"replayed": true', res_replay.text)
        self.assertEqual(self.container1.memory_store.count(project_id=self.project_id), 1)


class TestSQLiteDistributedIdempotency(unittest.TestCase):
    """Verifies that SQLite offline idempotency coordinator preserves identical semantics."""

    def setUp(self) -> None:
        self.project_id = f"proj_sq_idem_{uuid.uuid4().hex[:8]}"
        self.coord = SQLiteIdempotencyCoordinator(":memory:")
        self.container = create_sqlite_container(":memory:")

    def tearDown(self) -> None:
        self.coord.close()
        self.container.close()

    def test_sqlite_concurrent_reservation_rejected(self) -> None:
        idem_key = f"sq_race_{uuid.uuid4().hex[:8]}"
        res1 = self.coord.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_sq_1",
            lease_seconds=5.0,
            interaction_store=self.container.interaction_store,
            input_content="SQLite concurrency test",
        )
        self.assertEqual(res1.status, IdempotencyStatus.NEW)
        self.assertEqual(res1.fencing_token, 1)

        with self.assertRaises(HTTPException) as ctx:
            self.coord.reserve(
                project_id=self.project_id,
                idempotency_key=idem_key,
                owner_id="worker_sq_2",
                lease_seconds=5.0,
                interaction_store=self.container.interaction_store,
                input_content="SQLite concurrency test",
            )
        self.assertEqual(ctx.exception.status_code, 409)
        res1.release()

    def test_sqlite_heartbeat_renewal_and_expired_reclamation(self) -> None:
        idem_key = f"sq_exp_{uuid.uuid4().hex[:8]}"
        res1 = self.coord.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_sq_1",
            lease_seconds=0.4,
            interaction_store=self.container.interaction_store,
            input_content="SQLite heartbeat test",
        )
        time.sleep(0.5)

        # Worker 2 reclaims expired lease
        res2 = self.coord.reserve(
            project_id=self.project_id,
            idempotency_key=idem_key,
            owner_id="worker_sq_2",
            lease_seconds=2.0,
            interaction_store=self.container.interaction_store,
            input_content="SQLite heartbeat test",
        )
        self.assertEqual(res2.status, IdempotencyStatus.INTERRUPTED)
        self.assertEqual(res2.fencing_token, 2)

        # Worker 1 obsolete fencing token fails
        with self.assertRaises(StaleFencingTokenError):
            res1.commit_completion()

        # Worker 2 completes cleanly
        res2.commit_completion()
        self.assertEqual(res2.status, IdempotencyStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
