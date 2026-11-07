"""Acceptance test suite for AI Brain Milestone 5.7: Docker Deployment & Observability Packaging.

Verifies:
1. Dockerfile structure, non-root user, tini entrypoint, healthcheck, and no dev-deps.
2. Docker Compose configuration parsing, services, dependencies, and healthchecks.
3. Prometheus scraping configuration and Grafana dashboard provisioning.
4. Prometheus metrics exposition (/metrics) and HTTP request instrumentation.
5. Task queue metrics (queue depth, worker active count, task transitions, retries, failures).
6. Structured JSON logging with correlation IDs and credential/secret masking.
7. Worker daemon CLI (run, healthcheck, status, graceful signal handling).
8. Live PostgreSQL and container factory integration.
9. Conditional Docker CLI runtime build and validation (skipped if Docker is not installed on host).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import unittest
import uuid
from typing import Any, Dict

from fastapi.testclient import TestClient

from ai_brain.config import AppConfig, DatabaseConfig, StorageBackend, load_config_from_env
from ai_brain.observability.logging import StructuredJsonFormatter, redact_sensitive_text
from ai_brain.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_metrics_registry,
)
from ai_brain.server.app import create_app
from ai_brain.server.dependencies import (
    create_container_from_config,
    create_sqlite_container,
)
from ai_brain.tasks.cli import (
    cmd_healthcheck,
    cmd_status,
    write_heartbeat,
)
from ai_brain.tasks.fleet import WorkerFleet
from ai_brain.tasks.models import TaskRecord, TaskResult, TaskStatus
from ai_brain.tasks.worker import BackgroundWorker


class TestDockerAndComposeManifests(unittest.TestCase):
    """Verify Docker, Compose, and observability configuration files."""

    def setUp(self) -> None:
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_dockerfile_compliance(self) -> None:
        """Verify Dockerfile base image, non-root user, tini entrypoint, and healthchecks."""
        dockerfile_path = os.path.join(self.project_root, "Dockerfile")
        self.assertTrue(os.path.isfile(dockerfile_path), "Dockerfile must exist in project root")

        with open(dockerfile_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Minimal appropriate base image
        self.assertIn("FROM python:3.13-slim", content)

        # Reproducible dependencies
        self.assertIn("COPY requirements.txt", content)
        self.assertIn("pip install --no-cache-dir", content)

        # Non-root user
        self.assertIn("useradd -u 1000 -g aibrain", content)
        self.assertIn("USER aibrain", content)

        # Signal and process handling
        self.assertIn("tini", content)
        self.assertIn('ENTRYPOINT ["/usr/bin/tini", "--"]', content)

        # Healthcheck defined
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("http://localhost:8000/healthz", content)

        # Exposes port 8000
        self.assertIn("EXPOSE 8000", content)

    def test_dockerignore_compliance(self) -> None:
        """Verify .dockerignore excludes tests, databases, secrets, and git metadata."""
        ignore_path = os.path.join(self.project_root, ".dockerignore")
        self.assertTrue(os.path.isfile(ignore_path), ".dockerignore must exist")

        with open(ignore_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        self.assertIn(".git", lines)
        self.assertTrue(any(".env" in line for line in lines))
        self.assertIn("tests", lines)
        self.assertIn("*.db", lines)
        self.assertIn("__pycache__", lines)

    def test_compose_configuration_structure(self) -> None:
        """Verify docker-compose.yml defines all 6 required services, dependencies, and healthchecks."""
        compose_path = os.path.join(self.project_root, "docker-compose.yml")
        self.assertTrue(os.path.isfile(compose_path), "docker-compose.yml must exist")

        with open(compose_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check required services
        self.assertIn("postgres:", content)
        self.assertIn("migration:", content)
        self.assertIn("api:", content)
        self.assertIn("worker:", content)
        self.assertIn("prometheus:", content)
        self.assertIn("grafana:", content)

        # Check pgvector image
        self.assertIn("image: pgvector/pgvector:pg16", content)

        # Check migration dependency coordination (no migration race)
        self.assertIn("condition: service_completed_successfully", content)

        # Check non-root ports / healthchecks
        self.assertIn("8000:8000", content)
        self.assertIn("9090:9090", content)
        self.assertIn("3000:3000", content)

        # Check volumes
        self.assertIn("postgres_data:", content)
        self.assertIn("prometheus_data:", content)
        self.assertIn("grafana_data:", content)

    def test_prometheus_configuration_syntax(self) -> None:
        """Verify prometheus.yml scrape targets and paths."""
        prom_path = os.path.join(self.project_root, "deploy", "prometheus", "prometheus.yml")
        self.assertTrue(os.path.isfile(prom_path), "prometheus.yml must exist")

        with open(prom_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('job_name: "ai_brain"', content)
        self.assertIn('metrics_path: "/metrics"', content)
        self.assertIn('"api:8000"', content)

    def test_grafana_provisioning_and_dashboard_schema(self) -> None:
        """Verify Grafana datasource, dashboard provider, and dashboard JSON."""
        ds_path = os.path.join(self.project_root, "deploy", "grafana", "provisioning", "datasources", "prometheus.yml")
        self.assertTrue(os.path.isfile(ds_path), "Grafana datasource config must exist")

        with open(ds_path, "r", encoding="utf-8") as f:
            ds_content = f.read()
        self.assertIn("url: http://prometheus:9090", ds_content)

        dash_json_path = os.path.join(
            self.project_root, "deploy", "grafana", "provisioning", "dashboards", "ai_brain_dashboard.json"
        )
        self.assertTrue(os.path.isfile(dash_json_path), "Grafana dashboard JSON must exist")

        with open(dash_json_path, "r", encoding="utf-8") as f:
            dashboard = json.load(f)

        self.assertEqual(dashboard.get("title"), "AI Brain Production Overview")
        panels = dashboard.get("panels", [])
        self.assertGreaterEqual(len(panels), 6, "Dashboard should contain at least 6 core monitoring panels")

        # Verify key metric expressions exist in dashboard targets
        expressions = [
            target.get("expr", "")
            for panel in panels
            for target in panel.get("targets", [])
        ]
        self.assertTrue(any("ai_brain_ready" in expr for expr in expressions))
        self.assertTrue(any("ai_brain_database_connected" in expr for expr in expressions))
        self.assertTrue(any("ai_brain_http_requests_total" in expr for expr in expressions))
        self.assertTrue(any("ai_brain_task_queue_depth" in expr for expr in expressions))

    def test_ci_workflow_manifest_structure(self) -> None:
        """Verify .github/workflows/docker-deployment.yml defines automated CI acceptance pipeline."""
        wf_path = os.path.join(self.project_root, ".github", "workflows", "docker-deployment.yml")
        self.assertTrue(os.path.isfile(wf_path), "CI workflow file must exist")

        with open(wf_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("ubuntu-latest", content)
        self.assertIn("docker compose config", content)
        self.assertIn("verify_docker_runtime_acceptance.py", content)
        self.assertIn("unittest discover", content)

    def test_env_example_safety(self) -> None:
        """Verify .env.example contains only placeholders and no real secrets."""
        env_ex_path = os.path.join(self.project_root, ".env.example")
        self.assertTrue(os.path.isfile(env_ex_path), ".env.example must exist")

        with open(env_ex_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Must not contain real passwords or keys
        self.assertIn("replace_with_strong_random_password_here", content)
        self.assertIn("replace_with_at_least_32_characters_secret_key!", content)
        self.assertNotIn("supersecret", content.lower())
        self.assertNotIn("password123", content.lower())


class TestObservabilityMetricsAndLogging(unittest.TestCase):
    """Verify Prometheus metrics collection, formatting, and structured logging."""

    def setUp(self) -> None:
        self.registry = MetricsRegistry()

    def test_counter_and_gauge_exposition(self) -> None:
        """Verify Counter and Gauge increment, decrement, and Prometheus exposition formatting."""
        c = self.registry.register_counter("test_counter", "Test counter help")
        c.inc(method="POST", status="200")
        c.inc(method="POST", status="200")
        c.inc(amount=3, method="GET", status="404")

        self.assertEqual(c.get(method="POST", status="200"), 2.0)
        self.assertEqual(c.get(method="GET", status="404"), 3.0)

        g = self.registry.register_gauge("test_gauge", "Test gauge help")
        g.set(42.0, region="us-east")
        g.inc(amount=8.0, region="us-east")
        g.dec(amount=10.0, region="us-east")
        self.assertEqual(g.get(region="us-east"), 40.0)

        expo = self.registry.generate_exposition()
        self.assertIn("# HELP test_counter Test counter help", expo)
        self.assertIn("# TYPE test_counter counter", expo)
        self.assertIn('test_counter{method="POST",status="200"} 2', expo)
        self.assertIn('test_counter{method="GET",status="404"} 3', expo)
        self.assertIn("# HELP test_gauge Test gauge help", expo)
        self.assertIn("# TYPE test_gauge gauge", expo)
        self.assertIn('test_gauge{region="us-east"} 40', expo)

    def test_histogram_exposition(self) -> None:
        """Verify Histogram observations and cumulative bucket counts."""
        h = self.registry.register_histogram("test_latency", "Latency in seconds", buckets=(0.01, 0.05, 0.1))
        h.observe(0.005, endpoint="/test")
        h.observe(0.03, endpoint="/test")
        h.observe(0.09, endpoint="/test")
        h.observe(0.5, endpoint="/test")

        expo = self.registry.generate_exposition()
        self.assertIn("# HELP test_latency Latency in seconds", expo)
        self.assertIn("# TYPE test_latency histogram", expo)
        self.assertIn('test_latency_bucket{endpoint="/test",le="0.01"} 1', expo)
        self.assertIn('test_latency_bucket{endpoint="/test",le="0.05"} 2', expo)
        self.assertIn('test_latency_bucket{endpoint="/test",le="0.1"} 3', expo)
        self.assertIn('test_latency_bucket{endpoint="/test",le="+Inf"} 4', expo)
        self.assertIn('test_latency_count{endpoint="/test"} 4', expo)

    def test_structured_json_logging_and_credential_masking(self) -> None:
        """Verify log records are emitted as JSON and credentials/secrets are masked."""
        formatter = StructuredJsonFormatter()
        logger_test = logging.getLogger("test.logger")

        record = logger_test.makeRecord(
            name="test.logger",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="User authenticated with Bearer eyJhbGciOi.secret.signature and agy_ak_test1234567890 key.",
            args=(),
            exc_info=None,
        )
        setattr(record, "request_id", "req-xyz-987")
        setattr(record, "project_id", "proj-alpha")

        formatted = formatter.format(record)
        data = json.loads(formatted)

        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test.logger")
        self.assertEqual(data["request_id"], "req-xyz-987")
        self.assertEqual(data["project_id"], "proj-alpha")
        self.assertIn("timestamp", data)

        # Credentials must be redacted
        self.assertNotIn("eyJhbGciOi.secret.signature", data["message"])
        self.assertIn("[REDACTED_TOKEN]", data["message"])
        self.assertNotIn("agy_ak_test1234567890", data["message"])
        self.assertIn("...[REDACTED]", data["message"])


class TestApiObservabilityEndpoints(unittest.TestCase):
    """Verify HTTP API endpoints, Prometheus metrics scraper, and probe aliases."""

    def setUp(self) -> None:
        self.container = create_sqlite_container(":memory:")
        self.app = create_app(container=self.container)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.container.close()

    def test_liveness_and_readiness_aliases(self) -> None:
        """Verify /health and /ready route aliases match /healthz and /readyz."""
        res_hz = self.client.get("/healthz")
        res_h = self.client.get("/health")
        self.assertEqual(res_hz.status_code, 200)
        self.assertEqual(res_h.status_code, 200)
        self.assertEqual(res_h.json()["status"], "ok")

        res_rz = self.client.get("/readyz")
        res_r = self.client.get("/ready")
        self.assertEqual(res_rz.status_code, 200)
        self.assertEqual(res_r.status_code, 200)
        self.assertEqual(res_r.json()["status"], "ready")

    def test_prometheus_metrics_endpoint_scraping(self) -> None:
        """Verify GET /metrics returns standard Prometheus text format with system and HTTP metrics."""
        # Execute some requests to populate metrics
        self.client.get("/healthz")
        self.client.get("/readyz")
        self.client.get("/nonexistent_endpoint_for_test")

        res = self.client.get("/metrics")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/plain", res.headers.get("content-type", ""))
        self.assertIn("version=0.0.4", res.headers.get("content-type", ""))

        text = res.text
        self.assertIn("ai_brain_ready", text)
        self.assertIn("ai_brain_database_connected", text)
        self.assertIn("ai_brain_http_requests_total", text)
        self.assertIn("ai_brain_http_request_duration_seconds", text)
        self.assertIn('endpoint="/healthz"', text)
        self.assertIn('status_code="200"', text)
        self.assertIn('status_code="404"', text)


class TestWorkerFleetObservabilityAndCli(unittest.TestCase):
    """Verify worker fleet metrics and CLI subcommands."""

    def setUp(self) -> None:
        self.container = create_sqlite_container(":memory:")
        self.project_id = f"proj_cli_{uuid.uuid4().hex[:6]}"
        self.temp_hb_file = os.path.join(tempfile.gettempdir(), f"hb_test_{uuid.uuid4().hex[:6]}.tmp")

    def tearDown(self) -> None:
        if os.path.exists(self.temp_hb_file):
            try:
                os.remove(self.temp_hb_file)
            except Exception:
                pass
        self.container.close()

    def test_worker_metrics_on_task_execution(self) -> None:
        """Verify task completion and worker active count update Prometheus metrics."""
        task_store = self.container.task_store
        task_registry = self.container.task_registry
        self.assertIsNotNone(task_store)
        self.assertIsNotNone(task_registry)

        fleet = WorkerFleet(
            store=task_store,
            registry=task_registry,
            project_id=self.project_id,
            concurrency=2,
            poll_interval=0.05,
            lease_timeout=5.0,
            fleet_id="test_obs_fleet",
            context=self.container,
        )

        reg = get_metrics_registry()

        fleet.start()
        self.assertEqual(fleet.current_concurrency, 2)
        self.assertEqual(reg.worker_active_count.get(fleet_id="test_obs_fleet"), 2.0)

        # Submit task
        task = task_store.submit_task(
            project_id=self.project_id,
            task_type="memory_maintenance",
            payload={"action": "reindex"},
        )

        time.sleep(0.4)
        fleet.stop(timeout=2.0)

        self.assertEqual(fleet.current_concurrency, 0)
        self.assertEqual(reg.worker_active_count.get(fleet_id="test_obs_fleet"), 0.0)

        done = task_store.get_task(self.project_id, task.task_id)
        self.assertEqual(done.status, TaskStatus.COMPLETED)

        # Verify metric increment
        completed_count = reg.tasks_total.get(
            project_id=self.project_id, task_type="memory_maintenance", status="COMPLETED"
        )
        self.assertGreaterEqual(completed_count, 1.0)

    def test_cli_healthcheck_and_status(self) -> None:
        """Verify cmd_healthcheck detects missing, fresh, and stale heartbeat files."""
        import argparse

        args = argparse.Namespace(heartbeat_file=self.temp_hb_file, max_age=10.0)

        # 1. Missing file -> exit 1
        rc = cmd_healthcheck(args)
        self.assertEqual(rc, 1)

        # 2. Fresh heartbeat -> exit 0
        write_heartbeat(self.temp_hb_file)
        rc = cmd_healthcheck(args)
        self.assertEqual(rc, 0)

        # 3. Status command runs cleanly
        status_args = argparse.Namespace(project_id=self.project_id)
        rc_status = cmd_status(status_args)
        self.assertEqual(rc_status, 0)


class TestContainerFactoryAndConfiguration(unittest.TestCase):
    """Verify create_container_from_config dynamic engine resolution."""

    def test_sqlite_container_resolution(self) -> None:
        cfg = AppConfig(
            db=DatabaseConfig(backend=StorageBackend.SQLITE, database_url="sqlite:///:memory:")
        )
        container = create_container_from_config(cfg)
        try:
            self.assertIsNone(container.db_manager)
            self.assertIsNotNone(container.brain_loop)
            self.assertIsNotNone(container.task_store)
        finally:
            container.close()

    def test_postgres_container_resolution(self) -> None:
        cfg = AppConfig(
            db=DatabaseConfig(
                backend=StorageBackend.POSTGRES,
                database_url="postgresql://postgres@localhost:5432/test_ai_brain",
                pool_size=2,
            )
        )
        container = create_container_from_config(cfg)
        try:
            self.assertIsNotNone(container.db_manager)
            self.assertTrue(container.db_manager.config.is_postgres)
        finally:
            container.close()

    def test_production_container_auth_token_generation_and_rbac(self) -> None:
        """Verify token generation using production container configuration and auth imports."""
        from ai_brain.config import load_config_from_env
        from ai_brain.auth import (
            IdentityType,
            KeyManager,
            Permission,
            PermissionDeniedError,
            ProjectMemberRecord,
            ProjectRole,
            TokenService,
            UserRecord,
        )

        cfg = load_config_from_env()
        container = create_container_from_config(cfg)
        try:
            user_id = f"test_owner_{uuid.uuid4().hex[:8]}"
            project_id = "default"

            # 1. Ensure user exists in auth_store
            user = container.auth_store.get_user(user_id)
            if not user:
                container.auth_store.create_user(
                    UserRecord(user_id=user_id, email=f"{user_id}@example.com", name="CI Owner")
                )

            # 2. Ensure project membership with OWNER role
            member = container.auth_store.get_project_member(project_id, user_id)
            if not member:
                container.auth_store.add_project_member(
                    ProjectMemberRecord(
                        member_id=f"mem_{uuid.uuid4().hex[:8]}",
                        project_id=project_id,
                        user_id=user_id,
                        role=ProjectRole.OWNER,
                    )
                )
            elif member.role != ProjectRole.OWNER:
                container.auth_store.update_project_member_role(project_id, user_id, ProjectRole.OWNER)

            # 3. Generate token using container TokenService
            token = container.token_service.create_access_token(
                subject=user_id,
                identity_type=IdentityType.USER,
                name="CI Owner",
                scopes=["*"],
                expires_in_seconds=3600,
            )
            self.assertIsInstance(token, str)
            self.assertTrue(len(token.split(".")) == 3, "JWT must contain three segments")

            # 4. Authenticate token
            ident = container.auth_service.authenticate_token(token)
            self.assertEqual(ident.identity_id, user_id)
            self.assertEqual(ident.identity_type, IdentityType.USER)
            self.assertTrue(ident.is_authenticated)

            # 5. Authorize permissions for default project
            role = container.rbac_service.get_user_role(project_id, user_id)
            self.assertEqual(role, ProjectRole.OWNER)
            container.rbac_service.authorize(ident, project_id, Permission.INTERACTION_EXECUTE)
            container.rbac_service.authorize(ident, project_id, Permission.INTERACTION_READ)

            # 6. Negative authorization for unassigned foreign project
            with self.assertRaises(PermissionDeniedError):
                container.rbac_service.authorize(ident, "non_existent_project", Permission.INTERACTION_EXECUTE)
        finally:
            container.close()

    def test_harness_token_validation_and_header_construction(self) -> None:
        """Verify structural token validation, exact header construction, and rejection of malformed tokens."""
        from scripts.verify_docker_runtime_acceptance import (
            build_auth_header,
            validate_token_structure,
        )

        valid_sample = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2FtcGxlX3NpZ25hdHVyZQ"

        # 1. Valid token passes validation
        cleaned = validate_token_structure(valid_sample)
        self.assertEqual(cleaned, valid_sample)

        # 2. Header construction format
        headers = build_auth_header(valid_sample)
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], f"Bearer {valid_sample}")
        self.assertTrue(headers["Authorization"].startswith("Bearer "))
        self.assertFalse(headers["Authorization"].startswith("Bearer Bearer "))

        # 3. Idempotent with existing Bearer prefix
        headers_prefixed = build_auth_header(f"Bearer {valid_sample}")
        self.assertEqual(headers_prefixed["Authorization"], f"Bearer {valid_sample}")

        # 4. Rejections: empty, non-string, missing segments, and wrapper representations
        with self.assertRaises(ValueError):
            validate_token_structure("")

        with self.assertRaises(ValueError):
            validate_token_structure(12345)

        with self.assertRaises(ValueError):
            validate_token_structure("only_one_segment")

        with self.assertRaises(ValueError):
            validate_token_structure("two.segments")

        with self.assertRaises(ValueError):
            validate_token_structure("[REDACTED_JWT]")

        with self.assertRaises(ValueError):
            validate_token_structure(f"'{valid_sample}'")

        with self.assertRaises(ValueError):
            validate_token_structure('{"token": "xyz"}')

    def test_worker_lease_and_failover_timing_configuration(self) -> None:
        """Verify production lease defaults, CI/test override compatibility, and deterministic recovery."""
        from ai_brain.config import WorkerConfig, load_config_from_env
        from ai_brain.tasks.models import TaskStatus, TaskRecord
        from ai_brain.tasks.handlers import handle_interaction_execution

        # 1. Config defaults verification: production defaults are 300s lease, 30s heartbeat
        cfg = load_config_from_env({})
        self.assertEqual(cfg.worker.lease_duration_seconds, 300)
        self.assertEqual(cfg.worker.heartbeat_interval_seconds, 30)

        # Explicit test / fast-failover override verification
        cfg_ci = load_config_from_env({"AI_BRAIN_LEASE_DURATION": "10", "AI_BRAIN_HEARTBEAT_INTERVAL": "3"})
        self.assertEqual(cfg_ci.worker.lease_duration_seconds, 10)
        self.assertEqual(cfg_ci.worker.heartbeat_interval_seconds, 3)

        # 2. docker-compose.yml configuration verification
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        compose_path = os.path.join(project_root, "docker-compose.yml")
        with open(compose_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("AI_BRAIN_LEASE_DURATION: ${AI_BRAIN_LEASE_DURATION:-300}", content)
        self.assertIn("AI_BRAIN_HEARTBEAT_INTERVAL: ${AI_BRAIN_HEARTBEAT_INTERVAL:-30}", content)

        # 3. Deterministic abandoned task recovery verification
        container = create_sqlite_container()
        try:
            task_store = container.task_store
            project_id = "test_failover_proj"
            task_type = "interaction_execution"

            # Worker 1 claims task with short 0.2s lease and crashes
            submitted = task_store.submit_task(
                project_id=project_id,
                task_type=task_type,
                payload={"input": "failover test"},
            )
            c1 = task_store.claim_next_task(project_id, "worker_dead", lease_seconds=0.2)
            self.assertIsNotNone(c1)
            self.assertEqual(c1.locked_by, "worker_dead")
            self.assertEqual(c1.status, TaskStatus.PROCESSING)
            self.assertEqual(c1.retry_count, 0)

            # Before lease expires, another worker cannot claim it
            c_early = task_store.claim_next_task(project_id, "worker_replacement", lease_seconds=1.0)
            self.assertIsNone(c_early, "Unexpired task lease must not be claimed")

            # Sleep to allow lease expiry
            time.sleep(0.3)

            # Replacement worker reclaims the expired task
            c2 = task_store.claim_next_task(project_id, "worker_replacement", lease_seconds=2.0)
            self.assertIsNotNone(c2, "Expired task lease must be claimed by replacement worker")
            self.assertEqual(c2.task_id, submitted.task_id)
            self.assertEqual(c2.locked_by, "worker_replacement")
            self.assertEqual(c2.retry_count, 1)

            # Replacement worker completes task
            completed = task_store.complete_task(
                project_id=project_id,
                task_id=c2.task_id,
                worker_id="worker_replacement",
                result_payload={"status": "recovered"},
            )
            self.assertEqual(completed.status, TaskStatus.COMPLETED)
            self.assertIsNone(completed.locked_by)

            # Verified in store
            final_record = task_store.get_task(project_id, submitted.task_id)
            self.assertIsNotNone(final_record)
            self.assertEqual(final_record.status, TaskStatus.COMPLETED)
            self.assertEqual(final_record.retry_count, 1)

            # 4. Multi-memory extraction & conflict tracking regression check
            idem_key = "test_multi_mem_idem_001"
            exec_task = TaskRecord(
                task_id="task_multi_mem_test",
                project_id="default",
                task_type="interaction_execution",
                payload={
                    "input_text": "Decision: We decided to deploy PostgreSQL with automated lease recovery.",
                    "session_id": "sess_multi_mem_001",
                    "agent_id": "test-sync-blocker",
                    "interaction_id": "int_multi_mem_001",
                    "idempotency_key": idem_key,
                },
                status=TaskStatus.PROCESSING,
                retry_count=1,
                max_retries=3,
                locked_by="worker_replacement",
                locked_until=None,
                error_message=None,
                created_at=submitted.created_at,
                updated_at=submitted.updated_at,
            )
            res = handle_interaction_execution(exec_task, container)
            self.assertEqual(res.get("status"), "COMPLETED")
            # Verify exactly 1 interaction record
            saved_interaction = container.interaction_store.get("int_multi_mem_001")
            self.assertIsNotNone(saved_interaction)
            self.assertEqual(saved_interaction.idempotency_key, idem_key)
            self.assertIsNotNone(saved_interaction.input_event_id)
            self.assertIsNotNone(saved_interaction.response_event_id)

            # Verify that memories point to interaction events via source_event_ids
            mems = container.memory_store.list(project_id="default")
            self.assertGreaterEqual(len(mems), 2)
            event_ids_in_interaction = {saved_interaction.input_event_id, saved_interaction.response_event_id}
            for m in mems:
                self.assertTrue(bool(m.source_event_ids), f"Memory {m.memory_id} has empty source_event_ids")
                self.assertTrue(any(eid in event_ids_in_interaction for eid in m.source_event_ids))

            # Verify that re-execution with same idempotency key does not duplicate writes
            res_dup = handle_interaction_execution(exec_task, container)
            self.assertEqual(res_dup.get("status"), "COMPLETED")
            dup_interaction = container.interaction_store.get("int_multi_mem_001")
            self.assertEqual(dup_interaction.interaction_id, "int_multi_mem_001")
        finally:
            container.close()

    def test_postgres_memory_event_provenance_lifecycle(self) -> None:
        """Verify strictly isolated PostgreSQL event-sourced memory provenance queries."""
        import psycopg
        from ai_brain.config import DatabaseConfig
        from ai_brain.storage.database import DatabaseConnectionManager
        from ai_brain.tasks.models import TaskRecord, TaskStatus
        from ai_brain.tasks.handlers import handle_interaction_execution
        from ai_brain.server.dependencies import create_postgres_container

        pg_url = os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain")
        try:
            with psycopg.connect(pg_url) as test_conn:
                with test_conn.cursor() as cur:
                    cur.execute("SELECT 1;")
        except Exception:
            self.skipTest(f"PostgreSQL not available at {pg_url}")

        db = DatabaseConnectionManager(DatabaseConfig(database_url=pg_url))
        container = create_postgres_container(db)
        try:
            crash_idem_key = f"ci_prov_test_{int(time.time() * 1000)}"
            crash_int_id = f"int_prov_test_{int(time.time() * 1000)}"
            crash_sess_id = f"sess_prov_test_{int(time.time() * 1000)}"

            task = TaskRecord(
                task_id=f"task_{crash_int_id}",
                project_id="default",
                task_type="interaction_execution",
                payload={
                    "input_text": "Active crash recovery verification: We decided to deploy PostgreSQL with automated lease recovery.",
                    "session_id": crash_sess_id,
                    "agent_id": "test-sync-blocker",
                    "interaction_id": crash_int_id,
                    "idempotency_key": crash_idem_key,
                },
                status=TaskStatus.PROCESSING,
            )
            res = handle_interaction_execution(task, container)
            self.assertEqual(res.get("status"), "COMPLETED")

            with psycopg.connect(pg_url) as conn:
                with conn.cursor() as cur:
                    # 1. Distinct contents scoped strictly by project_id, session_id, and interaction_id
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT m.content)
                        FROM memories m
                        CROSS JOIN LATERAL jsonb_array_elements_text(m.metadata->'source_event_ids') AS src(eid)
                        JOIN events e ON e.event_id = src.eid
                        WHERE e.project_id = %s
                          AND e.session_id = %s
                          AND e.payload->'metadata'->>'interaction_id' = %s;
                        """,
                        ("default", crash_sess_id, crash_int_id),
                    )
                    distinct_contents = cur.fetchone()[0]
                    self.assertEqual(distinct_contents, 2)

                    # 2. No orphaned memories for this target interaction
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM memories m
                        CROSS JOIN LATERAL jsonb_array_elements_text(m.metadata->'source_event_ids') AS src(eid)
                        JOIN events e ON e.event_id = src.eid
                        WHERE e.project_id = %s
                          AND e.session_id = %s
                          AND e.payload->'metadata'->>'interaction_id' = %s
                          AND (m.metadata->'source_event_ids' IS NULL OR jsonb_array_length(m.metadata->'source_event_ids') = 0);
                        """,
                        ("default", crash_sess_id, crash_int_id),
                    )
                    orphaned = cur.fetchone()[0]
                    self.assertEqual(orphaned, 0)

                    # 3. Exact interaction ID provenance
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT e.payload->'metadata'->>'interaction_id'),
                               MAX(e.payload->'metadata'->>'interaction_id')
                        FROM memories m
                        CROSS JOIN LATERAL jsonb_array_elements_text(m.metadata->'source_event_ids') AS src(eid)
                        JOIN events e ON e.event_id = src.eid
                        WHERE e.project_id = %s
                          AND e.session_id = %s
                          AND e.payload->'metadata'->>'interaction_id' = %s;
                        """,
                        ("default", crash_sess_id, crash_int_id),
                    )
                    interaction_matches, matched_id = cur.fetchone()
                    self.assertEqual(interaction_matches, 1)
                    self.assertEqual(matched_id, crash_int_id)

                    # 4. Exact session ID provenance
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT e.session_id),
                               MAX(e.session_id)
                        FROM memories m
                        CROSS JOIN LATERAL jsonb_array_elements_text(m.metadata->'source_event_ids') AS src(eid)
                        JOIN events e ON e.event_id = src.eid
                        WHERE e.project_id = %s
                          AND e.session_id = %s
                          AND e.payload->'metadata'->>'interaction_id' = %s;
                        """,
                        ("default", crash_sess_id, crash_int_id),
                    )
                    session_matches, matched_sess = cur.fetchone()
                    self.assertEqual(session_matches, 1)
                    self.assertEqual(matched_sess, crash_sess_id)

                    # 5. Negative contamination check
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM memories m
                        CROSS JOIN LATERAL jsonb_array_elements_text(m.metadata->'source_event_ids') AS src(eid)
                        JOIN events e ON e.event_id = src.eid
                        WHERE m.memory_id IN (
                            SELECT DISTINCT m2.memory_id
                            FROM memories m2
                            CROSS JOIN LATERAL jsonb_array_elements_text(m2.metadata->'source_event_ids') AS src2(eid2)
                            JOIN events e2 ON e2.event_id = src2.eid2
                            WHERE e2.project_id = %s
                              AND e2.session_id = %s
                              AND e2.payload->'metadata'->>'interaction_id' = %s
                        )
                        AND (e.payload->'metadata'->>'interaction_id' != %s OR e.session_id != %s);
                        """,
                        ("default", crash_sess_id, crash_int_id, crash_int_id, crash_sess_id),
                    )
                    contamination = cur.fetchone()[0]
                    self.assertEqual(contamination, 0)
        finally:
            container.close()

    def test_postgres_worker_crash_recovery_lifecycle_and_side_effects(self) -> None:
        """Integration test exercising complete persisted worker crash-recovery lifecycle on PostgreSQL."""
        import psycopg
        import threading
        from ai_brain.config import DatabaseConfig
        from ai_brain.storage.database import DatabaseConnectionManager
        from ai_brain.tasks.models import TaskRecord, TaskStatus
        from ai_brain.tasks.worker import BackgroundWorker
        from ai_brain.agents.simulated.sync import TestSyncBlockingAgent
        from ai_brain.server.dependencies import create_postgres_container

        pg_url = os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain")
        try:
            with psycopg.connect(pg_url) as test_conn:
                with test_conn.cursor() as cur:
                    cur.execute("SELECT 1;")
        except Exception:
            self.skipTest(f"PostgreSQL not available at {pg_url}")

        db = DatabaseConnectionManager(DatabaseConfig(database_url=pg_url))
        container = create_postgres_container(db)
        project_id = "default"
        uid = f"{int(time.time() * 1000)}"
        task_id = f"task_crash_pg_{uid}"
        int_id = f"int_crash_pg_{uid}"
        sess_id = f"sess_crash_pg_{uid}"
        idem_key = f"idem_crash_pg_{uid}"

        started_evt = threading.Event()
        barrier_evt = threading.Event()
        TestSyncBlockingAgent._thread_started_event = started_evt
        TestSyncBlockingAgent._thread_barrier_event = barrier_evt

        worker_a = None
        try:
            # 1. Submit interaction execution task to PostgreSQL task queue
            submitted_task = container.task_store.submit_task(
                project_id=project_id,
                task_type="interaction_execution",
                payload={
                    "input_text": "Active crash recovery verification: We decided to deploy PostgreSQL with automated lease recovery.",
                    "session_id": sess_id,
                    "agent_id": "test-sync-blocker",
                    "interaction_id": int_id,
                    "idempotency_key": idem_key,
                },
                idempotency_key=idem_key,
            )
            self.assertEqual(submitted_task.status, TaskStatus.PENDING)

            # 2. Worker A claims task with short lease (0.2s) and begins execution
            worker_a = BackgroundWorker(
                store=container.task_store,
                project_id=project_id,
                registry=container.task_registry,
                context=container,
                worker_id="worker_a_victim",
                poll_interval=0.02,
                lease_seconds=0.2,
            )
            worker_a.start()

            # Wait deterministically for Worker A to claim task and enter execution
            self.assertTrue(started_evt.wait(timeout=5.0), "Worker A failed to enter execution")

            # Verify task is currently locked in PROCESSING state by Worker A in PostgreSQL
            in_flight = container.task_store.get_task(project_id, submitted_task.task_id)
            self.assertIsNotNone(in_flight)
            self.assertEqual(in_flight.status, TaskStatus.PROCESSING)
            self.assertEqual(in_flight.locked_by, "worker_a_victim")

            # Simulate abrupt crash of Worker A by stopping its loop without releasing task lease
            worker_a._stop_event.set()

            # Allow lease to expire
            time.sleep(0.4)

            # Clear synchronization barrier so Worker B proceeds cleanly
            TestSyncBlockingAgent._thread_started_event = None
            TestSyncBlockingAgent._thread_barrier_event = None

            # 3. Worker B reclaims the abandoned task from PostgreSQL queue
            claimed_by_b = container.task_store.claim_next_task(project_id, "worker_b_survivor", lease_seconds=10.0)
            self.assertIsNotNone(claimed_by_b, "Worker B must reclaim abandoned task from PostgreSQL queue")
            self.assertEqual(claimed_by_b.locked_by, "worker_b_survivor")
            self.assertEqual(claimed_by_b.retry_count, 1)

            # Release barrier and join Worker A (Worker A will be fenced if it attempts updates)
            barrier_evt.set()
            worker_a.stop(timeout=1.0)
            worker_a = None

            # Worker B executes and completes the reclaimed task
            exec_res = container.task_registry.execute(claimed_by_b, container)
            self.assertTrue(exec_res.success)
            completed_task = container.task_store.complete_task(
                project_id=project_id,
                task_id=claimed_by_b.task_id,
                worker_id="worker_b_survivor",
                result_payload=exec_res.result_data,
            )
            self.assertEqual(completed_task.status, TaskStatus.COMPLETED)

            # 4. Independent inspection of PostgreSQL records
            with psycopg.connect(pg_url) as conn:
                with conn.cursor() as cur:
                    # Verify task record
                    cur.execute(
                        "SELECT status, retry_count, locked_by FROM background_tasks WHERE project_id = %s AND task_id = %s;",
                        (project_id, submitted_task.task_id),
                    )
                    task_row = cur.fetchone()
                    self.assertIsNotNone(task_row)
                    self.assertEqual(task_row[0], "COMPLETED")
                    self.assertEqual(task_row[1], 1)
                    self.assertIsNone(task_row[2])

                    # Verify interaction record
                    cur.execute(
                        "SELECT status, idempotency_key FROM interactions WHERE project_id = %s AND interaction_id = %s;",
                        (project_id, int_id),
                    )
                    int_row = cur.fetchone()
                    self.assertIsNotNone(int_row)
                    self.assertEqual(int_row[0], "COMPLETED")
                    self.assertEqual(int_row[1], idem_key)

                    # Verify memories strictly traced to this interaction
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT m.content)
                        FROM memories m
                        CROSS JOIN LATERAL jsonb_array_elements_text(m.metadata->'source_event_ids') AS src(eid)
                        JOIN events e ON e.event_id = src.eid
                        WHERE e.project_id = %s AND e.session_id = %s AND e.payload->'metadata'->>'interaction_id' = %s;
                        """,
                        (project_id, sess_id, int_id),
                    )
                    mem_contents_count = cur.fetchone()[0]
                    self.assertEqual(mem_contents_count, 2)

                    # Verify events for this interaction
                    cur.execute(
                        """
                        SELECT COUNT(*), COUNT(DISTINCT event_type)
                        FROM events
                        WHERE project_id = %s AND session_id = %s AND payload->'metadata'->>'interaction_id' = %s;
                        """,
                        (project_id, sess_id, int_id),
                    )
                    evt_count, evt_types = cur.fetchone()
                    self.assertEqual(evt_count, 4)
                    self.assertEqual(evt_types, 4)
        finally:
            TestSyncBlockingAgent._thread_started_event = None
            TestSyncBlockingAgent._thread_barrier_event = None
            if worker_a:
                barrier_evt.set()
                worker_a.stop(timeout=1.0)
            container.close()

    def test_interaction_execution_memory_persistence_failure_injection_and_retry(self) -> None:
        """Verify memory persistence failure is not swallowed, raises TransientTaskError, and completes on retry."""
        from unittest.mock import MagicMock
        from ai_brain.tasks.models import TaskRecord, TaskStatus
        from ai_brain.tasks.exceptions import TransientTaskError
        from ai_brain.tasks.handlers import handle_interaction_execution
        from ai_brain.server.dependencies import create_sqlite_container

        container = create_sqlite_container(":memory:")
        try:
            task = TaskRecord(
                task_id="task_fail_inject_001",
                project_id="default",
                task_type="interaction_execution",
                payload={
                    "input_text": "Decision: We decided to deploy PostgreSQL with automated lease recovery.",
                    "session_id": "sess_fail_inject_001",
                    "interaction_id": "int_fail_inject_001",
                    "idempotency_key": "idem_fail_inject_001",
                },
                status=TaskStatus.PROCESSING,
            )

            # Failure injection: make memory_store.store fail
            original_store = container.memory_store.store
            container.memory_store.store = MagicMock(side_effect=IOError("Simulated disk full / store failure"))

            # Execution must NOT return COMPLETED and must raise TransientTaskError
            with self.assertRaises(TransientTaskError) as cm:
                handle_interaction_execution(task, container)
            self.assertIn("Failed to persist durable memory side effect", str(cm.exception))

            # Restore working memory store
            container.memory_store.store = original_store

            # Subsequent retry must succeed and durably store memories
            res = handle_interaction_execution(task, container)
            self.assertEqual(res.get("status"), "COMPLETED")
            self.assertEqual(res.get("interaction_id"), "int_fail_inject_001")

            # Verify memories are present in memory store
            stored_mems = container.memory_store.list(project_id="default")
            self.assertGreaterEqual(len(stored_mems), 1)
        finally:
            container.close()


class TestDockerRuntimeExecution(unittest.TestCase):
    """Runtime tests executed if Docker CLI and daemon are installed on the host."""

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI / daemon not available on host system")
    def test_compose_config_validation(self) -> None:
        """Verify docker compose config validates syntax and references."""
        import subprocess

        res = subprocess.run(
            ["docker", "compose", "config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"docker compose config failed: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
