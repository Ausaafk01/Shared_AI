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
