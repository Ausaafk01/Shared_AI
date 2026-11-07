"""Runtime acceptance test harness for AI Brain Docker deployment (M5.7).

Executes comprehensive runtime acceptance verification on a Docker-capable system:
1. Validates Compose configuration (docker compose config).
2. Builds images and starts full stack (postgres, migration, api, worker, prometheus, grafana).
3. Verifies container health states and service dependencies.
4. Verifies external HTTP probes (/healthz, /readyz, /metrics).
5. Verifies Prometheus target scraping and Grafana dashboard provisioning.
6. Submits live background task via API, confirms worker execution and interaction storage.
7. Simulates worker crash during active task processing and verifies lease recovery and completion.
8. Verifies database-backed exact-once side effects after crash recovery.
9. Verifies PostgreSQL container restart and data persistence.
10. Performs graceful shutdown (default: non-destructive down; --clean-volumes is opt-in).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("docker_acceptance")

API_BASE = "http://localhost:8000"
PROM_BASE = "http://localhost:9090"
GRAFANA_BASE = "http://localhost:3000"


def sanitize_sensitive_data(text: str) -> str:
    """Sanitize JWT tokens, API keys, credentials, and passwords from log and error messages."""
    if not text:
        return ""
    # Redact standard JWTs
    text = re.sub(r'eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+', '[REDACTED_JWT]', text)
    # Redact Bearer authorization headers
    text = re.sub(r'(Bearer\s+)[A-Za-z0-9_\-\.]+', r'\1[REDACTED_TOKEN]', text, flags=re.IGNORECASE)
    # Redact API keys matching format
    text = re.sub(r'agy_[A-Za-z0-9_]{16,}', '[REDACTED_API_KEY]', text)
    # Redact passwords in connection URLs
    text = re.sub(r'://([^:]+):([^@]+)@', r'://\1:[REDACTED_PASSWORD]@', text)
    return text


def run_cmd(args: list[str], check: bool = True) -> Tuple[int, str, str]:
    sanitized_cmd = " ".join(sanitize_sensitive_data(arg) for arg in args)
    logger.info("Executing: %s", sanitized_cmd)
    proc = subprocess.run(args, capture_output=True, text=True)
    clean_stdout = sanitize_sensitive_data(proc.stdout)
    clean_stderr = sanitize_sensitive_data(proc.stderr)
    if check and proc.returncode != 0:
        logger.error(
            "Command failed (rc=%d):\n--- STDOUT ---\n%s\n--- STDERR ---\n%s",
            proc.returncode,
            clean_stdout.strip() or "(empty)",
            clean_stderr.strip() or "(empty)",
        )
        err_detail = clean_stderr.strip() or clean_stdout.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"Command '{sanitized_cmd}' failed (rc={proc.returncode}): {err_detail}")
    return proc.returncode, clean_stdout, clean_stderr


def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> Tuple[int, str, Dict[str, str]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
            resp_headers = dict(resp.headers)
            return resp.status, content, resp_headers
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        return e.code, content, dict(e.headers)


def http_post_json(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> Tuple[int, str, Dict[str, str]]:
    body = json.dumps(data).encode("utf-8")
    hdrs = dict(headers or {})
    hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, content, dict(resp.headers)
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        return e.code, content, dict(e.headers)


def wait_for_http(url: str, expected_status: int = 200, timeout: float = 60.0, step: float = 2.0) -> str:
    start = time.time()
    last_err = ""
    while time.time() - start < timeout:
        try:
            status, text, _ = http_get(url, timeout=3.0)
            if status == expected_status:
                return text
            last_err = f"Status {status} != {expected_status}"
        except Exception as e:
            last_err = str(e)
        time.sleep(step)
    raise TimeoutError(f"Timed out waiting for {url} ({last_err})")


def get_container_auth_token() -> str:
    """Generate a valid OWNER JWT token using the deployed API container's configuration."""
    script = (
        "import sys\n"
        "from ai_brain.config import load_config_from_env\n"
        "from ai_brain.server.dependencies import create_container_from_config\n"
        "from ai_brain.auth import UserRecord, ProjectMemberRecord, ProjectRole, IdentityType\n"
        "\n"
        "cfg = load_config_from_env()\n"
        "container = create_container_from_config(cfg)\n"
        "try:\n"
        "    user_id = 'ci_owner_user'\n"
        "    project_id = 'default'\n"
        "    user = container.auth_store.get_user(user_id)\n"
        "    if not user:\n"
        "        container.auth_store.create_user(UserRecord(user_id=user_id, email='ci_owner@example.com', name='CI Owner'))\n"
        "    member = container.auth_store.get_project_member(project_id, user_id)\n"
        "    if not member:\n"
        "        container.auth_store.add_project_member(\n"
        "            ProjectMemberRecord(member_id='mem_ci_owner', project_id=project_id, user_id=user_id, role=ProjectRole.OWNER)\n"
        "        )\n"
        "    elif member.role != ProjectRole.OWNER:\n"
        "        container.auth_store.update_project_member_role(project_id, user_id, ProjectRole.OWNER)\n"
        "    token = container.token_service.create_access_token(\n"
        "        subject=user_id,\n"
        "        identity_type=IdentityType.USER,\n"
        "        name='CI Owner',\n"
        "        scopes=['*'],\n"
        "        expires_in_seconds=3600,\n"
        "    )\n"
        "    print(token)\n"
        "finally:\n"
        "    container.close()\n"
    )
    cmd = ["docker", "compose", "exec", "-T", "api", "python", "-c", script]
    _, out, _ = run_cmd(cmd)
    token = out.strip().splitlines()[-1].strip()
    return token


def query_postgres(sql: str) -> str:
    """Execute SQL query directly against the PostgreSQL container."""
    cmd = [
        "docker", "compose", "exec", "-T", "postgres",
        "psql", "-U", "ai_brain_user", "-d", "ai_brain", "-t", "-A", "-c", sql,
    ]
    _, out, _ = run_cmd(cmd)
    return out.strip()


def run_acceptance(clean_volumes: bool = False) -> int:
    if not shutil.which("docker"):
        logger.error("FATAL: 'docker' executable not found in PATH.")
        return 1

    logger.info("===========================================================================")
    logger.info("AI Brain Docker Runtime Acceptance Test (clean_volumes=%s)", clean_volumes)
    logger.info("===========================================================================")

    # 1. Validate Compose Configuration
    logger.info("[Step 1] Validating Compose configuration syntax...")
    run_cmd(["docker", "compose", "config", "-q"])
    logger.info("Compose configuration is valid.")

    # 2. Build deployment images
    logger.info("[Step 2] Building Docker images...")
    run_cmd(["docker", "compose", "build"])
    logger.info("Docker build completed successfully.")

    # 3. Start full deployment stack
    logger.info("[Step 3] Starting Docker Compose stack...")
    run_cmd(["docker", "compose", "up", "-d"])

    try:
        # 4. Wait for services to become healthy
        logger.info("[Step 4] Waiting for services to become healthy...")
        wait_for_http(f"{API_BASE}/healthz", expected_status=200, timeout=60.0)
        wait_for_http(f"{API_BASE}/readyz", expected_status=200, timeout=60.0)
        logger.info("API is healthy and ready.")

        # 5. Verify API Probes & Metrics
        logger.info("[Step 5] Checking /healthz, /ready, and /metrics endpoints...")
        status, healthz_body, _ = http_get(f"{API_BASE}/healthz")
        assert status == 200, f"Expected 200 for /healthz, got {status}"
        assert '"status":"ok"' in healthz_body.replace(" ", "")

        status, readyz_body, _ = http_get(f"{API_BASE}/readyz")
        assert status == 200, f"Expected 200 for /readyz, got {status}"
        assert '"status":"ready"' in readyz_body.replace(" ", "")

        status, metrics_body, hdrs = http_get(f"{API_BASE}/metrics")
        assert status == 200, f"Expected 200 for /metrics, got {status}"
        assert "ai_brain_ready 1" in metrics_body
        assert "ai_brain_database_connected 1" in metrics_body
        logger.info("External HTTP probes and Prometheus /metrics verified.")

        # 6. Verify Prometheus Scraping
        logger.info("[Step 6] Verifying Prometheus scraping...")
        wait_for_http(f"{PROM_BASE}/api/v1/targets", expected_status=200, timeout=45.0)
        time.sleep(5.0)  # Allow initial scrape cycle
        _, targets_body, _ = http_get(f"{PROM_BASE}/api/v1/targets")
        targets_json = json.loads(targets_body)
        active_targets = targets_json.get("data", {}).get("activeTargets", [])
        ai_brain_targets = [t for t in active_targets if "ai-brain-api" in str(t.get("labels", {}))]
        assert len(ai_brain_targets) > 0, "Prometheus did not find ai_brain scrape target"
        logger.info("Prometheus successfully configured and scraping AI Brain target.")

        # 7. Verify Grafana Provisioning
        logger.info("[Step 7] Verifying Grafana provisioning...")
        wait_for_http(f"{GRAFANA_BASE}/api/health", expected_status=200, timeout=45.0)
        auth_hdr = {"Authorization": "Basic YWRtaW46YWRtaW4="}  # admin:admin
        status, dash_body, _ = http_get(f"{GRAFANA_BASE}/api/dashboards/uid/ai-brain-overview", headers=auth_hdr)
        assert status == 200, f"Grafana dashboard failed with status {status}"
        assert "AI Brain Production Overview" in dash_body
        logger.info("Grafana dashboard provisioned and loaded successfully.")

        # 8. Live Background Task Submission & Worker Execution
        logger.info("[Step 8] Executing live background task test...")
        auth_token = get_container_auth_token()
        api_hdrs = {"Authorization": f"Bearer {auth_token}"}

        task_payload = {
            "task_type": "interaction_execution",
            "payload": {
                "input_text": "Live Docker Deployment Verification: Test durable background processing.",
                "session_id": "sess_ci_docker_001",
                "agent_id": "agent-coder",
            },
            "idempotency_key": "ci_live_task_key_001",
        }

        status, submit_res, _ = http_post_json(
            f"{API_BASE}/api/v1/projects/default/tasks",
            data=task_payload,
            headers=api_hdrs,
        )
        assert status == 202, f"Task submission failed with status {status}: {submit_res}"
        task_data = json.loads(submit_res)
        task_id = task_data["task_id"]
        logger.info("Submitted task '%s', status: %s", task_id, task_data["status"])

        # Poll until COMPLETED
        start_wait = time.time()
        final_task_data = None
        while time.time() - start_wait < 30.0:
            status, get_res, _ = http_get(f"{API_BASE}/api/v1/projects/default/tasks/{task_id}", headers=api_hdrs)
            assert status == 200, f"Failed to get task status: {status}"
            polled = json.loads(get_res)
            if polled["status"] in ("COMPLETED", "FAILED"):
                final_task_data = polled
                break
            time.sleep(1.0)

        assert final_task_data is not None, "Task did not reach terminal state within 30s"
        assert final_task_data["status"] == "COMPLETED", f"Task failed: {final_task_data.get('error_message')}"
        logger.info("Task '%s' completed successfully by deployed worker fleet.", task_id)

        # Duplicate submission check
        status_dup, _, _ = http_post_json(
            f"{API_BASE}/api/v1/projects/default/tasks",
            data=task_payload,
            headers=api_hdrs,
        )
        assert status_dup == 409, f"Expected 409 for duplicate task, got {status_dup}"
        logger.info("Idempotency protection verified: duplicate task submission rejected with 409.")

        # 9. Worker Crash Recovery During Active Execution
        logger.info("[Step 9] Executing worker crash-recovery during active task execution...")
        crash_idempotency_key = f"ci_crash_recovery_{int(time.time())}"
        crash_interaction_id = f"int_crash_{int(time.time())}"
        barrier_path = "/tmp/ai_brain_test_sync_barrier"
        proc_path = "/tmp/ai_brain_test_sync_processing"

        # Arm the test-only synchronization barrier exclusively in the test harness
        run_cmd(["docker", "compose", "exec", "-T", "worker", "rm", "-f", proc_path])
        run_cmd(["docker", "compose", "exec", "-T", "worker", "touch", barrier_path])
        logger.info("Test synchronization barrier armed in worker container (%s).", barrier_path)

        crash_payload = {
            "task_type": "interaction_execution",
            "payload": {
                "input_text": "Active crash recovery verification: We decided to deploy PostgreSQL with automated lease recovery.",
                "session_id": "sess_ci_crash_002",
                "agent_id": "test-sync-blocker",
                "interaction_id": crash_interaction_id,
            },
            "idempotency_key": crash_idempotency_key,
        }

        status, crash_res, _ = http_post_json(
            f"{API_BASE}/api/v1/projects/default/tasks",
            data=crash_payload,
            headers=api_hdrs,
        )
        assert status == 202, f"Failed to submit crash recovery task: {crash_res}"
        crash_task_id = json.loads(crash_res)["task_id"]
        logger.info("Crash recovery task submitted: %s", crash_task_id)

        # Poll until worker claims task and enters PROCESSING state
        logger.info("Waiting for worker to claim task and enter PROCESSING state...")
        claimed_and_processing = False
        start_claim = time.time()
        while time.time() - start_claim < 15.0:
            status, get_res, _ = http_get(f"{API_BASE}/api/v1/projects/default/tasks/{crash_task_id}", headers=api_hdrs)
            if status == 200:
                polled_state = json.loads(get_res)
                if polled_state["status"] == "PROCESSING":
                    claimed_and_processing = True
                    logger.info("Confirmed task '%s' is actively PROCESSING. Abruptly killing worker container mid-flight...", crash_task_id)
                    break
            time.sleep(0.05)

        assert claimed_and_processing, f"Task '{crash_task_id}' was not claimed into PROCESSING within 15s"

        # Kill worker container abruptly while actively executing task
        run_cmd(["docker", "compose", "kill", "worker"])
        logger.info("Worker container terminated during active task execution.")

        # Verify task is currently locked in PostgreSQL
        db_status = query_postgres(
            f"SELECT status FROM background_tasks WHERE task_id = '{crash_task_id}';"
        )
        assert db_status == "PROCESSING", f"Expected PROCESSING in DB, got: {db_status}"
        logger.info("Database confirms task '%s' was abandoned in PROCESSING state by dead worker.", crash_task_id)

        # Remove dead container to ensure clean /tmp filesystem without barrier file
        run_cmd(["docker", "compose", "rm", "-f", "worker"])

        # Re-start replacement worker container
        logger.info("Starting replacement worker container...")
        run_cmd(["docker", "compose", "up", "-d", "worker"])

        # Wait for replacement worker to reclaim task and complete it
        start_wait = time.time()
        recovered = False
        while time.time() - start_wait < 45.0:
            status, get_res, _ = http_get(f"{API_BASE}/api/v1/projects/default/tasks/{crash_task_id}", headers=api_hdrs)
            if status == 200:
                t_json = json.loads(get_res)
                if t_json["status"] == "COMPLETED":
                    recovered = True
                    break
            time.sleep(1.0)

        assert recovered, "Replacement worker failed to recover and complete crash-recovery task"
        logger.info("Replacement worker claimed abandoned task and successfully completed it.")

        # Strengthened database-backed verification of exactly-once side effects
        # 1. Exactly one completed task
        task_db_count = query_postgres(
            f"SELECT COUNT(*) FROM background_tasks WHERE task_id = '{crash_task_id}' AND status = 'COMPLETED';"
        )
        assert task_db_count == "1", f"Expected exactly 1 completed task in DB, got: {task_db_count}"

        # 2. Exactly one interaction record
        int_db_count = query_postgres(
            f"SELECT COUNT(*) FROM interactions WHERE idempotency_key = '{crash_idempotency_key}';"
        )
        assert int_db_count == "1", f"Expected exactly 1 interaction record in DB, got: {int_db_count}"

        # 3. Exactly one corresponding set of memory side effects
        mem_db_count = query_postgres(
            f"SELECT COUNT(*) FROM memories WHERE content ILIKE '%automated lease recovery%';"
        )
        assert mem_db_count == "1", f"Expected exactly 1 memory record in DB, got: {mem_db_count}"

        # 4. No duplicate event writes
        evt_db_count = query_postgres(
            f"SELECT COUNT(*) FROM events WHERE payload->>'idempotency_key' = '{crash_idempotency_key}';"
        )
        assert evt_db_count == "1", f"Expected exactly 1 event record in DB, got: {evt_db_count}"

        # 5. External HTTP verification of resulting interaction
        status, int_body, _ = http_get(
            f"{API_BASE}/api/v1/projects/default/interactions/{crash_interaction_id}",
            headers=api_hdrs,
        )
        assert status == 200, f"Expected 200 retrieving interaction, got {status}"
        logger.info("Database-backed task completion and exactly-once side effects (task, interaction, memory, event) verified.")

        # 10. Database Persistence Across Container Restarts
        logger.info("[Step 10] Testing PostgreSQL data persistence across restarts...")
        run_cmd(["docker", "compose", "restart", "postgres"])
        wait_for_http(f"{API_BASE}/readyz", expected_status=200, timeout=45.0)

        # Confirm previously completed tasks are still readable and intact
        status, persisted_res, _ = http_get(f"{API_BASE}/api/v1/projects/default/tasks/{task_id}", headers=api_hdrs)
        assert status == 200
        persisted_json = json.loads(persisted_res)
        assert persisted_json["status"] == "COMPLETED"
        logger.info("PostgreSQL persistence verified across container restarts.")

    finally:
        # 11. Graceful Shutdown (Default: Non-destructive 'docker compose down'; -v requires opt-in)
        logger.info("[Step 11] Performing graceful shutdown (clean_volumes=%s)...", clean_volumes)
        down_cmd = ["docker", "compose", "down"]
        if clean_volumes:
            down_cmd.append("-v")
        run_cmd(down_cmd)
        logger.info("Docker Compose deployment cleanly stopped.")

    logger.info("===========================================================================")
    logger.info("ALL DOCKER RUNTIME ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    logger.info("===========================================================================")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Brain Docker Runtime Acceptance Test Harness")
    parser.add_argument(
        "--clean-volumes",
        action="store_true",
        default=False,
        help="Explicitly destroy persistent Docker volumes on shutdown (disabled by default for safety).",
    )
    args = parser.parse_args()
    return run_acceptance(clean_volumes=args.clean_volumes)


if __name__ == "__main__":
    sys.exit(main())
