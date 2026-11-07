"""CLI and daemon entrypoint for AI Brain Background Worker Fleet (M5.7).

Usage:
    python -m ai_brain.tasks.cli run [--concurrency N] [--project-id PROJ]
    python -m ai_brain.tasks.cli healthcheck [--heartbeat-file PATH] [--max-age SECONDS]
    python -m ai_brain.tasks.cli status
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import tempfile
import time
from typing import Optional

from ai_brain.config import AppConfig, load_config_from_env
from ai_brain.observability.logging import setup_structured_logging
from ai_brain.server.dependencies import create_container_from_config
from ai_brain.tasks.fleet import WorkerFleet
from ai_brain.tasks.handlers import register_default_handlers

logger = logging.getLogger("ai_brain.tasks.cli")

DEFAULT_HEARTBEAT_FILE = os.path.join(tempfile.gettempdir(), "ai_brain_worker_heartbeat")


def get_heartbeat_path(custom_path: Optional[str] = None) -> str:
    return custom_path or os.environ.get("AI_BRAIN_WORKER_HEARTBEAT_FILE") or DEFAULT_HEARTBEAT_FILE


def write_heartbeat(path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{time.time()}\n")
    except Exception as e:
        logger.warning("Failed to touch heartbeat file '%s': %s", path, e)


def cmd_healthcheck(args: argparse.Namespace) -> int:
    """Verify that worker daemon is active by checking heartbeat file timestamp."""
    hb_path = get_heartbeat_path(args.heartbeat_file)
    max_age = args.max_age or 60.0

    if not os.path.isfile(hb_path):
        print(f"HEALTHCHECK FAILED: Heartbeat file '{hb_path}' does not exist.", file=sys.stderr)
        return 1

    try:
        mtime = os.path.getmtime(hb_path)
        age = time.time() - mtime
        if age > max_age:
            print(
                f"HEALTHCHECK FAILED: Heartbeat file is stale ({age:.1f}s old > {max_age}s threshold).",
                file=sys.stderr,
            )
            return 1

        print(f"HEALTHCHECK OK: Heartbeat file '{hb_path}' age is {age:.1f}s.")
        return 0
    except Exception as e:
        print(f"HEALTHCHECK ERROR: {e}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print current background task queue statistics."""
    cfg = load_config_from_env()
    container = create_container_from_config(cfg)
    try:
        if not container.task_store:
            print("ERROR: No task store configured.", file=sys.stderr)
            return 1
        stats = container.task_store.get_queue_stats(args.project_id)
        print("=== AI Brain Task Queue Statistics ===")
        for k, v in stats.items():
            print(f"  {k:15}: {v}")
        return 0
    finally:
        container.close()


def cmd_run(args: argparse.Namespace) -> int:
    """Start and supervise the background worker fleet."""
    cfg = load_config_from_env()
    setup_structured_logging(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        use_json=os.environ.get("AI_BRAIN_LOG_FORMAT", "json").lower() == "json",
    )

    concurrency = args.concurrency or cfg.worker.worker_concurrency
    project_id = args.project_id or cfg.security.default_project_id
    poll_interval = args.poll_interval or cfg.worker.poll_interval_seconds
    lease_timeout = args.lease_timeout or cfg.worker.lease_duration_seconds
    hb_path = get_heartbeat_path(args.heartbeat_file)

    logger.info(
        "Initializing AI Brain Worker Fleet (concurrency=%d, project_id='%s', poll_interval=%.2fs, lease_timeout=%.1fs)",
        concurrency,
        project_id,
        poll_interval,
        lease_timeout,
    )

    container = create_container_from_config(cfg)
    if not container.task_store:
        logger.error("Failed to initialize worker fleet: task store is not available.")
        container.close()
        return 1

    if container.task_registry is None:
        from ai_brain.tasks.registry import TaskHandlerRegistry
        container.task_registry = TaskHandlerRegistry()
    register_default_handlers(container.task_registry)

    fleet = WorkerFleet(
        store=container.task_store,
        registry=container.task_registry,
        project_id=project_id,
        concurrency=concurrency,
        poll_interval=poll_interval,
        lease_timeout=lease_timeout,
        fleet_id=args.fleet_id or f"fleet_daemon_{os.getpid()}",
        context=container,
    )

    stop_requested = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        logger.info("Signal %s received: initiating graceful worker fleet shutdown...", signum)
        stop_requested = True

    # Register standard UNIX and Windows signals
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except Exception:
                pass

    try:
        fleet.start()
        write_heartbeat(hb_path)
        logger.info("Worker fleet '%s' running with %d workers.", fleet.fleet_id, fleet.current_concurrency)

        last_hb = time.time()
        while not stop_requested:
            time.sleep(1.0)
            now = time.time()
            if now - last_hb >= 5.0:
                write_heartbeat(hb_path)
                last_hb = now

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received: shutting down...")
    finally:
        logger.info("Stopping worker fleet...")
        fleet.stop(timeout=10.0, graceful=True)
        container.close()
        if os.path.exists(hb_path):
            try:
                os.remove(hb_path)
            except Exception:
                pass
        logger.info("Worker fleet cleanly stopped.")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Brain Background Worker CLI")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run background worker fleet daemon")
    run_parser.add_argument("--concurrency", "-c", type=int, default=None, help="Number of concurrent worker threads")
    run_parser.add_argument("--project-id", "-p", type=str, default=None, help="Target project ID")
    run_parser.add_argument("--poll-interval", type=float, default=None, help="Polling interval in seconds")
    run_parser.add_argument("--lease-timeout", type=float, default=None, help="Lease timeout duration in seconds")
    run_parser.add_argument("--fleet-id", type=str, default=None, help="Custom identifier for worker fleet")
    run_parser.add_argument("--heartbeat-file", type=str, default=None, help="Path to heartbeat timestamp file")

    # 'healthcheck' subcommand
    hc_parser = subparsers.add_parser("healthcheck", help="Check worker heartbeat health")
    hc_parser.add_argument("--heartbeat-file", type=str, default=None, help="Path to heartbeat file")
    hc_parser.add_argument("--max-age", type=float, default=60.0, help="Maximum allowed heartbeat age in seconds")

    # 'status' subcommand
    status_parser = subparsers.add_parser("status", help="Display background task queue statistics")
    status_parser.add_argument("--project-id", "-p", type=str, default=None, help="Filter by project ID")

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "healthcheck":
        sys.exit(cmd_healthcheck(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    else:
        # Default to 'run' if no subcommand provided
        sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
