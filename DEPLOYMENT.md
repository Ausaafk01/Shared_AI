# AI Brain — Production Deployment & Observability Guide (M5.7)

This document provides complete instructions for deploying, configuring, monitoring, and operating the AI Brain REST API and Background Worker Fleet using Docker, Docker Compose, PostgreSQL (with pgvector), Prometheus, and Grafana.

---

## 1. Architecture Overview

```
                          ┌──────────────────────────┐
                          │   External HTTP / SSE    │
                          └─────────────┬────────────┘
                                        │ (Port 8000)
                                        ▼
                      ┌────────────────────────────────────┐
                      │            AI Brain API            │
                      │  (FastAPI / Uvicorn, Non-Root)     │
                      └─────────┬────────────────┬─────────┘
                                │                │
                        /metrics│                │ SQL / Advisory Locks
                                ▼                ▼
┌──────────────────┐  Scrapes ┌────────────┐   ┌────────────────────────────┐
│     Grafana      ├─────────►│ Prometheus │   │         PostgreSQL         │
│   (Port 3000)    │          │(Port 9090) │   │  (pgvector:pg16, RLS)      │
└──────────────────┘          └────────────┘   └──────────────▲─────────────┘
                                                              │
                                                              │ Leases / Queue
                                               ┌──────────────┴─────────────┐
                                               │ Background Worker Fleet    │
                                               │ (M5.6 Durable Tasks, Tini) │
                                               └────────────────────────────┘
```

The system architecture consists of:
1. **AI Brain API**: Asynchronous FastAPI service handling interaction lifecycles, Server-Sent Events (SSE) streaming, knowledge graph updates, hybrid retrieval, and authentication/RBAC.
2. **Background Worker Fleet**: Autonomous worker daemon claiming and executing durable background tasks (`interaction_execution`, `workflow_step`, `memory_maintenance`) using database-backed leases and atomic fencing.
3. **Database Migration Runner**: Dedicated one-shot Alembic container that executes transactional migrations (`alembic upgrade head`) before API or worker replicas initialize, preventing race conditions.
4. **PostgreSQL 16 with pgvector**: Relational storage engine with Row-Level Security (RLS), range-partitioned event tables, HNSW vector indexes, and advisory locks.
5. **Prometheus**: Time-series metrics scraper collecting API throughput/latency, task queue depths, worker states, and database health via `/metrics`.
6. **Grafana**: Pre-provisioned dashboards visualizing service metrics in real time.

---

## 2. Prerequisites

- **Docker Engine**: Version 24.0 or higher.
- **Docker Compose**: Version 2.20 or higher.
- **System Memory**: Minimum 4 GB RAM recommended for API, workers, and database.
- **Disk Storage**: At least 10 GB persistent volume storage.

---

## 3. Environment Configuration & Secret Setup

1. Copy the template `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Configure environment variables in `.env`:
   - `POSTGRES_PASSWORD`: Generate a cryptographically secure random password:
     ```bash
     openssl rand -hex 24
     ```
   - `AI_BRAIN_JWT_SECRET`: High-entropy key (at least 32 bytes) for JWT signing and verification.
   - `GRAFANA_ADMIN_PASSWORD`: Strong password for Grafana administration.

> [!WARNING]
> Never commit `.env` or production credentials to version control. The `.dockerignore` and `.gitignore` files automatically exclude environment files from image builds and repositories.

---

## 4. Building and Starting Services

### Build and Launch Full Stack
Run the entire stack in detached mode:
```bash
docker compose up -d --build
```

### Startup Sequence & Dependency Graph
Docker Compose orchestrates services in strict order:
1. `postgres` starts and completes healthchecks (`pg_isready`).
2. `migration` container launches `alembic upgrade head` to apply database schema updates.
3. Upon migration success (`condition: service_completed_successfully`), `api` and `worker` containers initialize.
4. `prometheus` starts and begins scraping `api:8000/metrics`.
5. `grafana` starts and connects to `prometheus:9090` using automated provisioning.

---

## 5. Running Database Migrations

### Automatic Migrations
The `migration` service in `docker-compose.yml` runs automatically on startup:
```bash
docker compose run --rm migration alembic upgrade head
```

### Manual Migration Commands
To inspect migration status or perform manual upgrades/downgrades:
```bash
# Check current migration revision
docker compose run --rm migration alembic current

# Upgrade to latest schema
docker compose run --rm migration alembic upgrade head

# Downgrade by one revision (if testing rollbacks)
docker compose run --rm migration alembic downgrade -1
```

---

## 6. Health and Readiness Checks

AI Brain provides standardized Kubernetes- and Docker-compatible probes:

| Endpoint | Method | Purpose | Response |
|---|---|---|---|
| `/healthz` | GET | Liveness Probe | `{"status": "ok", "version": "1.0.0", "timestamp": "..."}` |
| `/readyz` | GET | Readiness Probe | `{"status": "ready", "backend": "postgres", "database": "connected"}` |
| `/health` | GET | Liveness Alias | Same as `/healthz` |
| `/ready` | GET | Readiness Alias | Same as `/readyz` |

### Testing Probes
```bash
# Check API liveness
curl -i http://localhost:8000/healthz

# Check API readiness
curl -i http://localhost:8000/readyz

# Check worker daemon health
docker compose exec worker python -m ai_brain.tasks.cli healthcheck
```

---

## 7. Accessing Prometheus and Grafana

- **AI Brain API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus Metrics Scraping Endpoint**: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Prometheus Web UI**: [http://localhost:9090](http://localhost:9090)
- **Grafana Dashboard UI**: [http://localhost:3000](http://localhost:3000)
  - Default Username: `admin`
  - Password: As set in `GRAFANA_ADMIN_PASSWORD` (default: `admin`)
  - Dashboard: Navigate to **Dashboards** → **AI Brain** → **AI Brain Production Overview**.

---

## 8. Viewing Worker and Queue Health

Inspect background task queue status and worker fleet health via CLI:
```bash
# View live task queue statistics
docker compose exec worker python -m ai_brain.tasks.cli status

# View worker logs in real time
docker compose logs -f worker
```

### Exposed Metrics in Prometheus / Grafana
- `ai_brain_task_queue_depth{status="PENDING"}`: Number of pending tasks waiting for a worker.
- `ai_brain_task_queue_depth{status="PROCESSING"}`: Number of tasks currently actively executing.
- `ai_brain_worker_active_count`: Count of active worker threads in the fleet.
- `ai_brain_task_retries_total`: Cumulative count of transient task retries.
- `ai_brain_task_failures_total`: Cumulative count of terminal task failures.
- `ai_brain_task_dead_letter_total`: Depth of dead-letter queue.

---

## 9. Scaling API and Worker Services

Docker Compose allows horizontal scaling of worker and API replicas:

### Scale Background Workers
Scale workers to 3 containers (each running 4 threads by default = 12 concurrent workers):
```bash
docker compose up -d --scale worker=3
```
Because workers utilize database-backed atomic reservations (`SELECT ... FOR UPDATE SKIP LOCKED` and advisory fencing), multiple worker replicas safely coordinate without task duplication or race conditions.

### Scale API Instances
Scale API to 3 containers behind an external load balancer:
```bash
docker compose up -d --scale api=3
```

---

## 10. Backup and Restore Procedures

### Database Backup (Logical Dump)
Create a compressed PostgreSQL backup with schema and data:
```bash
docker compose exec -T postgres pg_dump -U ai_brain_user -d ai_brain -F c > ai_brain_backup_$(date +%Y%m%d_%H%M%S).dump
```

### Database Restore
Restore into a fresh PostgreSQL instance:
```bash
docker compose exec -T postgres pg_restore -U ai_brain_user -d ai_brain --clean --if-exists < ai_brain_backup.dump
```

---

## 11. Shutdown and Recovery Procedures

### Graceful Shutdown
To stop the fleet without interrupting running tasks:
```bash
docker compose down
```
Docker signals `SIGTERM` to `api` and `worker`. The `WorkerFleet` stops claiming new tasks and waits up to 10 seconds for active tasks to complete or safely release their leases.

### Worker Crash Recovery
If a worker container crashes or is forcibly terminated (`SIGKILL`), its active task leases expire automatically after `AI_BRAIN_LEASE_DURATION` (production default: 300 seconds; fast failover / CI test override: 10 seconds). Surviving workers automatically reclaim expired tasks, and atomic fencing tokens prevent zombie workers from writing stale updates. Workers renew active leases continuously every `AI_BRAIN_HEARTBEAT_INTERVAL` (production default: 30 seconds; CI override: 3 seconds).

---

## 12. Common Deployment Troubleshooting

### Issue 1: API returns `503 Service Unavailable` or `/readyz` reports unready
- **Cause**: Database connection failure or incomplete migrations.
- **Remedy**:
  ```bash
  docker compose logs postgres
  docker compose logs migration
  ```
  Ensure PostgreSQL is healthy and Alembic migration completed successfully.

### Issue 2: `DuplicateTaskError` in worker logs
- **Cause**: Normal idempotency protection. Client submitted duplicate task with same idempotency key.
- **Remedy**: Expected behavior; idempotency coordination prevented duplicate writes.

### Issue 3: Prometheus cannot scrape `api:8000/metrics`
- **Cause**: Network partition or API container crashed.
- **Remedy**: Check container status:
  ```bash
  docker compose ps
  curl -i http://localhost:8000/metrics
  ```
