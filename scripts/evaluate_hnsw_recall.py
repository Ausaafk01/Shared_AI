"""Reproducible evaluation script comparing pgvector HNSW approximate index search

against exact cosine-similarity search on a deterministic dataset in PostgreSQL 16.

Computes empirical Recall@K across multiple search parameter configurations.
"""

from __future__ import annotations

import math
import os
import random
import sys
import uuid
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.storage.database import DatabaseConnectionManager
from ai_brain.vectors.models import VectorRecord
from ai_brain.vectors.storage.postgres import PostgreSQLVectorStore

load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)
APP_PG_URL = os.environ.get(
    "TEST_POSTGRES_APP_URL",
    f"postgresql://ai_brain_app_user:{os.environ.get('TEST_POSTGRES_APP_PASSWORD', '')}@127.0.0.1:5432/test_ai_brain",
)


def _generate_synthetic_vectors(
    num_vectors: int,
    dimension: int = 384,
    seed: int = 42,
) -> List[List[float]]:
    """Generate deterministic normalized float vectors with multi-cluster structure."""
    rng = random.Random(seed)
    num_clusters = 8
    # Generate cluster centers
    centers: List[List[float]] = []
    for _ in range(num_clusters):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        centers.append([x / norm for x in raw])

    vectors: List[List[float]] = []
    for i in range(num_vectors):
        center = centers[i % num_clusters]
        # Add noise to cluster center
        raw = [c + rng.gauss(0.0, 0.25) for c in center]
        norm = math.sqrt(sum(x * x for x in raw))
        vectors.append([x / norm for x in raw])

    return vectors


def run_hnsw_evaluation(
    dataset_size: int = 500,
    num_queries: int = 25,
    dimension: int = 384,
    k_values: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    """Execute evaluation comparing HNSW approximate search against exact ground truth."""
    cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=APP_PG_URL,
        pool_size=5,
        pool_timeout=5.0,
    )
    admin_cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=3,
        pool_timeout=5.0,
    )
    db = DatabaseConnectionManager(cfg)
    admin_db = DatabaseConnectionManager(admin_cfg)

    # 1. Inspect PostgreSQL and pgvector versions
    with admin_db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_version_str = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            ext_row = cur.fetchone()
            pgvector_version = ext_row[0] if ext_row else "Not Installed"

    project_id = f"proj_eval_{uuid.uuid4().hex[:8]}"
    store = PostgreSQLVectorStore(db, default_project_id=project_id)

    print(f"=== HNSW Recall Evaluation vs Exact Ground Truth ===")
    print(f"PostgreSQL Version: {pg_version_str.split(',')[0]}")
    print(f"pgvector Version  : {pgvector_version}")
    print(f"Target Dataset    : {dataset_size} vectors ({dimension}-d)")
    print(f"Query Count       : {num_queries} queries")
    print(f"Project ID        : {project_id}")
    print(f"Index Parameters  : HNSW (m = 16, ef_construction = 64, metric = cosine)")
    print()

    # 2. Populate dataset
    print(f"Populating {dataset_size} vectors...")
    data_vectors = _generate_synthetic_vectors(dataset_size, dimension=dimension, seed=100)
    for i, vec in enumerate(data_vectors):
        rec = VectorRecord.create(
            record_id=f"rec_eval_{i}_{uuid.uuid4().hex[:6]}",
            project_id=project_id,
            memory_id=f"mem_eval_{i}",
            embedding=vec,
            model_id="eval-model",
            version=1,
            content=f"Evaluation memory item {i}",
            status="ACTIVE",
        )
        store.add_vector(rec)
    print(f"Successfully populated {dataset_size} records.")

    # 3. Generate distinct evaluation queries
    query_vectors = _generate_synthetic_vectors(num_queries, dimension=dimension, seed=999)

    # Parameter sweeps to evaluate
    configurations = [
        {"ef_search": 10, "iterative_scan": "off", "label": "ef_search=10, iterative=off"},
        {"ef_search": 40, "iterative_scan": "off", "label": "ef_search=40 (default), iterative=off"},
        {"ef_search": 40, "iterative_scan": "relaxed_order", "label": "ef_search=40, iterative=relaxed_order"},
        {"ef_search": 100, "iterative_scan": "relaxed_order", "label": "ef_search=100, iterative=relaxed_order"},
    ]

    results: Dict[str, Dict[int, float]] = {}

    for config in configurations:
        ef_s = config["ef_search"]
        iter_s = config["iterative_scan"]
        lbl = config["label"]
        results[lbl] = {}

        for k in k_values:
            total_recall = 0.0

            for q_idx, q_vec in enumerate(query_vectors):
                vec_literal = "[" + ",".join(str(float(x)) for x in q_vec) + "]"

                # Ground truth: Exact sequential scan
                exact_ids: Set[str] = set()
                with db.tenant_context(project_id) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL enable_indexscan = off;")
                        cur.execute(
                            """
                            SELECT memory_id
                            FROM memory_vectors_384
                            WHERE project_id = %s AND status = 'ACTIVE'
                            ORDER BY embedding <=> %s::vector ASC, memory_id ASC
                            LIMIT %s;
                            """,
                            (project_id, vec_literal, k),
                        )
                        exact_ids = {row[0] for row in cur.fetchall()}

                # Approximate HNSW search with configuration
                hnsw_ids: Set[str] = set()
                with db.tenant_context(project_id) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL enable_indexscan = on;")
                        cur.execute("SELECT set_config('hnsw.ef_search', %s, true);", (str(ef_s),))
                        cur.execute("SELECT set_config('hnsw.iterative_scan', %s, true);", (iter_s,))
                        cur.execute(
                            """
                            SELECT memory_id
                            FROM memory_vectors_384
                            WHERE project_id = %s AND status = 'ACTIVE'
                            ORDER BY embedding <=> %s::vector ASC, memory_id ASC
                            LIMIT %s;
                            """,
                            (project_id, vec_literal, k),
                        )
                        hnsw_ids = {row[0] for row in cur.fetchall()}

                # Recall@K = |HNSW & Exact| / |Exact|
                intersect_count = len(hnsw_ids.intersection(exact_ids))
                recall_q = intersect_count / len(exact_ids) if exact_ids else 1.0
                total_recall += recall_q

            avg_recall = total_recall / num_queries
            results[lbl][k] = avg_recall

    # 4. Display results table
    print("--- Empirical Recall@K Results Table ---")
    headers = ["Configuration"] + [f"Recall@{k}" for k in k_values]
    row_fmt = "{:<42} | " + " | ".join(["{:<12}"] * len(k_values))
    print(row_fmt.format(*headers))
    print("-" * 80)
    for lbl, k_scores in results.items():
        score_strs = [f"{k_scores[k] * 100:.2f}%" for k in k_values]
        print(row_fmt.format(lbl, *score_strs))
    print()

    # 5. Cleanup
    with admin_db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_vectors_384 WHERE project_id = %s;", (project_id,))
            cur.execute("DELETE FROM memories WHERE project_id = %s;", (project_id,))
            cur.execute("DELETE FROM projects WHERE project_id = %s;", (project_id,))
        conn.commit()

    db.close()
    admin_db.close()

    return {
        "pg_version": pg_version_str,
        "pgvector_version": pgvector_version,
        "dataset_size": dataset_size,
        "num_queries": num_queries,
        "results": results,
    }


if __name__ == "__main__":
    run_hnsw_evaluation()
