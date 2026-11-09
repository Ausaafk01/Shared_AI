"""Comprehensive, reproducible evaluation comparing pgvector HNSW approximate index search

against exact cosine-distance sequential scan ground truth on 10,000 vectors and 200 queries
in PostgreSQL 16.15 with pgvector 0.8.1.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
import uuid
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.storage.database import DatabaseConnectionManager

load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)
APP_PG_URL = os.environ.get(
    "TEST_POSTGRES_APP_URL",
    f"postgresql://ai_brain_app_user:{os.environ.get('TEST_POSTGRES_APP_PASSWORD', '')}@127.0.0.1:5432/test_ai_brain",
)


def _generate_dataset(
    total_vectors: int = 10000,
    dimension: int = 384,
    seed: int = 42,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """Generate 10,000 deterministic normalized vectors across varied distributions.

    Composition:
    - 4,000 vectors in 10 tight clusters (sigma = 0.05)
    - 4,000 vectors in 20 diffuse clusters (sigma = 0.35)
    - 2,000 vectors in uniform isotropic distribution
    """
    rng = random.Random(seed)

    # 1. 10 Tight cluster centers
    dense_centers = [[rng.gauss(0.0, 1.0) for _ in range(dimension)] for _ in range(10)]
    dense_centers = [[x / math.sqrt(sum(c * c for c in center)) for x in center] for center in dense_centers]

    # 2. 20 Diffuse cluster centers
    diffuse_centers = [[rng.gauss(0.0, 1.0) for _ in range(dimension)] for _ in range(20)]
    diffuse_centers = [[x / math.sqrt(sum(c * c for c in center)) for x in center] for center in diffuse_centers]

    vectors: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []

    # 4,000 dense cluster vectors
    for i in range(4000):
        c = dense_centers[i % 10]
        raw = [x + rng.gauss(0.0, 0.05) for x in c]
        norm = math.sqrt(sum(x * x for x in raw))
        vectors.append([x / norm for x in raw])
        metadata.append({"distribution": "dense_cluster", "cluster_id": i % 10})

    # 4,000 diffuse cluster vectors
    for i in range(4000):
        c = diffuse_centers[i % 20]
        raw = [x + rng.gauss(0.0, 0.35) for x in c]
        norm = math.sqrt(sum(x * x for x in raw))
        vectors.append([x / norm for x in raw])
        metadata.append({"distribution": "diffuse_cluster", "cluster_id": i % 20})

    # 2,000 uniform isotropic vectors
    for i in range(2000):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        vectors.append([x / norm for x in raw])
        metadata.append({"distribution": "uniform_isotropic", "cluster_id": -1})

    return vectors, metadata


def _generate_queries(
    num_queries: int = 200,
    dimension: int = 384,
    seed: int = 999,
) -> List[List[float]]:
    """Generate 200 deterministic query vectors across varied regions."""
    rng = random.Random(seed)

    # Cluster seeds for query generation
    tight_centers = [[rng.gauss(0.0, 1.0) for _ in range(dimension)] for _ in range(10)]
    diffuse_centers = [[rng.gauss(0.0, 1.0) for _ in range(dimension)] for _ in range(20)]

    queries: List[List[float]] = []
    # 80 near tight clusters
    for i in range(80):
        c = tight_centers[i % 10]
        raw = [x + rng.gauss(0.0, 0.1) for x in c]
        norm = math.sqrt(sum(x * x for x in raw))
        queries.append([x / norm for x in raw])

    # 80 near diffuse clusters
    for i in range(80):
        c = diffuse_centers[i % 20]
        raw = [x + rng.gauss(0.0, 0.4) for x in c]
        norm = math.sqrt(sum(x * x for x in raw))
        queries.append([x / norm for x in raw])

    # 40 uniform isotropic
    for i in range(40):
        raw = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        queries.append([x / norm for x in raw])

    return queries


def run_comprehensive_hnsw_evaluation(
    dataset_size: int = 10000,
    num_queries: int = 200,
    dimension: int = 384,
    k_values: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    """Execute rigorous HNSW recall and latency benchmarking against exact sequential scan ground truth."""
    admin_cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=ADMIN_PG_URL,
        pool_size=3,
        pool_timeout=5.0,
    )
    app_cfg = DatabaseConfig(
        backend=StorageBackend.POSTGRES,
        database_url=APP_PG_URL,
        pool_size=5,
        pool_timeout=5.0,
    )
    admin_db = DatabaseConnectionManager(admin_cfg)
    app_db = DatabaseConnectionManager(app_cfg)

    # Inspect versions
    with admin_db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_version_str = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            ext_row = cur.fetchone()
            pgvector_version = ext_row[0] if ext_row else "Not Installed"

    project_id = f"proj_bench_{uuid.uuid4().hex[:8]}"

    print("=" * 80)
    print("AI BRAIN — PRODUCTION-SCALE HNSW RECALL & LATENCY BENCHMARK")
    print("=" * 80)
    print(f"PostgreSQL Version : {pg_version_str.split(',')[0]}")
    print(f"pgvector Extension : {pgvector_version}")
    print(f"Vector Dimensions  : {dimension} (Table: memory_vectors_384)")
    print(f"Dataset Size       : {dataset_size:,} vectors")
    print(f"Query Count        : {num_queries} queries")
    print(f"Tenant Project ID  : {project_id}")
    print(f"HNSW Build Config  : m = 16, ef_construction = 64, ops = vector_cosine_ops")
    print(f"Evaluated K Values : {list(k_values)}")
    print("-" * 80)

    # 1. Generate dataset
    print(f"Generating {dataset_size:,} deterministic multi-distribution vectors (seed=42)...")
    vectors, _ = _generate_dataset(dataset_size, dimension=dimension, seed=42)
    queries = _generate_queries(num_queries, dimension=dimension, seed=999)

    # 2. Populate PostgreSQL
    print(f"Populating PostgreSQL tenant `{project_id}` in batches of 1,000...")
    t_pop_start = time.perf_counter()
    with admin_db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO projects (project_id, name) VALUES (%s, %s);", (project_id, project_id))

            batch_size = 1000
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                # Insert memories
                mem_rows = [
                    (f"mem_bench_{i}", project_id, 1, f"Benchmark content {i}", "ACTIVE")
                    for i in range(start_idx, end_idx)
                ]
                cur.executemany(
                    """
                    INSERT INTO memories (memory_id, project_id, version, content, status)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    mem_rows,
                )

                # Insert vectors
                vec_rows = []
                for i in range(start_idx, end_idx):
                    v_lit = "[" + ",".join(str(x) for x in vectors[i]) + "]"
                    vec_rows.append(
                        (f"vec_bench_{i}", project_id, f"mem_bench_{i}", 1, v_lit, "eval-model", f"Content {i}", "ACTIVE", "{}")
                    )
                cur.executemany(
                    """
                    INSERT INTO memory_vectors_384 (
                        vector_id, project_id, memory_id, version, embedding, model_name, content, status, metadata
                    ) VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s::jsonb);
                    """,
                    vec_rows,
                )
            conn.commit()

            # Ensure statistics are analyzed for accurate cost planner decisions
            cur.execute("ANALYZE memory_vectors_384;")
            conn.commit()
    t_pop_end = time.perf_counter()
    print(f"Populated {dataset_size:,} vectors in {t_pop_end - t_pop_start:.2f}s.")
    print()

    # 3. Query Plan Verification via EXPLAIN
    print("--- Query Plan Verification (EXPLAIN) ---")
    sample_query_literal = "[" + ",".join(str(x) for x in queries[0]) + "]"

    # Verify Approximate HNSW plan
    with app_db.tenant_context(project_id) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_seqscan = off;")
            cur.execute(
                """
                EXPLAIN SELECT memory_id FROM memory_vectors_384
                WHERE project_id = %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 10;
                """,
                (project_id, sample_query_literal),
            )
            hnsw_plan = [r[0] for r in cur.fetchall()]
    print("[Approximate Path EXPLAIN Plan]:")
    for line in hnsw_plan:
        print("  " + line)
    assert any("idx_memory_vectors_384_hnsw" in l for l in hnsw_plan), "Approximate path must use HNSW index"

    # Verify Ground-Truth Exact Scan plan
    with app_db.tenant_context(project_id) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
            cur.execute(
                """
                EXPLAIN SELECT memory_id FROM memory_vectors_384
                WHERE project_id = %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 10;
                """,
                (project_id, sample_query_literal),
            )
            exact_plan = [r[0] for r in cur.fetchall()]
    print("\n[Exact Ground Truth EXPLAIN Plan]:")
    for line in exact_plan:
        print("  " + line)
    assert any("Seq Scan on memory_vectors_384" in l for l in exact_plan), "Ground truth path must perform Seq Scan"
    print("-" * 80)
    print()

    # 4. Compute Ground Truth for all 200 queries across K = (5, 10, 20)
    print(f"Computing exact cosine sequential scan ground truth for {num_queries} queries...")
    t_exact_start = time.perf_counter()
    ground_truth: Dict[Tuple[int, int], List[str]] = {}  # (query_idx, k) -> list of memory_ids

    with app_db.tenant_context(project_id) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
            max_k = max(k_values)
            for q_idx, q_vec in enumerate(queries):
                q_lit = "[" + ",".join(str(x) for x in q_vec) + "]"
                cur.execute(
                    """
                    SELECT memory_id FROM memory_vectors_384
                    WHERE project_id = %s AND status = 'ACTIVE'
                    ORDER BY embedding <=> %s::vector ASC, memory_id ASC
                    LIMIT %s;
                    """,
                    (project_id, q_lit, max_k),
                )
                all_ids = [r[0] for r in cur.fetchall()]
                for k in k_values:
                    ground_truth[(q_idx, k)] = all_ids[:k]
    t_exact_end = time.perf_counter()
    exact_duration = t_exact_end - t_exact_start
    exact_ms_per_q = (exact_duration / num_queries) * 1000.0
    exact_qps = num_queries / exact_duration
    print(f"Exact ground truth computed in {exact_duration:.2f}s ({exact_ms_per_q:.2f} ms/query, {exact_qps:.1f} QPS).")
    print()

    # 5. Benchmark HNSW configurations
    configurations = [
        {"ef_search": 10, "iterative_scan": "off", "label": "ef_search=10, iterative=off"},
        {"ef_search": 40, "iterative_scan": "off", "label": "ef_search=40 (default), iterative=off"},
        {"ef_search": 40, "iterative_scan": "relaxed_order", "label": "ef_search=40, iterative=relaxed_order"},
        {"ef_search": 100, "iterative_scan": "relaxed_order", "label": "ef_search=100, iterative=relaxed_order"},
        {"ef_search": 200, "iterative_scan": "relaxed_order", "label": "ef_search=200, iterative=relaxed_order"},
    ]

    benchmark_records: List[Dict[str, Any]] = []

    print("Executing HNSW parameter sweep across 200 queries...")
    for config in configurations:
        ef_s = config["ef_search"]
        iter_s = config["iterative_scan"]
        lbl = config["label"]

        config_rec: Dict[str, Any] = {
            "label": lbl,
            "ef_search": ef_s,
            "iterative_scan": iter_s,
            "recalls": {},
            "latency_ms": 0.0,
            "qps": 0.0,
        }

        # Measure execution time across all queries at max_k
        t_conf_start = time.perf_counter()
        query_results: Dict[Tuple[int, int], List[str]] = {}

        with app_db.tenant_context(project_id) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL enable_seqscan = off;")
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true);", (str(ef_s),))
                cur.execute("SELECT set_config('hnsw.iterative_scan', %s, true);", (iter_s,))

                for q_idx, q_vec in enumerate(queries):
                    q_lit = "[" + ",".join(str(x) for x in q_vec) + "]"
                    cur.execute(
                        """
                        SELECT memory_id FROM memory_vectors_384
                        WHERE project_id = %s AND status = 'ACTIVE'
                        ORDER BY embedding <=> %s::vector ASC, memory_id ASC
                        LIMIT %s;
                        """,
                        (project_id, q_lit, max_k),
                    )
                    retrieved = [r[0] for r in cur.fetchall()]
                    for k in k_values:
                        query_results[(q_idx, k)] = retrieved[:k]

        t_conf_end = time.perf_counter()
        conf_duration = t_conf_end - t_conf_start
        conf_ms_per_q = (conf_duration / num_queries) * 1000.0
        conf_qps = num_queries / conf_duration
        config_rec["latency_ms"] = conf_ms_per_q
        config_rec["qps"] = conf_qps

        # Compute empirical recall for each K
        for k in k_values:
            total_recall_k = 0.0
            for q_idx in range(num_queries):
                true_set = set(ground_truth[(q_idx, k)])
                pred_set = set(query_results[(q_idx, k)])
                overlap = len(true_set.intersection(pred_set))
                recall_q = overlap / len(true_set) if true_set else 1.0
                total_recall_k += recall_q
            config_rec["recalls"][k] = total_recall_k / num_queries

        benchmark_records.append(config_rec)

    # 6. Format and Display Report
    print("=" * 95)
    print("EMPIRICAL BENCHMARK RESULTS TABLE (10,000 Vectors, 200 Queries)")
    print("=" * 95)
    headers = ["Configuration", "Recall@5", "Recall@10", "Recall@20", "Avg Latency", "Throughput"]
    row_fmt = "{:<38} | {:<10} | {:<10} | {:<10} | {:<11} | {:<10}"
    print(row_fmt.format(*headers))
    print("-" * 95)

    # Ground truth reference row
    print(row_fmt.format(
        "Exact Sequential Scan (Ground Truth)",
        "100.00%",
        "100.00%",
        "100.00%",
        f"{exact_ms_per_q:.2f} ms",
        f"{exact_qps:.1f} QPS",
    ))
    print("-" * 95)

    for rec in benchmark_records:
        r5 = f"{rec['recalls'][5] * 100:.2f}%"
        r10 = f"{rec['recalls'][10] * 100:.2f}%"
        r20 = f"{rec['recalls'][20] * 100:.2f}%"
        lat = f"{rec['latency_ms']:.2f} ms"
        qps = f"{rec['qps']:.1f} QPS"
        print(row_fmt.format(rec["label"], r5, r10, r20, lat, qps))
    print("=" * 95)
    print()

    # 7. Cleanup evaluation project
    print("Cleaning up evaluation test data...")
    with admin_db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_vectors_384 WHERE project_id = %s;", (project_id,))
            cur.execute("DELETE FROM memories WHERE project_id = %s;", (project_id,))
            cur.execute("DELETE FROM projects WHERE project_id = %s;", (project_id,))
            conn.commit()
    print("Cleanup completed.")

    admin_db.close()
    app_db.close()

    return {
        "pg_version": pg_version_str,
        "pgvector_version": pgvector_version,
        "dataset_size": dataset_size,
        "num_queries": num_queries,
        "exact_latency_ms": exact_ms_per_q,
        "exact_qps": exact_qps,
        "records": benchmark_records,
        "hnsw_plan": hnsw_plan,
        "exact_plan": exact_plan,
    }


if __name__ == "__main__":
    run_comprehensive_hnsw_evaluation()
