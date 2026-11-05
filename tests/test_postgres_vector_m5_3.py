"""Integration tests for AI Brain Phase 5 Milestone 5.3: pgvector Vector Storage & Hybrid Retrieval.

Acceptance Test Matrix:
1. All three vector tables (384, 768, 1536) exist with pgvector extension and HNSW indexes.
2. Vector CRUD operations across all three supported dimensions (384, 768, 1536).
3. Dimension validation and strict rejection of unsupported dimensions (DimensionMismatchError).
4. Malformed vector rejection (NaN, Inf, non-numeric values -> VectorValidationError).
5. Duplicate vector prevention and fail_if_exists semantics (DuplicateVectorError).
6. Memory versioning, supersession lifecycle, and status filtering.
7. HNSW nearest-neighbor similarity search with threshold and top_k pruning.
8. Deterministic tie-breaking on matching similarity scores.
9. Project-level multi-tenant isolation via Row-Level Security (RLS).
10. Transaction rollback on failure and connection sanitization.
11. Full integration with VectorMemoryManager and domain Memory instances.
12. Full integration with RetrievalEngine for vector and hybrid search via RRF fusion.
"""

from __future__ import annotations

import math
import os
import unittest
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple

import psycopg

from ai_brain.config import DatabaseConfig, StorageBackend, load_env_file
from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.graph.models import GraphNode, GraphRelationship
from ai_brain.graph.storage.postgres import PostgreSQLGraphStore
from ai_brain.graph.types import NodeType, RelationshipType
from ai_brain.memories.models import Memory
from ai_brain.memories.storage.postgres import PostgreSQLMemoryStore
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.retrieval.engine import RetrievalEngine
from ai_brain.retrieval.graph_retriever import GraphRetriever
from ai_brain.retrieval.models import ProcessedQuery, QueryFilters
from ai_brain.retrieval.types import RetrievalMethod
from ai_brain.retrieval.vector_retriever import VectorRetriever
from ai_brain.storage.database import DatabaseConnectionManager
from ai_brain.vectors.exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    PgvectorUnavailableError,
    ProjectIsolationError,
    VectorNotFoundError,
    VectorValidationError,
)
from ai_brain.vectors.index import VectorIndex
from ai_brain.vectors.manager import VectorMemoryManager
from ai_brain.vectors.models import VectorRecord
from ai_brain.vectors.providers.mock import MockEmbeddingProvider
from ai_brain.vectors.storage.postgres import (
    DIMENSION_TABLE_MAP,
    SUPPORTED_DIMENSIONS,
    PostgreSQLVectorStore,
)

# Load local test environment variables
load_env_file()

ADMIN_PG_URL = os.environ.get(
    "TEST_POSTGRES_ADMIN_URL",
    os.environ.get("TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test_ai_brain"),
)
APP_PG_URL = os.environ.get(
    "TEST_POSTGRES_APP_URL",
    f"postgresql://ai_brain_app_user:{os.environ.get('TEST_POSTGRES_APP_PASSWORD', '')}@127.0.0.1:5432/test_ai_brain",
)


def _check_postgres_live() -> tuple[bool, str, Optional[DatabaseConnectionManager]]:
    """Probe whether live PostgreSQL instance is reachable."""
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
        return True, str(ver[0]) if ver else "Connected", mgr
    except Exception as e:
        return False, f"PostgreSQL instance unreachable at {ADMIN_PG_URL}: {e}", None


PG_AVAILABLE, PG_STATUS, ADMIN_MGR = _check_postgres_live()


def _make_unit_vector(dimension: int, active_index: int = 0) -> List[float]:
    """Create a unit vector with 1.0 at active_index and 0.0 elsewhere."""
    vec = [0.0] * dimension
    vec[active_index % dimension] = 1.0
    return vec


def _make_dense_vector(dimension: int, seed: float = 1.0) -> List[float]:
    """Create a normalized dense float vector."""
    raw = [math.sin(seed + i * 0.1) for i in range(dimension)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@unittest.skipUnless(PG_AVAILABLE, f"Live PostgreSQL test skipped: {PG_STATUS}")
class TestPostgresVectorM53(unittest.TestCase):
    """Integration test suite executing all M5.3 pgvector features against live server."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app_cfg = DatabaseConfig(
            backend=StorageBackend.POSTGRES,
            database_url=APP_PG_URL,
            pool_size=10,
            pool_timeout=5.0,
        )
        cls.app_db = DatabaseConnectionManager(cls.app_cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "app_db") and cls.app_db:
            cls.app_db.close()
        if ADMIN_MGR:
            ADMIN_MGR.close()

    def setUp(self) -> None:
        self.project_id = f"proj_vec_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    # 1. Schema & Extension Verification
    # -------------------------------------------------------------------------
    def test_01_all_three_vector_tables_and_hnsw_indexes_exist(self) -> None:
        """Verify all three dimensional tables, pgvector extension, and HNSW indexes."""
        with ADMIN_MGR.connection() as conn:
            with conn.cursor() as cur:
                # 1. pgvector extension is installed
                cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                ext = cur.fetchone()
                self.assertIsNotNone(ext, "pgvector extension must be installed in PostgreSQL")
                self.assertEqual(ext[0], "vector")

                # 2. Check all three dimensional tables exist with RLS enabled and forced
                for dim, tbl in DIMENSION_TABLE_MAP.items():
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s;
                        """,
                        (tbl,),
                    )
                    row = cur.fetchone()
                    self.assertIsNotNone(row, f"Vector table {tbl} must exist")

                    # Check RLS is enabled and forced in pg_class
                    cur.execute(
                        """
                        SELECT c.relrowsecurity, c.relforcerowsecurity
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = 'public' AND c.relname = %s;
                        """,
                        (tbl,),
                    )
                    sec_row = cur.fetchone()
                    self.assertIsNotNone(sec_row, f"Security settings for {tbl} must exist")
                    self.assertTrue(sec_row[0], f"RLS must be enabled on {tbl}")
                    self.assertTrue(sec_row[1], f"RLS must be forced on {tbl}")

                # 3. Check HNSW indexes exist
                expected_indexes = [
                    "idx_memory_vectors_384_hnsw",
                    "idx_memory_vectors_768_hnsw",
                    "idx_memory_vectors_1536_hnsw",
                ]
                for idx_name in expected_indexes:
                    cur.execute(
                        """
                        SELECT indexname, indexdef FROM pg_indexes
                        WHERE schemaname = 'public' AND indexname = %s;
                        """,
                        (idx_name,),
                    )
                    idx_row = cur.fetchone()
                    self.assertIsNotNone(idx_row, f"HNSW index {idx_name} must exist")
                    self.assertIn("hnsw", idx_row[1].lower(), f"{idx_name} must use hnsw access method")
                    self.assertIn("vector_cosine_ops", idx_row[1].lower(), f"{idx_name} must use vector_cosine_ops")

    # -------------------------------------------------------------------------
    # 2. Vector CRUD Across Dimensions
    # -------------------------------------------------------------------------
    def test_02_vector_crud_across_all_three_dimensions(self) -> None:
        """Verify adding, getting, updating, and deleting vectors for 384, 768, and 1536 dimensions."""
        store = PostgreSQLVectorStore(self.app_db)

        for dim in SUPPORTED_DIMENSIONS:
            mem_id = f"mem_crud_{dim}_{uuid.uuid4().hex[:6]}"
            rec_id = f"rec_crud_{dim}_{uuid.uuid4().hex[:6]}"
            vec = _make_dense_vector(dim, seed=float(dim))

            record = VectorRecord.create(
                record_id=rec_id,
                project_id=self.project_id,
                memory_id=mem_id,
                embedding=vec,
                model_id=f"model-{dim}d",
                version=1,
                content=f"Content for {dim}-d vector",
                status="ACTIVE",
                metadata={"test_dim": dim},
            )

            stored = store.add_vector(record)

            self.assertIsNotNone(stored.record_id)
            self.assertEqual(stored.project_id, self.project_id)
            self.assertEqual(stored.memory_id, mem_id)
            self.assertEqual(stored.dimensions, dim)
            self.assertEqual(len(stored.embedding), dim)
            self.assertEqual(stored.version, 1)
            self.assertEqual(stored.status, "ACTIVE")
            self.assertEqual(stored.metadata.get("test_dim"), dim)

            # Retrieve by record_id
            fetched = store.get_vector(stored.record_id, project_id=self.project_id)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.record_id, stored.record_id)
            self.assertEqual(len(fetched.embedding), dim)

            # Retrieve by memory tuple
            by_mem = store.get_vector_by_memory(self.project_id, mem_id, 1, f"model-{dim}d")
            self.assertIsNotNone(by_mem)
            self.assertEqual(by_mem.record_id, stored.record_id)

            # Update status and metadata
            updated_rec = stored.with_updates(
                status="ARCHIVED",
                metadata={"test_dim": dim, "updated": True},
            )
            store.update_vector(updated_rec)
            re_fetched = store.get_vector(stored.record_id, project_id=self.project_id)
            self.assertIsNotNone(re_fetched)
            self.assertEqual(re_fetched.status, "ARCHIVED")
            self.assertTrue(re_fetched.metadata.get("updated"))

            # Delete by record_id
            deleted = store.delete_vector(stored.record_id, project_id=self.project_id)
            self.assertTrue(deleted)
            self.assertIsNone(store.get_vector(stored.record_id, project_id=self.project_id))

    # -------------------------------------------------------------------------
    # 3. Dimension Validation & Mismatch Rejection
    # -------------------------------------------------------------------------
    def test_03_dimension_validation_and_mismatch_rejection(self) -> None:
        """Verify unsupported dimensions and mismatching lengths raise DimensionMismatchError."""
        store = PostgreSQLVectorStore(self.app_db)

        # Unsupported dimension 512
        vec_512 = [0.1] * 512
        rec_512 = VectorRecord.create(
            record_id="rec_512",
            project_id=self.project_id,
            memory_id="mem_invalid_dim",
            embedding=vec_512,
            model_id="model-512",
            version=1,
            content="512 dimension content",
        )
        with self.assertRaises(DimensionMismatchError):
            store.add_vector(rec_512)

        # Dimension parameter (384) does not match embedding length (768)
        vec_768 = [0.1] * 768
        with self.assertRaises(DimensionMismatchError):
            VectorRecord(
                record_id="rec_mismatch",
                project_id=self.project_id,
                memory_id="mem_mismatch",
                embedding=tuple(vec_768),
                dimensions=384,
                model_id="model-384",
                version=1,
                content="mismatch",
            )

    # -------------------------------------------------------------------------
    # 4. Malformed Vector Rejection
    # -------------------------------------------------------------------------
    def test_04_malformed_vector_rejection(self) -> None:
        """Verify NaN, Inf, and non-numeric vectors raise VectorValidationError."""
        store = PostgreSQLVectorStore(self.app_db)

        # NaN element
        nan_vec = [0.1] * 384
        nan_vec[10] = float("nan")
        nan_rec = VectorRecord(
            record_id="rec_nan",
            project_id=self.project_id,
            memory_id="mem_nan",
            dimensions=384,
            embedding=tuple(nan_vec),
            model_id="model-384",
            version=1,
            content="nan content",
        )
        with self.assertRaises(VectorValidationError):
            store.add_vector(nan_rec)

        # Inf element
        inf_vec = [0.1] * 384
        inf_vec[20] = float("inf")
        inf_rec = VectorRecord(
            record_id="rec_inf",
            project_id=self.project_id,
            memory_id="mem_inf",
            dimensions=384,
            embedding=tuple(inf_vec),
            model_id="model-384",
            version=1,
            content="inf content",
        )
        with self.assertRaises(VectorValidationError):
            store.add_vector(inf_rec)

        # Non-numeric element in direct store validation
        str_vec: Any = [0.1] * 384
        str_vec[5] = "not_a_number"
        with self.assertRaises((VectorValidationError, ValueError)):
            store._validate_vector(str_vec, 384)

    # -------------------------------------------------------------------------
    # 5. Duplicate Vector Handling & fail_if_exists Semantics
    # -------------------------------------------------------------------------
    def test_05_duplicate_vector_handling_and_fail_if_exists(self) -> None:
        """Verify duplicate (project_id, memory_id, version, model_name) handling."""
        store = PostgreSQLVectorStore(self.app_db)
        mem_id = f"mem_dup_{uuid.uuid4().hex[:6]}"
        vec = _make_dense_vector(384, seed=42.0)

        rec1 = VectorRecord.create(
            record_id=f"rec_orig_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id=mem_id,
            embedding=vec,
            model_id="test-model",
            version=1,
            content="Original record",
        )
        stored1 = store.add_vector(rec1, fail_if_exists=True)

        dup = VectorRecord.create(
            record_id=f"rec_dup_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id=mem_id,
            embedding=vec,
            model_id="test-model",
            version=1,
            content="Duplicate record attempt",
        )

        # Duplicate with fail_if_exists=True raises DuplicateVectorError
        with self.assertRaises(DuplicateVectorError):
            store.add_vector(dup, fail_if_exists=True)

        # Duplicate with fail_if_exists=False returns the existing record
        stored2 = store.add_vector(dup, fail_if_exists=False)
        self.assertEqual(stored1.record_id, stored2.record_id)

    # -------------------------------------------------------------------------
    # 6. Memory Versioning & Supersession Lifecycle
    # -------------------------------------------------------------------------
    def test_06_memory_versioning_and_superseded_status(self) -> None:
        """Verify older memory versions are marked SUPERSEDED and excluded from default searches."""
        store = PostgreSQLVectorStore(self.app_db)
        mem_id = f"mem_ver_{uuid.uuid4().hex[:6]}"
        vec_v1 = _make_unit_vector(384, active_index=0)
        vec_v2 = _make_unit_vector(384, active_index=1)

        # Insert Version 1
        rec_v1 = VectorRecord.create(
            record_id=f"rec_v1_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id=mem_id,
            embedding=vec_v1,
            model_id="test-model",
            version=1,
            status="ACTIVE",
            content="Version 1 content",
        )
        store.add_vector(rec_v1)

        # Supersede Version 1 and insert Version 2
        store.update_vector(rec_v1.with_updates(status="SUPERSEDED"))

        rec_v2 = VectorRecord.create(
            record_id=f"rec_v2_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id=mem_id,
            embedding=vec_v2,
            model_id="test-model",
            version=2,
            status="ACTIVE",
            content="Version 2 content",
        )
        store.add_vector(rec_v2)

        # Default similarity search only includes ACTIVE statuses
        result_default = store.search_similarity(
            query_vector=vec_v1,
            project_id=self.project_id,
            allowed_statuses=["ACTIVE"],
        )
        returned_mems = [m.memory_id for m in result_default.matches]
        returned_vers = [m.version for m in result_default.matches if m.memory_id == mem_id]
        self.assertIn(mem_id, returned_mems)
        self.assertEqual(returned_vers, [2], "Default search must return v2, not superseded v1")

        # Explicit search for SUPERSEDED memories includes Version 1
        result_superseded = store.search_similarity(
            query_vector=vec_v1,
            project_id=self.project_id,
            allowed_statuses=["SUPERSEDED"],
        )
        sup_vers = [m.version for m in result_superseded.matches if m.memory_id == mem_id]
        self.assertEqual(sup_vers, [1], "Explicit status search must retrieve superseded v1")

    # -------------------------------------------------------------------------
    # 7. HNSW Nearest-Neighbor Search & Threshold Filtering
    # -------------------------------------------------------------------------
    def test_07_hnsw_nearest_neighbor_search_and_threshold(self) -> None:
        """Verify HNSW nearest-neighbor search ranks by cosine distance with threshold filtering."""
        store = PostgreSQLVectorStore(self.app_db)

        # Target query vector
        query_vec = _make_unit_vector(384, active_index=0)

        # Vector A: Identical direction (cosine sim ~ 1.0)
        rec_a = VectorRecord.create(
            record_id=f"rec_a_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id="mem_target_a",
            embedding=query_vec,
            model_id="test-model",
            version=1,
            content="Target A identical",
        )
        store.add_vector(rec_a)

        # Vector B: Partially aligned (cosine sim ~ 0.707)
        partially_aligned = [0.0] * 384
        partially_aligned[0] = 0.70710678
        partially_aligned[1] = 0.70710678
        rec_b = VectorRecord.create(
            record_id=f"rec_b_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id="mem_target_b",
            embedding=partially_aligned,
            model_id="test-model",
            version=1,
            content="Target B partial",
        )
        store.add_vector(rec_b)

        # Vector C: Orthogonal (cosine sim ~ 0.0)
        orthogonal = _make_unit_vector(384, active_index=50)
        rec_c = VectorRecord.create(
            record_id=f"rec_c_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id="mem_target_c",
            embedding=orthogonal,
            model_id="test-model",
            version=1,
            content="Target C orthogonal",
        )
        store.add_vector(rec_c)

        # Search top_k = 2 without threshold
        res = store.search_similarity(
            query_vector=query_vec,
            project_id=self.project_id,
            top_k=2,
        )
        self.assertEqual(len(res.matches), 2)
        self.assertEqual(res.matches[0].memory_id, "mem_target_a")
        self.assertAlmostEqual(res.matches[0].score, 1.0, places=3)
        self.assertEqual(res.matches[1].memory_id, "mem_target_b")
        self.assertAlmostEqual(res.matches[1].score, 0.707, places=2)

        # Search with similarity_threshold = 0.85 (should return only mem_target_a)
        res_thresh = store.search_similarity(
            query_vector=query_vec,
            project_id=self.project_id,
            similarity_threshold=0.85,
        )
        self.assertEqual(len(res_thresh.matches), 1)
        self.assertEqual(res_thresh.matches[0].memory_id, "mem_target_a")

    # -------------------------------------------------------------------------
    # 8. Deterministic Tie-Breaking
    # -------------------------------------------------------------------------
    def test_08_deterministic_tie_breaking(self) -> None:
        """Verify tie-breaking: identical score -> memory_id ASC -> version DESC."""
        store = PostgreSQLVectorStore(self.app_db)
        same_vec = _make_unit_vector(384, active_index=5)

        # Add records with same vector and different memory_id / versions
        store.add_vector(
            VectorRecord.create(
                record_id=f"rec_beta_1_{uuid.uuid4().hex[:6]}",
                project_id=self.project_id,
                memory_id="mem_beta",
                embedding=same_vec,
                model_id="test-model",
                version=1,
                content="Beta v1",
            )
        )
        store.add_vector(
            VectorRecord.create(
                record_id=f"rec_beta_2_{uuid.uuid4().hex[:6]}",
                project_id=self.project_id,
                memory_id="mem_beta",
                embedding=same_vec,
                model_id="test-model",
                version=2,
                content="Beta v2",
            )
        )
        store.add_vector(
            VectorRecord.create(
                record_id=f"rec_alpha_1_{uuid.uuid4().hex[:6]}",
                project_id=self.project_id,
                memory_id="mem_alpha",
                embedding=same_vec,
                model_id="test-model",
                version=1,
                content="Alpha v1",
            )
        )

        res = store.search_similarity(
            query_vector=same_vec,
            project_id=self.project_id,
            allowed_statuses=["ACTIVE"],
            top_k=10,
        )

        # Ordering should be: mem_alpha (v1) -> mem_beta (v2) -> mem_beta (v1)
        # Because:
        # Score is equal for all.
        # "mem_alpha" < "mem_beta" (memory_id ASC).
        # For "mem_beta", v2 > v1 (version DESC).
        mem_version_order = [(m.memory_id, m.version) for m in res.matches]
        expected_order = [
            ("mem_alpha", 1),
            ("mem_beta", 2),
            ("mem_beta", 1),
        ]
        self.assertEqual(mem_version_order[:3], expected_order)

    # -------------------------------------------------------------------------
    # 9. Tenant Isolation via Row-Level Security
    # -------------------------------------------------------------------------
    def test_09_tenant_isolation_via_rls(self) -> None:
        """Verify strict multi-tenant isolation: project_a data is completely hidden from project_b."""
        store = PostgreSQLVectorStore(self.app_db)
        project_a = f"tenant_a_{uuid.uuid4().hex[:6]}"
        project_b = f"tenant_b_{uuid.uuid4().hex[:6]}"

        vec = _make_dense_vector(384, seed=123.0)

        store.add_vector(
            VectorRecord.create(
                record_id=f"rec_ta_{uuid.uuid4().hex[:6]}",
                project_id=project_a,
                memory_id="mem_tenant_a",
                embedding=vec,
                model_id="test-model",
                version=1,
                content="Confidential project A data",
            )
        )
        store.add_vector(
            VectorRecord.create(
                record_id=f"rec_tb_{uuid.uuid4().hex[:6]}",
                project_id=project_b,
                memory_id="mem_tenant_b",
                embedding=vec,
                model_id="test-model",
                version=1,
                content="Confidential project B data",
            )
        )

        # Search under project_a: only project_a data visible
        res_a = store.search_similarity(query_vector=vec, project_id=project_a)
        records_a = [m.memory_id for m in res_a.matches]
        self.assertIn("mem_tenant_a", records_a)
        self.assertNotIn("mem_tenant_b", records_a)

        # Search under project_b: only project_b data visible
        res_b = store.search_similarity(query_vector=vec, project_id=project_b)
        records_b = [m.memory_id for m in res_b.matches]
        self.assertIn("mem_tenant_b", records_b)
        self.assertNotIn("mem_tenant_a", records_b)

        # Cross-tenant direct insert rejection: setting context to project_a but inserting project_b
        with self.assertRaises(Exception):
            with self.app_db.tenant_context(project_a) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_vectors_384 (
                            vector_id, project_id, memory_id, version, embedding, model_name
                        ) VALUES (%s, %s, %s, %s, %s::vector, %s);
                        """,
                        (f"vec_leak_{uuid.uuid4().hex[:6]}", project_b, "mem_leak", 1, "[" + ",".join(map(str, vec)) + "]", "model"),
                    )

    # -------------------------------------------------------------------------
    # 10. Transaction Rollback & Connection Pool Sanitization
    # -------------------------------------------------------------------------
    def test_10_transaction_rollback_and_connection_cleanup(self) -> None:
        """Verify transaction rollback removes uncommitted vectors and resets tenant context."""
        store = PostgreSQLVectorStore(self.app_db)
        mem_id = f"mem_tx_{uuid.uuid4().hex[:6]}"
        vec = _make_unit_vector(384, active_index=7)

        # Failing transaction block
        try:
            with self.app_db.tenant_context(self.project_id) as conn:
                with conn.transaction():
                    store._ensure_project_and_memory(conn, self.project_id, mem_id, 1, "test content", "ACTIVE")
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO memory_vectors_384 (
                                vector_id, project_id, memory_id, version, embedding, model_name
                            ) VALUES (%s, %s, %s, %s, %s::vector, %s);
                            """,
                            (f"vec_fail_{uuid.uuid4().hex[:6]}", self.project_id, mem_id, 1, "[" + ",".join(map(str, vec)) + "]", "model"),
                        )
                    # Force exception
                    raise RuntimeError("Simulated transaction crash")
        except RuntimeError:
            pass

        # Verify no vector record was persisted
        records = store.get_vectors_by_project(self.project_id)
        self.assertEqual(len(records), 0, "Rolled-back vector must not be persisted")

        # Verify connection returned to pool is sanitized
        conn = self.app_db.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW app.current_project_id;")
                row = cur.fetchone()
                val = row[0] if row else ""
                self.assertIn(val, ("", "none"), "Sanitized connection must not leak tenant context")
        finally:
            self.app_db.return_connection(conn)

    # -------------------------------------------------------------------------
    # 11. VectorMemoryManager Integration
    # -------------------------------------------------------------------------
    def test_11_vector_memory_manager_integration(self) -> None:
        """Verify VectorMemoryManager works seamlessly with PostgreSQLVectorStore."""
        store = PostgreSQLVectorStore(self.app_db)
        provider = MockEmbeddingProvider(dimension=384, model_id="mock-minilm-384")
        manager = VectorMemoryManager(storage=store, provider=provider)

        mem = Memory.create(
            memory_id=f"mem_mgr_{uuid.uuid4().hex[:6]}",
            content="Architectural decision: Use PostgreSQL pgvector for semantic retrieval",
            project_id=self.project_id,
            memory_type=MemoryType.DECISION,
            status=MemoryStatus.ACTIVE,
            source_agent_ids=("agent:architect",),
        )

        # Index memory
        rec = manager.index_memory(mem)
        self.assertEqual(rec.memory_id, mem.memory_id)
        self.assertEqual(rec.project_id, self.project_id)
        self.assertEqual(rec.dimensions, 384)

        # Search via VectorMemoryManager
        search_res = manager.search(
            "PostgreSQL pgvector semantic retrieval",
            project_id=self.project_id,
            top_k=5,
        )
        self.assertGreater(len(search_res.matches), 0)
        top = search_res.top_match
        self.assertIsNotNone(top)
        self.assertEqual(top.memory_id, mem.memory_id)
        self.assertGreater(top.score, 0.5)

    # -------------------------------------------------------------------------
    # 12. RetrievalEngine Hybrid Search Integration
    # -------------------------------------------------------------------------
    def test_12_retrieval_engine_hybrid_search_integration(self) -> None:
        """Verify RetrievalEngine coordinates vector and graph retrieval with RRF hybrid fusion."""
        # Setup Postgres vector storage adapter
        vec_store = PostgreSQLVectorStore(self.app_db)
        mem_store = PostgreSQLMemoryStore(self.app_db)

        provider = MockEmbeddingProvider(dimension=384, model_id="mock-minilm-384")
        vec_index = VectorIndex(storage=vec_store, provider=provider)
        vec_retriever = VectorRetriever(vec_index)

        graph = KnowledgeGraph()
        graph_retriever = GraphRetriever(graph)

        # Setup RetrievalEngine
        engine = RetrievalEngine(
            vector_retriever=vec_retriever,
            graph_retriever=graph_retriever,
        )

        # Create target memory
        target_mem_id = f"mem_hybrid_{uuid.uuid4().hex[:6]}"
        mem = Memory.create(
            memory_id=target_mem_id,
            content="PostgreSQL database persistence and relational queries",
            project_id=self.project_id,
            memory_type=MemoryType.FACT,
            status=MemoryStatus.ACTIVE,
            importance=0.9,
            confidence=0.95,
        )
        mem_store.store(mem)

        # Index vector for memory in PostgreSQL pgvector
        vec = provider.embed_text(mem.content)
        vec_rec = VectorRecord.create(
            record_id=f"rec_hyb_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id=target_mem_id,
            embedding=vec,
            model_id=provider.model_id,
            content=mem.content,
            status="ACTIVE",
        )
        vec_store.add_vector(vec_rec)

        # Add graph nodes and relationship pointing to memory
        node_concept = GraphNode.create(
            node_id=f"node_concept_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            node_type=NodeType.CONCEPT,
            label="PostgreSQL",
            metadata={"domain": "database"},
        )
        node_mem = GraphNode.create(
            node_id=f"memory:{target_mem_id}",
            project_id=self.project_id,
            node_type=NodeType.MEMORY,
            label="PostgreSQL Persistence Memory",
            source_ref=target_mem_id,
        )
        graph.add_node(node_concept)
        graph.add_node(node_mem)

        rel = GraphRelationship.create(
            relationship_id=f"rel_{target_mem_id}",
            project_id=self.project_id,
            source_id=node_concept.node_id,
            target_id=node_mem.node_id,
            relationship_type=RelationshipType.RELATED_TO,
        )
        graph.add_relationship(rel)

        # 1. Test Vector-only retrieval using pgvector
        res_vec = engine.retrieve(
            query="PostgreSQL database query",
            project_id=self.project_id,
            method=RetrievalMethod.VECTOR,
            top_k=5,
        )
        self.assertGreater(len(res_vec.results), 0)
        self.assertEqual(res_vec.results[0].memory_id, target_mem_id)
        self.assertIsNotNone(res_vec.results[0].vector_score)

        # 2. Test Hybrid retrieval with RRF fusion
        res_hybrid = engine.retrieve(
            query="PostgreSQL database query",
            project_id=self.project_id,
            method=RetrievalMethod.HYBRID,
            top_k=5,
        )
        self.assertGreater(len(res_hybrid.results), 0)
        self.assertEqual(res_hybrid.results[0].memory_id, target_mem_id)
        self.assertIsNotNone(res_hybrid.results[0].vector_score)
        self.assertEqual(res_hybrid.method, RetrievalMethod.HYBRID)
        self.assertIn("method", res_hybrid.to_dict())

    # -------------------------------------------------------------------------
    # 13. Missing pgvector Extension Handling & Explicit Opt-In Fallback
    # -------------------------------------------------------------------------
    def test_13_missing_pgvector_extension_error_and_explicit_opt_in_fallback(self) -> None:
        """Verify missing pgvector extension raises PgvectorUnavailableError unless explicitly opted-in."""
        # Case 1: Default behavior (allow_in_memory_fallback=False) raises PgvectorUnavailableError
        store_strict = PostgreSQLVectorStore(self.app_db, allow_in_memory_fallback=False)
        store_strict._has_vector_ext = False

        vec = _make_dense_vector(384, seed=1.0)
        rec = VectorRecord.create(
            record_id=f"rec_nofallback_{uuid.uuid4().hex[:6]}",
            project_id=self.project_id,
            memory_id="mem_nofallback",
            embedding=vec,
            model_id="test-model",
            content="test no fallback",
        )

        with self.assertRaises(PgvectorUnavailableError):
            store_strict.add_vector(rec)

        with self.assertRaises(PgvectorUnavailableError):
            store_strict.search_similarity(query_vector=vec, project_id=self.project_id)

        # Case 2: Explicit opt-in (allow_in_memory_fallback=True) allows in-memory calculation
        store_fallback = PostgreSQLVectorStore(self.app_db, allow_in_memory_fallback=True)
        store_fallback._has_vector_ext = False

        # Add record with live store so record exists in table
        store_live = PostgreSQLVectorStore(self.app_db)
        store_live.add_vector(rec)

        # Searching with store_fallback executes in-memory fallback successfully
        fallback_res = store_fallback.search_similarity(query_vector=vec, project_id=self.project_id)
        self.assertGreater(len(fallback_res.matches), 0)
        self.assertEqual(fallback_res.matches[0].memory_id, "mem_nofallback")

    # -------------------------------------------------------------------------
    # 14. HNSW Search Under Multi-Tenant Isolation with Distractor Projects
    # -------------------------------------------------------------------------
    def test_14_hnsw_search_under_tenant_isolation_with_distractor_projects(self) -> None:
        """Verify HNSW search under multi-tenant isolation returns complete top_k without cross-tenant leakage."""
        target_project = f"proj_target_{uuid.uuid4().hex[:6]}"
        distractor_project = f"proj_dist_{uuid.uuid4().hex[:6]}"
        store = PostgreSQLVectorStore(self.app_db, default_project_id=target_project)

        # Base query vector
        query_vec = [1.0] + [0.0] * 383

        # Add 30 distractor records in distractor_project that are extremely close to query_vec
        for i in range(30):
            d_vec = [0.999] + [0.0] * 383
            d_vec[1] = 0.001 * (i + 1)
            norm = math.sqrt(sum(x * x for x in d_vec))
            d_vec = [x / norm for x in d_vec]
            store.add_vector(
                VectorRecord.create(
                    record_id=f"rec_dist_{i}_{uuid.uuid4().hex[:6]}",
                    project_id=distractor_project,
                    memory_id=f"mem_dist_{i}",
                    embedding=d_vec,
                    model_id="eval-model",
                    content=f"Distractor content {i}",
                )
            )

        # Add 10 records in target_project that are slightly further
        for i in range(10):
            t_vec = [0.95] + [0.0] * 383
            t_vec[2] = 0.01 * (i + 1)
            norm = math.sqrt(sum(x * x for x in t_vec))
            t_vec = [x / norm for x in t_vec]
            store.add_vector(
                VectorRecord.create(
                    record_id=f"rec_tgt_{i}_{uuid.uuid4().hex[:6]}",
                    project_id=target_project,
                    memory_id=f"mem_tgt_{i}",
                    embedding=t_vec,
                    model_id="eval-model",
                    content=f"Target content {i}",
                )
            )

        # Execute search under target_project with top_k = 10
        res = store.search_similarity(
            query_vector=query_vec,
            project_id=target_project,
            top_k=10,
        )

        # 1. Exactly 10 results returned (no loss of candidates due to project filtering)
        self.assertEqual(len(res.matches), 10, "Target project search must return all 10 eligible candidates")

        # 2. Every single result strictly belongs to target_project
        for m in res.matches:
            self.assertEqual(m.project_id, target_project)
            self.assertTrue(m.memory_id.startswith("mem_tgt_"))

        # 3. Distractor records are completely absent
        retrieved_ids = {m.memory_id for m in res.matches}
        for i in range(30):
            self.assertNotIn(f"mem_dist_{i}", retrieved_ids)


if __name__ == "__main__":
    unittest.main()
