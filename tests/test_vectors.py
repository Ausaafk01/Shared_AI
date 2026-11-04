"""
Comprehensive test suite for AI Brain Vector Memory subsystem.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest

from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType
from ai_brain.vectors import (
    DimensionMismatchError,
    DuplicateVectorError,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderError,
    MockEmbeddingProvider,
    ProjectIsolationError,
    SQLiteVectorStore,
    VectorError,
    VectorIndex,
    VectorMatch,
    VectorMemoryManager,
    VectorNotFoundError,
    VectorRecord,
    VectorSearchResult,
    VectorStorageError,
    VectorStore,
    VectorValidationError,
    compute_cosine_similarity,
)


class TestVectorMemory(unittest.TestCase):
    """Test suite covering embedding providers, vector storage, similarity search, and memory integration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_vectors.db"
        self.provider = MockEmbeddingProvider(dimension=64, model_id="test-mock-64")
        self.store = SQLiteVectorStore(database_path=self.db_path)
        self.index = VectorIndex(storage=self.store, provider=self.provider)
        self.manager = VectorMemoryManager(storage=self.store, provider=self.provider)

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    # =========================================================================
    # Section 1: Embedding Provider Interface & Mock Implementation
    # =========================================================================

    def test_mock_embedding_generation_and_properties(self) -> None:
        """Verify dimension, model_id, determinism, and normalization of MockEmbeddingProvider."""
        self.assertEqual(self.provider.dimension, 64)
        self.assertEqual(self.provider.model_id, "test-mock-64")

        text = "The project uses PostgreSQL as its primary database."
        vec1 = self.provider.embed_text(text)
        vec2 = self.provider.embed_text(text)

        # Dimension validation
        self.assertEqual(len(vec1), 64)
        # Determinism
        self.assertEqual(vec1, vec2)

        # L2-normalization (unit length)
        norm = sum(x * x for x in vec1)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_embedding_batch_generation(self) -> None:
        """Verify batch embedding generation matches individual outputs."""
        texts = ["Text one", "Text two", "Text three"]
        batch_vecs = self.provider.embed_batch(texts)
        self.assertEqual(len(batch_vecs), 3)
        for t, v in zip(texts, batch_vecs):
            self.assertEqual(v, self.provider.embed_text(t))

    def test_embedding_empty_and_invalid_inputs(self) -> None:
        """Verify empty and invalid inputs raise VectorValidationError."""
        with self.assertRaises(VectorValidationError):
            self.provider.embed_text("")

        with self.assertRaises(VectorValidationError):
            self.provider.embed_text("   \n\t  ")

        with self.assertRaises(VectorValidationError):
            self.provider.embed_text(None)  # type: ignore

        with self.assertRaises(VectorValidationError):
            self.provider.embed_text(123)  # type: ignore

    def test_vector_dimension_validation(self) -> None:
        """Verify mismatched and non-numeric vector inputs are rejected."""
        # Mismatched dimension (e.g. 3 dimensions instead of 64)
        with self.assertRaises(DimensionMismatchError):
            self.provider.validate_embedding_vector([1.0, 2.0, 3.0])

        # NaN or Infinite values
        nan_vec = [0.0] * 64
        nan_vec[0] = float("nan")
        with self.assertRaises(VectorValidationError):
            self.provider.validate_embedding_vector(nan_vec)

        inf_vec = [0.0] * 64
        inf_vec[0] = float("inf")
        with self.assertRaises(VectorValidationError):
            self.provider.validate_embedding_vector(inf_vec)

    def test_provider_failure_handling(self) -> None:
        """Verify EmbeddingProviderError is raised when provider encounters a failure."""
        failing_provider = MockEmbeddingProvider(dimension=64, simulate_failure=True)
        with self.assertRaises(EmbeddingProviderError):
            failing_provider.embed_text("Test query")

        with self.assertRaises(EmbeddingProviderError):
            failing_provider.embed_batch(["Test batch"])

    # =========================================================================
    # Section 2: Vector Storage & Persistence
    # =========================================================================

    def test_vector_record_creation_and_immutability(self) -> None:
        """Verify VectorRecord fields, serialization, and immutability."""
        vec = self.provider.embed_text("Sample memory content")
        rec = VectorRecord.create(
            record_id="rec-1",
            memory_id="mem-1",
            project_id="proj-1",
            version=1,
            model_id=self.provider.model_id,
            embedding=vec,
            content="Sample memory content",
            status="ACTIVE",
            metadata={"domain": "database"},
        )
        self.assertEqual(rec.record_id, "rec-1")
        self.assertEqual(rec.memory_id, "mem-1")
        self.assertEqual(rec.dimensions, 64)
        self.assertEqual(rec.status, "ACTIVE")

        # Immutability check
        with self.assertRaises(AttributeError):
            rec.content = "New content"  # type: ignore

        # Serialization roundtrip
        j = rec.to_json()
        restored = VectorRecord.from_json(j)
        self.assertEqual(restored.record_id, rec.record_id)
        self.assertEqual(restored.embedding, rec.embedding)

    def test_storage_crud_operations(self) -> None:
        """Verify adding, retrieving, updating, and deleting vector records."""
        vec = self.provider.embed_text("Authentication with JWT")
        rec = VectorRecord.create(
            record_id="rec-auth",
            memory_id="mem-auth",
            project_id="proj-1",
            version=1,
            model_id=self.provider.model_id,
            embedding=vec,
            content="Authentication with JWT",
        )
        stored = self.store.add_vector(rec)
        self.assertEqual(stored.record_id, "rec-auth")

        # Retrieve by record_id
        fetched = self.store.get_vector("rec-auth")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.memory_id, "mem-auth")

        # Retrieve by memory tuple
        by_mem = self.store.get_vector_by_memory("proj-1", "mem-auth", 1, self.provider.model_id)
        self.assertIsNotNone(by_mem)
        self.assertEqual(by_mem.record_id, "rec-auth")

        # Update record status
        updated = fetched.with_updates(status="SUPERSEDED")
        self.store.update_vector(updated)
        re_fetched = self.store.get_vector("rec-auth")
        self.assertEqual(re_fetched.status, "SUPERSEDED")

        # Delete record
        self.assertTrue(self.store.delete_vector("rec-auth"))
        self.assertIsNone(self.store.get_vector("rec-auth"))

    def test_duplicate_vector_prevention(self) -> None:
        """Verify duplicate records for the same memory version and model are prevented."""
        vec = self.provider.embed_text("Redis caching layer")
        rec = VectorRecord.create(
            record_id="rec-r1",
            memory_id="mem-redis",
            project_id="proj-1",
            version=1,
            model_id=self.provider.model_id,
            embedding=vec,
            content="Redis caching layer",
        )
        self.store.add_vector(rec)

        # Attempting to add with fail_if_exists=True raises DuplicateVectorError
        dup = VectorRecord.create(
            record_id="rec-r2",
            memory_id="mem-redis",
            project_id="proj-1",
            version=1,
            model_id=self.provider.model_id,
            embedding=vec,
            content="Redis caching layer",
        )
        with self.assertRaises(DuplicateVectorError):
            self.store.add_vector(dup, fail_if_exists=True)

        # Without fail_if_exists returns existing record
        existing = self.store.add_vector(dup, fail_if_exists=False)
        self.assertEqual(existing.record_id, "rec-r1")
        self.assertEqual(self.store.count("proj-1"), 1)

    def test_persistence_across_reloads(self) -> None:
        """Verify vector records persist on disk after reopening the store."""
        vec = self.provider.embed_text("Persisted vector")
        rec = VectorRecord.create(
            record_id="rec-persist",
            memory_id="mem-persist",
            project_id="proj-p",
            version=1,
            model_id=self.provider.model_id,
            embedding=vec,
            content="Persisted vector",
        )
        self.store.add_vector(rec)
        self.store.close()

        # Reopen from disk
        reopened_store = SQLiteVectorStore(database_path=self.db_path)
        fetched = reopened_store.get_vector("rec-persist")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.content, "Persisted vector")
        reopened_store.close()

    # =========================================================================
    # Section 3: Semantic Similarity & Search
    # =========================================================================

    def test_cosine_similarity_calculation(self) -> None:
        """Verify cosine similarity calculation properties and edge cases."""
        # Identical vectors: similarity = 1.0
        v1 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_cosine_similarity(v1, v1), 1.0, places=5)

        # Orthogonal vectors: similarity = 0.0
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(compute_cosine_similarity(v1, v2), 0.0, places=5)

        # Opposite vectors: similarity = -1.0
        v3 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_cosine_similarity(v1, v3), -1.0, places=5)

        # Zero vector handling (returns 0.0 without division by zero error)
        v_zero = [0.0, 0.0, 0.0]
        self.assertEqual(compute_cosine_similarity(v1, v_zero), 0.0)

        # Dimension mismatch
        with self.assertRaises(DimensionMismatchError):
            compute_cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_semantic_search_retrieval(self) -> None:
        """Verify searching retrieves semantically relevant memories."""
        mems = [
            Memory.create(
                memory_id="mem-pg",
                memory_type=MemoryType.FACT,
                content="The application uses PostgreSQL relational database.",
                project_id="proj-search",
            ),
            Memory.create(
                memory_id="mem-auth",
                memory_type=MemoryType.FACT,
                content="Authentication is handled via JWT tokens.",
                project_id="proj-search",
            ),
            Memory.create(
                memory_id="mem-pref",
                memory_type=MemoryType.PREFERENCE,
                content="The developer prefers concise, functional code.",
                project_id="proj-search",
            ),
        ]
        for m in mems:
            self.manager.index_memory(m)

        # Search for database topic
        res = self.manager.search("What SQL database is used?", project_id="proj-search", top_k=2)
        self.assertGreaterEqual(len(res.matches), 1)
        self.assertEqual(res.top_match.memory_id, "mem-pg")

        # Search for auth topic
        res_auth = self.manager.search("How do users log in?", project_id="proj-search", top_k=2)
        self.assertGreaterEqual(len(res_auth.matches), 1)
        self.assertEqual(res_auth.top_match.memory_id, "mem-auth")

    def test_top_k_and_similarity_threshold(self) -> None:
        """Verify top_k limiting and similarity threshold cutoff."""
        for i in range(10):
            m = Memory.create(
                memory_id=f"mem-scale-{i}",
                memory_type=MemoryType.FACT,
                content=f"Database configuration parameter number {i}.",
                project_id="proj-cutoff",
            )
            self.manager.index_memory(m)

        # top_k = 3
        res3 = self.manager.search("database parameter", project_id="proj-cutoff", top_k=3)
        self.assertEqual(len(res3.matches), 3)

        # High threshold filters out weaker matches
        res_thresh = self.manager.search(
            "database parameter", project_id="proj-cutoff", top_k=10, similarity_threshold=0.999
        )
        for match in res_thresh.matches:
            self.assertGreaterEqual(match.score, 0.999)

    def test_project_isolation_in_search(self) -> None:
        """Verify semantic search strictly isolates memories by project_id."""
        m_proj1 = Memory.create(
            memory_id="mem-p1",
            memory_type=MemoryType.FACT,
            content="Sensitive data for project Alpha.",
            project_id="proj-alpha",
        )
        m_proj2 = Memory.create(
            memory_id="mem-p2",
            memory_type=MemoryType.FACT,
            content="Sensitive data for project Beta.",
            project_id="proj-beta",
        )
        self.manager.index_memory(m_proj1)
        self.manager.index_memory(m_proj2)

        # Search in proj-alpha MUST NEVER return proj-beta
        res_alpha = self.manager.search("Sensitive data", project_id="proj-alpha")
        self.assertEqual(len(res_alpha.matches), 1)
        self.assertEqual(res_alpha.matches[0].memory_id, "mem-p1")
        self.assertEqual(res_alpha.matches[0].project_id, "proj-alpha")

        # Search in proj-beta MUST NEVER return proj-alpha
        res_beta = self.manager.search("Sensitive data", project_id="proj-beta")
        self.assertEqual(len(res_beta.matches), 1)
        self.assertEqual(res_beta.matches[0].memory_id, "mem-p2")
        self.assertEqual(res_beta.matches[0].project_id, "proj-beta")

    def test_deterministic_tie_breaking(self) -> None:
        """Verify deterministic ordering when similarity scores are equal."""
        # Index memories with identical text in the same project
        m1 = Memory.create(memory_id="mem-b", memory_type=MemoryType.FACT, content="Identical claim text", project_id="p-tie")
        m2 = Memory.create(memory_id="mem-a", memory_type=MemoryType.FACT, content="Identical claim text", project_id="p-tie")
        self.manager.index_memory(m1)
        self.manager.index_memory(m2)

        res = self.manager.search("Identical claim text", project_id="p-tie")
        self.assertEqual(len(res.matches), 2)
        # mem-a comes before mem-b alphabetically on tie
        self.assertEqual(res.matches[0].memory_id, "mem-a")
        self.assertEqual(res.matches[1].memory_id, "mem-b")

    # =========================================================================
    # Section 4: Memory Lifecycle & Versioning Integration
    # =========================================================================

    def test_superseded_memories_excluded_by_default(self) -> None:
        """Verify superseded versions are excluded from default search and parent is superseded."""
        m_v1 = Memory.create(
            memory_id="mem-db-1",
            memory_type=MemoryType.FACT,
            content="The project uses MySQL for persistence.",
            project_id="proj-life",
            version=1,
            root_memory_id="root-db",
        )
        self.manager.index_memory(m_v1)

        # Default search finds v1
        res1 = self.manager.search("database", project_id="proj-life")
        self.assertEqual(len(res1.matches), 1)
        self.assertEqual(res1.matches[0].memory_id, "mem-db-1")

        # New version v2 created
        m_v2 = Memory.create(
            memory_id="mem-db-2",
            memory_type=MemoryType.FACT,
            content="The project uses PostgreSQL for persistence.",
            project_id="proj-life",
            version=2,
            root_memory_id="root-db",
            parent_memory_id="mem-db-1",
        )
        self.manager.index_memory(m_v2)

        # Default search now returns v2 and excludes superseded v1
        res2 = self.manager.search("database", project_id="proj-life")
        self.assertEqual(len(res2.matches), 1)
        self.assertEqual(res2.matches[0].memory_id, "mem-db-2")

        # Historical vector of v1 is preserved in store with status SUPERSEDED
        v1_record = self.store.get_vector_by_memory("proj-life", "mem-db-1", 1, self.provider.model_id)
        self.assertIsNotNone(v1_record)
        self.assertEqual(v1_record.status, "SUPERSEDED")

        # Searching with explicit SUPERSEDED status allows historical inspection
        res_hist = self.manager.search("database", project_id="proj-life", allowed_statuses=["SUPERSEDED"])
        self.assertEqual(len(res_hist.matches), 1)
        self.assertEqual(res_hist.matches[0].memory_id, "mem-db-1")

    def test_archived_memory_filtering_and_lifecycle(self) -> None:
        """Verify archived memories are excluded by default and archive_memory updates status."""
        mem = Memory.create(
            memory_id="mem-arch-1",
            memory_type=MemoryType.TASK,
            content="Migrate auth subsystem.",
            project_id="proj-arch",
        )
        self.manager.index_memory(mem)

        # Active search finds it
        res = self.manager.search("auth", project_id="proj-arch")
        self.assertEqual(len(res.matches), 1)

        # Archive memory
        self.manager.archive_memory("mem-arch-1", "proj-arch")

        # Active search excludes it
        res_after = self.manager.search("auth", project_id="proj-arch")
        self.assertEqual(len(res_after.matches), 0)

        # Allowed statuses including ARCHIVED can find it
        res_archived = self.manager.search("auth", project_id="proj-arch", allowed_statuses=["ARCHIVED"])
        self.assertEqual(len(res_archived.matches), 1)

    def test_disputed_memory_behavior(self) -> None:
        """Verify disputed memories are indexed with DISPUTED status and handled explicitly."""
        mem_disp = Memory.create(
            memory_id="mem-disp-1",
            memory_type=MemoryType.FACT,
            content="Disputed architecture choice.",
            project_id="proj-disp",
            status=MemoryStatus.DISPUTED.value,
        )
        self.manager.index_memory(mem_disp)

        # Default active search excludes DISPUTED memories
        res_default = self.manager.search("architecture", project_id="proj-disp")
        self.assertEqual(len(res_default.matches), 0)

        # include_disputed=True surfaces it with explicit DISPUTED status
        res_disp = self.manager.search("architecture", project_id="proj-disp", include_disputed=True)
        self.assertEqual(len(res_disp.matches), 1)
        self.assertEqual(res_disp.matches[0].status, "DISPUTED")
        self.assertTrue(res_disp.matches[0].metadata.get("is_disputed"))

    def test_original_memory_immutability_preserved(self) -> None:
        """Verify VectorMemoryManager operations never mutate original Memory objects."""
        mem = Memory.create(
            memory_id="mem-immut",
            memory_type=MemoryType.PREFERENCE,
            content="Original content.",
            project_id="proj-immut",
        )
        self.manager.index_memory(mem)
        with self.assertRaises(AttributeError):
            mem.content = "Modified"  # type: ignore


if __name__ == "__main__":
    unittest.main()

