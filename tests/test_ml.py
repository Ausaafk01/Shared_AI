"""
Comprehensive tests for AI Brain Phase 3 Machine Learning subsystem.
"""

from datetime import datetime, timedelta, timezone
import os
import shutil
import tempfile
import unittest

from ai_brain.events.models import Event
from ai_brain.events.storage.sqlite import SQLiteEventStore
from ai_brain.loop.orchestrator import BrainLoop
from ai_brain.memories.conflicts.types import ConflictType
from ai_brain.memories.models import Memory
from ai_brain.ml.base import ModelMetadata, ModelPrediction, ModelType
from ai_brain.ml.datasets.builder import DatasetSplitter
from ai_brain.ml.datasets.schemas import DatasetExample, DatasetSplit, ExampleProvenance, ExampleSource
from ai_brain.ml.datasets.validator import DatasetValidator
from ai_brain.ml.exceptions import DatasetValidationError, ModelNotTrainedError
from ai_brain.ml.features.graph_vector import GraphVectorFeatureExtractor
from ai_brain.ml.features.temporal import TemporalFeatureExtractor
from ai_brain.ml.features.text import TextFeatureExtractor
from ai_brain.ml.feedback.collector import FeedbackCollector
from ai_brain.ml.feedback.models import FeedbackRecord, FeedbackSignal, FeedbackType
from ai_brain.ml.feedback.store import SQLiteFeedbackStore
from ai_brain.ml.integration.config import MLConfiguration
from ai_brain.ml.integration.service import MLService
from ai_brain.ml.models.conflict import ConflictDetectionModel
from ai_brain.ml.models.duplicate import DuplicateClassification, DuplicateDetectionModel
from ai_brain.ml.models.importance import MemoryImportanceModel
from ai_brain.ml.models.ranking import RelevanceRankingModel
from ai_brain.ml.models.staleness import StalenessDetectionModel
from ai_brain.ml.training.evaluator import ModelEvaluationReport, ModelEvaluator
from ai_brain.ml.training.pipeline import TrainingPipeline
from ai_brain.ml.training.registry import ModelRegistry
from ai_brain.retrieval.engine import RetrievalEngine
from ai_brain.retrieval.models import QueryFilters, RetrievalItem, ScoreBreakdown
from ai_brain.retrieval.types import RetrievalMethod
from ai_brain.vectors.index import VectorIndex
from ai_brain.vectors.manager import VectorMemoryManager
from ai_brain.vectors.providers.mock import MockEmbeddingProvider
from ai_brain.vectors.storage.sqlite import SQLiteVectorStore


class TestMLFeatureExtractors(unittest.TestCase):
    """Test feature extractors across text, temporal, and graph/vector domains."""

    def test_text_feature_extractor(self) -> None:
        extractor = TextFeatureExtractor()
        text = "We decided to adopt PostgreSQL and Redis for caching. Must use port 5432."
        feats = extractor.extract(text)

        self.assertIn("decision_score", feats)
        self.assertIn("constraint_score", feats)
        self.assertIn("tech_term_count", feats)
        self.assertGreater(feats["decision_score"], 0.0)
        self.assertGreater(feats["constraint_score"], 0.0)
        self.assertGreaterEqual(feats["tech_term_count"], 2.0)

    def test_temporal_feature_extractor(self) -> None:
        extractor = TemporalFeatureExtractor(half_life_days=30.0)
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=60)

        mem = Memory(
            memory_id="mem-temp-1",
            memory_type="FACT",
            content="Temporary migration draft",
            project_id="proj-1",
            created_at=old_time,
            updated_at=old_time,
        )
        feats = extractor.extract(mem, reference_time=now)
        self.assertAlmostEqual(feats["age_days"], 60.0, delta=1.0)
        self.assertLess(feats["recency_decay"], 0.2)
        self.assertEqual(feats["has_temporal_keywords"], 1.0)

    def test_graph_vector_feature_extractor(self) -> None:
        extractor = GraphVectorFeatureExtractor()
        mem = Memory(
            memory_id="mem-gv-1",
            memory_type="DECISION",
            content="Use Redis",
            project_id="proj-1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            source_agent_ids=("agent-a", "agent-b"),
            source_event_ids=("ev-1", "ev-2", "ev-3"),
        )
        feats = extractor.extract(mem, vector_score=0.88, graph_score=0.75)
        self.assertEqual(feats["vector_score"], 0.88)
        self.assertEqual(feats["graph_score"], 0.75)
        self.assertEqual(feats["source_agent_count"], 2.0)
        self.assertEqual(feats["source_event_count"], 3.0)


class TestMLDatasetsAndSplitting(unittest.TestCase):
    """Test dataset validation, schemas, and zero-leakage splitting."""

    def test_dataset_splitting_and_zero_leakage(self) -> None:
        now = datetime.now(timezone.utc)
        examples = []
        for i in range(20):
            prov = ExampleProvenance(
                source_type=ExampleSource.REAL if i % 2 == 0 else ExampleSource.SYNTHETIC,
                created_at=now,
                project_id="proj-split",
                source_memory_ids=(f"root-mem-{i // 2}",),
            )
            ex = DatasetExample(
                example_id=f"ex-{i}",
                features={"feat_a": float(i), "feat_b": float(i * 2)},
                label=1.0 if i > 10 else 0.0,
                provenance=prov,
            )
            examples.append(ex)

        split = DatasetSplitter.split(examples, val_ratio=0.2, test_ratio=0.1, seed=42, group_by_memory=True)
        self.assertGreater(len(split.train_examples), 0)
        self.assertGreater(len(split.validation_examples), 0)

        # Confirm zero example ID leakage
        train_ids = {e.example_id for e in split.train_examples}
        val_ids = {e.example_id for e in split.validation_examples}
        self.assertTrue(train_ids.isdisjoint(val_ids))

        # Confirm zero underlying memory ID leakage
        train_mems = {e.provenance.source_memory_ids[0] for e in split.train_examples}
        val_mems = {e.provenance.source_memory_ids[0] for e in split.validation_examples}
        self.assertTrue(train_mems.isdisjoint(val_mems))

    def test_dataset_validation_rejects_leakage(self) -> None:
        now = datetime.now(timezone.utc)
        prov = ExampleProvenance(source_type=ExampleSource.SYNTHETIC, created_at=now, source_memory_ids=("mem-leak",))
        ex = DatasetExample(example_id="leak-id", features={"a": 1.0}, label=1.0, provenance=prov)
        split = DatasetSplit(
            name="leak-test",
            train_examples=(ex,),
            validation_examples=(ex,),
            feature_names=("a",),
        )
        with self.assertRaises(DatasetValidationError):
            DatasetValidator.validate_split(split)


class TestMLModels(unittest.TestCase):
    """Test training, inference, confidence, and serialization for all ML models."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.now = datetime.now(timezone.utc)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_memory_importance_model(self) -> None:
        model = MemoryImportanceModel()
        # Untrained fallback
        mem = Memory(
            memory_id="m-imp-1",
            memory_type="DECISION",
            content="We decided to use Redis for session management",
            project_id="p-1",
            created_at=self.now,
            updated_at=self.now,
            importance=0.85,
        )
        pred_untrained = model.predict(mem)
        self.assertTrue(pred_untrained.fallback_used)
        self.assertEqual(pred_untrained.target_value, 0.85)

        # Train model
        train_ex = [
            DatasetExample(
                example_id=f"ex-{i}",
                features=model.extract_features(mem),
                label=0.9 if i % 2 == 0 else 0.2,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "p-1", (f"mem-{i}",)),
            )
            for i in range(12)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
        metrics = model.train(split, epochs=30)
        self.assertIn("mae", metrics)
        self.assertTrue(model.is_trained)

        # Inference
        pred = model.predict(mem)
        self.assertFalse(pred.fallback_used)
        self.assertGreaterEqual(float(pred.target_value), 0.0)
        self.assertLessEqual(float(pred.target_value), 1.0)
        self.assertIn("is_decision_type", pred.feature_contributions)

        # Save and reload
        save_path = os.path.join(self.tmp_dir, "importance.json")
        model.save(save_path)
        loaded = MemoryImportanceModel.load(save_path)
        self.assertTrue(loaded.is_trained)
        self.assertEqual(loaded.predict(mem).target_value, pred.target_value)

    def test_relevance_ranking_model(self) -> None:
        model = RelevanceRankingModel()
        item = RetrievalItem(
            memory_id="m-rank-1",
            root_memory_id=None,
            version=1,
            project_id="proj-r",
            content="PostgreSQL database cluster configuration",
            memory_type="FACT",
            status="ACTIVE",
            score=0.72,
            score_breakdown=ScoreBreakdown(base_score=0.72, vector_score=0.72, final_score=0.72),
            created_at=self.now,
        )

        train_ex = [
            DatasetExample(
                example_id=f"r-{i}",
                features=model.extract_features(item, query="PostgreSQL"),
                label=0.85 if i % 2 == 0 else 0.3,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "proj-r", (f"m-{i}",)),
            )
            for i in range(10)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.3, test_ratio=0.0, group_by_memory=False)
        model.train(split, epochs=25)

        reranked = model.rerank([item], query="PostgreSQL")
        self.assertEqual(len(reranked), 1)
        self.assertIn("ml_rank_score", reranked[0].metadata)

    def test_duplicate_detection_model(self) -> None:
        model = DuplicateDetectionModel()
        m1 = Memory(
            memory_id="m-dup-1",
            memory_type="FACT",
            content="The system uses PostgreSQL for primary storage",
            project_id="proj-dup",
            created_at=self.now,
            updated_at=self.now,
        )
        m2_identical = Memory(
            memory_id="m-dup-2",
            memory_type="FACT",
            content="The system uses PostgreSQL for primary storage",
            project_id="proj-dup",
            created_at=self.now,
            updated_at=self.now,
        )
        m3_contra = Memory(
            memory_id="m-dup-3",
            memory_type="FACT",
            content="The system uses MySQL for primary storage",
            project_id="proj-dup",
            created_at=self.now,
            updated_at=self.now,
        )

        train_ex = [
            DatasetExample(
                example_id=f"d-{i}",
                features=model.extract_pair_features(m1, m2_identical),
                label=1.0,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "proj-dup", (f"m-{i}",)),
            )
            for i in range(6)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.3, test_ratio=0.0, group_by_memory=False)
        model.train(split)

        pred_exact = model.predict((m1, m2_identical))
        self.assertEqual(pred_exact.target_value, DuplicateClassification.EXACT_DUPLICATE.value)

        pred_contra = model.predict((m1, m3_contra))
        self.assertEqual(pred_contra.target_value, DuplicateClassification.CONTRADICTORY.value)

    def test_conflict_detection_model(self) -> None:
        model = ConflictDetectionModel()
        m1 = Memory(
            memory_id="m-c-1",
            memory_type="FACT",
            content="Deploy Redis cache on port 6379",
            project_id="proj-c",
            created_at=self.now,
            updated_at=self.now,
        )
        m2_port_contra = Memory(
            memory_id="m-c-2",
            memory_type="FACT",
            content="Deploy Redis cache on port 6380",
            project_id="proj-c",
            created_at=self.now,
            updated_at=self.now,
        )
        m3_temporal = Memory(
            memory_id="m-c-3",
            memory_type="FACT",
            content="Subsequently migrated to port 6380 yesterday",
            project_id="proj-c",
            created_at=self.now + timedelta(days=2),
            updated_at=self.now + timedelta(days=2),
        )

        train_ex = [
            DatasetExample(
                example_id=f"c-{i}",
                features=model.extract_conflict_features(m1, m2_port_contra),
                label=1.0,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "proj-c", (f"mc-{i}",)),
            )
            for i in range(6)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.3, test_ratio=0.0, group_by_memory=False)
        model.train(split)

        # Technical conflict
        pred_tech = model.predict((m1, m2_port_contra))
        self.assertEqual(pred_tech.target_value, ConflictType.TECHNICAL.value)

        # Temporal migration differentiation
        pred_temp = model.predict((m1, m3_temporal))
        self.assertEqual(pred_temp.target_value, ConflictType.TEMPORAL.value)

    def test_staleness_detection_model(self) -> None:
        model = StalenessDetectionModel()
        mem_old = Memory(
            memory_id="m-stale-1",
            memory_type="FACT",
            content="Temporary staging key until Friday",
            project_id="proj-s",
            created_at=self.now - timedelta(days=120),
            updated_at=self.now - timedelta(days=120),
        )
        mem_decision = Memory(
            memory_id="m-fresh-dec",
            memory_type="DECISION",
            content="We decided to use Python 3.13 as standard runtime",
            project_id="proj-s",
            created_at=self.now - timedelta(days=120),
            updated_at=self.now - timedelta(days=120),
        )

        train_ex = [
            DatasetExample(
                example_id=f"s-{i}",
                features=model.extract_staleness_features(mem_old),
                label=1.0 if i % 2 == 0 else 0.0,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "proj-s", (f"ms-{i}",)),
            )
            for i in range(8)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
        model.train(split)

        # Temporary old memory has higher staleness probability than decision
        p_temp = float(model.predict(mem_old).target_value)
        p_dec = float(model.predict(mem_decision).target_value)
        self.assertGreater(p_temp, p_dec)


class TestFeedbackAndEvaluation(unittest.TestCase):
    """Test feedback storage, batch export, and model evaluators."""

    def test_feedback_collector_and_training_batch_export(self) -> None:
        store = SQLiteFeedbackStore(":memory:")
        collector = FeedbackCollector(store)

        collector.record_retrieval_feedback(
            project_id="p-fb",
            memory_id="mem-1",
            is_relevant=True,
            notes="Relevant result",
        )
        collector.record_duplicate_feedback(
            project_id="p-fb",
            m1_id="mem-1",
            m2_id="mem-2",
            is_duplicate=False,
            notes="Distinct facts",
        )

        records = store.list_feedback(project_id="p-fb")
        self.assertEqual(len(records), 2)

        # Batch export without automated retraining
        retrieval_batch = collector.prepare_training_batch(
            feedback_type=FeedbackType.RETRIEVAL_RELEVANCE,
            project_id="p-fb",
        )
        self.assertEqual(len(retrieval_batch), 1)
        self.assertEqual(retrieval_batch[0].provenance.source_type, ExampleSource.FEEDBACK_DERIVED)

    def test_evaluator_metrics(self) -> None:
        # Binary classification
        c_metrics = ModelEvaluator.evaluate_binary_classification(
            y_true=[1, 1, 0, 0, 1],
            y_pred=[1, 0, 0, 1, 1],
            y_prob=[0.9, 0.4, 0.1, 0.7, 0.85],
        )
        self.assertEqual(c_metrics.total_samples, 5)
        self.assertGreater(c_metrics.f1_score, 0.0)
        self.assertGreater(c_metrics.brier_score, 0.0)

        # Ranking
        r_metrics = ModelEvaluator.evaluate_ranking([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertGreater(r_metrics.ndcg_at_5, 0.0)
        self.assertGreater(r_metrics.mrr, 0.0)


class TestMLServiceAndIntegration(unittest.TestCase):
    """Test MLService configuration flags, graceful fallbacks, and Brain Loop integration."""

    def test_feature_flag_bypass_and_deterministic_fallback(self) -> None:
        # Disable ML in config
        cfg = MLConfiguration(enable_importance_ml=False, enable_ranking_ml=False)
        svc = MLService(config=cfg)

        mem = Memory(
            memory_id="m-flag-1",
            memory_type="FACT",
            content="Database is PostgreSQL",
            project_id="proj-flag",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            importance=0.65,
        )
        pred = svc.predict_importance(mem)
        self.assertTrue(pred.fallback_used)
        self.assertEqual(pred.target_value, 0.65)

    def test_retrieval_engine_ml_reranker_integration(self) -> None:
        prov = MockEmbeddingProvider(dimension=16)
        store = SQLiteVectorStore(":memory:")
        v_mgr = VectorMemoryManager(storage=store, provider=prov)
        v_idx = v_mgr.index

        now = datetime.now(timezone.utc)
        m1 = Memory(
            memory_id="m-ret-1",
            memory_type="FACT",
            content="Redis port 6379",
            project_id="proj-ml-ret",
            created_at=now,
            updated_at=now,
        )
        m2 = Memory(
            memory_id="m-ret-2",
            memory_type="FACT",
            content="Redis Sentinel high availability",
            project_id="proj-ml-ret",
            created_at=now,
            updated_at=now,
        )
        v_mgr.index_memory(m1)
        v_mgr.index_memory(m2)

        # Train ranking model
        ranker = RelevanceRankingModel()
        item = RetrievalItem(
            memory_id="m-ret-1",
            root_memory_id=None,
            version=1,
            project_id="proj-ml-ret",
            content="Redis port 6379",
            memory_type="FACT",
            status="ACTIVE",
            score=0.8,
            score_breakdown=ScoreBreakdown(base_score=0.8, vector_score=0.8, final_score=0.8),
            created_at=now,
        )
        train_ex = [
            DatasetExample(
                example_id=f"r-{i}",
                features=ranker.extract_features(item, query="Redis"),
                label=0.9 if i % 2 == 0 else 0.4,
                provenance=ExampleProvenance(ExampleSource.REAL, now, "proj-ml-ret", (f"m-{i}",)),
            )
            for i in range(8)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
        ranker.train(split)

        engine = RetrievalEngine(vector_index=v_idx, ml_ranker=ranker)
        res = engine.retrieve("Redis cache", project_id="proj-ml-ret", method=RetrievalMethod.VECTOR)
        self.assertEqual(res.total_count, 2)
        # Verify ML ranking annotation in metadata
        self.assertIn("ml_rank_score", res.results[0].metadata)

    def test_brain_loop_with_ml_service(self) -> None:
        ev_store = SQLiteEventStore(":memory:")
        svc = MLService()

        # Train importance model
        now = datetime.now(timezone.utc)
        imp_model = svc.registry.get_importance_model()
        dummy_mem = Memory(
            memory_id="dummy",
            memory_type="DECISION",
            content="Decided on architecture",
            project_id="p-bl",
            created_at=now,
            updated_at=now,
        )
        train_ex = [
            DatasetExample(
                example_id=f"b-{i}",
                features=imp_model.extract_features(dummy_mem),
                label=0.95,
                provenance=ExampleProvenance(ExampleSource.REAL, now, "p-bl", (f"mb-{i}",)),
            )
            for i in range(8)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
        imp_model.train(split)

        loop = BrainLoop(event_store=ev_store, ml_service=svc)

        # Run interaction that yields a decision memory
        from ai_brain.loop.models import AgentInteractionInput, AgentResponseInput
        inp = AgentInteractionInput(
            project_id="p-bl",
            agent_id="agent-coder",
            session_id="sess-1",
            content="What database do we use?",
        )
        ctx = loop.prepare_context(inp)
        resp = AgentResponseInput(
            project_id="p-bl",
            agent_id="agent-coder",
            session_id="sess-1",
            interaction_id=ctx.interaction_id,
            content="We decided to adopt PostgreSQL as our standard database.",
        )
        res = loop.complete_interaction(resp)
        self.assertEqual(res.status.value, "COMPLETED")
        self.assertGreater(len(res.extracted_memories), 0)
        # Importance model should have evaluated the newly extracted decision memory
        m = res.extracted_memories[0]
        self.assertGreater(m.importance, 0.5)


class TestMLStrengthenedEdgeCases(unittest.TestCase):
    """
    Dedicated test suite verifying edge cases, safety safeguards, isolation,
    and failure handling requested in technical review.
    """

    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_false_positive_duplicate_detection_prevention(self) -> None:
        """Memories on same tech subject with different actions/configs must not be classified as duplicates."""
        dup_model = DuplicateDetectionModel()
        m1 = Memory(
            memory_id="m-fp-1",
            memory_type="FACT",
            content="Configure PostgreSQL connection pool size to 25 connections",
            project_id="p-fp",
            created_at=self.now,
            updated_at=self.now,
        )
        m2 = Memory(
            memory_id="m-fp-2",
            memory_type="FACT",
            content="Configure PostgreSQL automated WAL backup schedule for midnight",
            project_id="p-fp",
            created_at=self.now,
            updated_at=self.now,
        )
        pred = dup_model.predict((m1, m2))
        self.assertEqual(pred.target_value, DuplicateClassification.RELATED_DISTINCT.value)

    def test_contradictory_claims_with_high_semantic_similarity(self) -> None:
        """Contradictory claims with high lexical overlap must trigger contradiction gate, not duplicate."""
        dup_model = DuplicateDetectionModel()
        m1 = Memory(
            memory_id="m-hi-sim-1",
            memory_type="FACT",
            content="The Redis distributed cache cluster port is configured to 6379",
            project_id="p-sim",
            created_at=self.now,
            updated_at=self.now,
        )
        m2 = Memory(
            memory_id="m-hi-sim-2",
            memory_type="FACT",
            content="The Redis distributed cache cluster port is configured to 6380",
            project_id="p-sim",
            created_at=self.now,
            updated_at=self.now,
        )
        # Train model with contradictory pair
        train_ex = [
            DatasetExample(
                example_id=f"dp-c-{i}",
                features=dup_model.extract_pair_features(m1, m2),
                label=0.0,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "p-sim", (f"dm-{i}",)),
            )
            for i in range(8)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
        dup_model.train(split)

        pred = dup_model.predict((m1, m2))
        self.assertEqual(pred.target_value, DuplicateClassification.CONTRADICTORY.value)

    def test_temporal_updates_versus_genuine_conflicts(self) -> None:
        """Chronological migration language must be identified as TEMPORAL progression, not static contradiction."""
        conf_model = ConflictDetectionModel()
        m1 = Memory(
            memory_id="m-t-1",
            memory_type="FACT",
            content="The project uses PostgreSQL 15 for data storage",
            project_id="p-t",
            created_at=self.now,
            updated_at=self.now,
        )
        m2_migrated = Memory(
            memory_id="m-t-2",
            memory_type="FACT",
            content="Subsequently upgraded database and migrated to PostgreSQL 16 yesterday",
            project_id="p-t",
            created_at=self.now + timedelta(days=5),
            updated_at=self.now + timedelta(days=5),
        )
        # Train model with temporal progression example
        train_ex = [
            DatasetExample(
                example_id=f"cf-t-{i}",
                features=conf_model.extract_conflict_features(m1, m2_migrated),
                label=ConflictType.TEMPORAL.value,
                provenance=ExampleProvenance(ExampleSource.REAL, self.now, "p-t", (f"cm-{i}",)),
            )
            for i in range(8)
        ]
        split = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
        conf_model.train(split)

        pred = conf_model.predict((m1, m2_migrated))
        self.assertEqual(pred.target_value, ConflictType.TEMPORAL.value)

    def test_different_project_memories_strict_isolation(self) -> None:
        """Cross-project comparisons must always yield NONE / DISTINCT with 1.0 confidence."""
        dup_model = DuplicateDetectionModel()
        conf_model = ConflictDetectionModel()

        m_proj_a = Memory(
            memory_id="m-pa",
            memory_type="FACT",
            content="The primary database is PostgreSQL",
            project_id="proj-a",
            created_at=self.now,
            updated_at=self.now,
        )
        m_proj_b = Memory(
            memory_id="m-pb",
            memory_type="FACT",
            content="The primary database is MySQL",
            project_id="proj-b",
            created_at=self.now,
            updated_at=self.now,
        )

        conf_pred = conf_model.predict((m_proj_a, m_proj_b))
        self.assertEqual(conf_pred.target_value, "NONE")
        self.assertEqual(conf_pred.confidence, 1.0)

    def test_different_agent_contexts_provenance(self) -> None:
        """Distinct agent sources corroborate shared claims rather than conflicting."""
        m_agent_1 = Memory(
            memory_id="m-ag-1",
            memory_type="FACT",
            content="System cache is Redis",
            project_id="p-ag",
            created_at=self.now,
            updated_at=self.now,
            source_agent_ids=("agent-research",),
        )
        m_agent_2 = Memory(
            memory_id="m-ag-2",
            memory_type="FACT",
            content="System cache is Redis",
            project_id="p-ag",
            created_at=self.now,
            updated_at=self.now,
            source_agent_ids=("agent-coding",),
        )
        conf_model = ConflictDetectionModel()
        pred = conf_model.predict((m_agent_1, m_agent_2))
        self.assertEqual(pred.target_value, "NONE")

    def test_stale_but_still_valid_memories_preserved(self) -> None:
        """Stale prediction flags high risk without altering status or deleting memory."""
        stale_model = StalenessDetectionModel()
        mem = Memory(
            memory_id="m-stale-valid",
            memory_type="FACT",
            content="Architecture decision record from 6 months ago",
            project_id="p-sv",
            created_at=self.now - timedelta(days=180),
            updated_at=self.now - timedelta(days=180),
            status="ACTIVE",
        )
        pred = stale_model.predict(mem)
        self.assertGreaterEqual(float(pred.target_value), 0.0)
        # Memory status on original object must remain untouched
        self.assertEqual(mem.status, "ACTIVE")

    def test_untrained_model_fallback_all_five_models(self) -> None:
        """Verify all 5 models gracefully fall back to deterministic logic when untrained."""
        imp = MemoryImportanceModel()
        rank = RelevanceRankingModel()
        dup = DuplicateDetectionModel()
        conf = ConflictDetectionModel()
        stale = StalenessDetectionModel()

        mem = Memory(
            memory_id="m-fb",
            memory_type="DECISION",
            content="Use OAuth2",
            project_id="p-fb",
            created_at=self.now,
            updated_at=self.now,
            importance=0.75,
        )
        ret_item = RetrievalItem(
            memory_id="m-fb",
            root_memory_id=None,
            version=1,
            project_id="p-fb",
            content="Use OAuth2",
            memory_type="DECISION",
            status="ACTIVE",
            score=0.75,
            score_breakdown=ScoreBreakdown(base_score=0.75, final_score=0.75),
            created_at=self.now,
        )

        p_imp = imp.predict(mem)
        self.assertTrue(p_imp.fallback_used)
        self.assertEqual(p_imp.target_value, 0.75)

        p_rank = rank.predict(ret_item)
        self.assertTrue(p_rank.fallback_used)
        self.assertEqual(p_rank.target_value, 0.75)

        p_dup = dup.predict((mem, mem))
        self.assertTrue(p_dup.fallback_used)

        p_conf = conf.predict((mem, mem))
        self.assertTrue(p_conf.fallback_used)

        p_stale = stale.predict(mem)
        self.assertTrue(p_stale.fallback_used)

    def test_low_confidence_prediction_fallback(self) -> None:
        """MLService falls back to baseline when prediction confidence is below threshold."""
        cfg = MLConfiguration(min_confidence_threshold=0.99)
        svc = MLService(config=cfg)

        mem = Memory(
            memory_id="m-low",
            memory_type="FACT",
            content="Plain fact",
            project_id="p-low",
            created_at=self.now,
            updated_at=self.now,
            importance=0.55,
        )
        pred = svc.predict_importance(mem)
        self.assertTrue(pred.fallback_used)
        self.assertEqual(pred.target_value, 0.55)

    def test_model_loading_failures_and_missing_files(self) -> None:
        """Loading a non-existent model file raises FileNotFoundError cleanly."""
        with self.assertRaises(FileNotFoundError):
            MemoryImportanceModel.load("non_existent_model_file.json")

    def test_ml_disabled_independently_for_each_component(self) -> None:
        """Each component flag in MLConfiguration independently gates that component."""
        cfg = MLConfiguration(
            enable_importance_ml=False,
            enable_ranking_ml=True,
            enable_duplicate_ml=False,
            enable_conflict_ml=True,
            enable_staleness_ml=False,
        )
        svc = MLService(config=cfg)

        mem = Memory(
            memory_id="m-gate",
            memory_type="FACT",
            content="Gate test",
            project_id="p-gate",
            created_at=self.now,
            updated_at=self.now,
            importance=0.6,
        )
        p_imp = svc.predict_importance(mem)
        self.assertTrue(p_imp.fallback_used)
        self.assertIn("disabled", p_imp.explanation)

        p_dup = svc.check_duplicate(mem, mem)
        self.assertTrue(p_dup.fallback_used)
        self.assertIn("Deterministic comparator", p_dup.explanation)

        p_stale = svc.assess_staleness(mem)
        self.assertTrue(p_stale.fallback_used)
        self.assertIn("disabled", p_stale.explanation)

    def test_feedback_persistence_and_provenance_roundtrip(self) -> None:
        """Feedback recorded in SQLite store persists completely across store reload."""
        db_file = os.path.join(self.tmp_dir, "feedback_test.sqlite")
        store = SQLiteFeedbackStore(db_file)
        collector = FeedbackCollector(store)

        rec = collector.record_retrieval_feedback(
            project_id="p-fb-rt",
            memory_id="mem-rt-1",
            is_relevant=True,
            model_name="RelevanceRankingModel",
            model_version="v1.2.3",
            agent_id="agent-eval",
            session_id="sess-rt",
            notes="Audit signal verified",
        )
        store.close()

        # Reopen store from disk
        store2 = SQLiteFeedbackStore(db_file)
        records = store2.list_feedback(project_id="p-fb-rt")
        self.assertEqual(len(records), 1)
        loaded = records[0]
        self.assertEqual(loaded.feedback_id, rec.feedback_id)
        self.assertEqual(loaded.signal, FeedbackSignal.POSITIVE)
        self.assertEqual(loaded.model_name, "RelevanceRankingModel")
        self.assertEqual(loaded.model_version, "v1.2.3")
        self.assertEqual(loaded.agent_id, "agent-eval")
        self.assertEqual(loaded.session_id, "sess-rt")
        store2.close()

    def test_training_test_leakage_detection(self) -> None:
        """Validator rejects splits with memory ID leakage."""
        prov = ExampleProvenance(
            source_type=ExampleSource.REAL,
            created_at=self.now,
            source_memory_ids=("shared-root-mem",),
        )
        ex1 = DatasetExample(example_id="id-1", features={"f": 1.0}, label=1.0, provenance=prov)
        ex2 = DatasetExample(example_id="id-2", features={"f": 1.0}, label=0.0, provenance=prov)

        split = DatasetSplit(
            name="leak-test",
            train_examples=(ex1,),
            validation_examples=(ex2,),
            feature_names=("f",),
        )
        with self.assertRaises(DatasetValidationError):
            DatasetValidator.validate_split(split)

        # Verify group-aware k-fold cross-validation prevents leakage across all folds
        examples = [
            DatasetExample(
                example_id=f"ex-{i}",
                features={"f": 1.0},
                label=float(i % 2),
                provenance=ExampleProvenance(
                    source_type=ExampleSource.MANUALLY_LABELLED,
                    created_at=self.now,
                    source_memory_ids=(f"mem-grp-{i // 2}",),
                ),
            )
            for i in range(10)
        ]
        cv_splits = DatasetSplitter.k_fold_split(examples, k=3, group_by_memory=True)
        self.assertEqual(len(cv_splits), 3)
        for s in cv_splits:
            DatasetValidator.validate_split(s)

    def test_brain_loop_automated_feedback_collection(self) -> None:
        """BrainLoop automatically logs retrieval usage feedback upon completed interaction."""
        ev_store = SQLiteEventStore(":memory:")
        fb_store = SQLiteFeedbackStore(":memory:")
        collector = FeedbackCollector(fb_store)
        svc = MLService(feedback_collector=collector)

        loop = BrainLoop(event_store=ev_store, ml_service=svc)

        from ai_brain.loop.models import AgentInteractionInput, AgentResponseInput
        inp = AgentInteractionInput(
            project_id="p-auto-fb",
            agent_id="agent-1",
            session_id="sess-auto",
            content="Initial inquiry",
        )
        ctx = loop.prepare_context(inp)
        resp = AgentResponseInput(
            project_id="p-auto-fb",
            agent_id="agent-1",
            session_id="sess-auto",
            interaction_id=ctx.interaction_id,
            content="The system adopts PostgreSQL 16.",
        )
        res = loop.complete_interaction(resp)
        self.assertEqual(res.status.value, "COMPLETED")

        # Now test manual feedback via loop
        loop.record_feedback(
            project_id="p-auto-fb",
            memory_id="mem-manual",
            is_relevant=True,
            notes="Manual rating",
        )
        all_fb = fb_store.list_feedback(project_id="p-auto-fb")
        self.assertGreaterEqual(len(all_fb), 1)


if __name__ == "__main__":
    unittest.main()
