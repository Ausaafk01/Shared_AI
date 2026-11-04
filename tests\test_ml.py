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


if __name__ == "__main__":
    unittest.main()
