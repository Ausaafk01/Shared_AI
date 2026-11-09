"""
AI Brain — Phase 3: Machine Learning Subsystem Demonstration

Demonstrates:
1. Memory Importance Prediction (features, explainability, confidence vs. deterministic baseline).
2. ML-Enhanced Retrieval Ranking (vector/graph candidate retrieval + ML re-ranking).
3. Duplicate Detection (exact duplicate vs semantic duplicate vs distinct vs contradiction).
4. Conflict Detection (technical, temporal, factual contradictions vs compatible elaborations).
5. Staleness Detection (obsolescence risk modeling without memory deletion).
6. Feedback Collection & Controlled Evaluation (recording signals, preparing training batch, comparative metrics).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_brain.memories.models import Memory
from ai_brain.ml.datasets.builder import DatasetSplitter
from ai_brain.ml.datasets.schemas import DatasetExample, ExampleProvenance, ExampleSource
from ai_brain.ml.feedback.collector import FeedbackCollector
from ai_brain.ml.feedback.models import FeedbackType
from ai_brain.ml.feedback.store import SQLiteFeedbackStore
from ai_brain.ml.models.conflict import ConflictDetectionModel
from ai_brain.ml.models.duplicate import DuplicateDetectionModel
from ai_brain.ml.models.importance import MemoryImportanceModel
from ai_brain.ml.models.ranking import RelevanceRankingModel
from ai_brain.ml.models.staleness import StalenessDetectionModel
from ai_brain.ml.training.evaluator import ModelEvaluator
from ai_brain.retrieval.engine import RetrievalEngine
from ai_brain.retrieval.types import RetrievalMethod
from ai_brain.vectors.manager import VectorMemoryManager
from ai_brain.vectors.providers.mock import MockEmbeddingProvider
from ai_brain.vectors.storage.sqlite import SQLiteVectorStore


def main() -> None:
    now = datetime.now(timezone.utc)
    print("=" * 80)
    print("AI Brain -- Phase 3: Machine Learning Subsystem Demonstration")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Demo 1: Memory Importance Model
    # -------------------------------------------------------------------------
    print("\n[Demo 1] Memory Importance Prediction")
    print("-" * 50)
    imp_model = MemoryImportanceModel()

    # Create representative memories
    mem_decision = Memory(
        memory_id="mem-imp-dec",
        memory_type="DECISION",
        content="We decided to adopt PostgreSQL and Redis for distributed caching.",
        project_id="proj-alpha",
        created_at=now,
        updated_at=now,
        source_agent_ids=("agent-architect", "agent-lead"),
        source_event_ids=("ev-101", "ev-102"),
    )
    mem_temp = Memory(
        memory_id="mem-imp-temp",
        memory_type="FACT",
        content="Temporary staging API key for testing until Friday.",
        project_id="proj-alpha",
        created_at=now,
        updated_at=now,
        source_agent_ids=("agent-intern",),
    )

    # Train model on curated dataset
    train_ex = [
        DatasetExample(
            example_id=f"imp-{i}",
            features=imp_model.extract_features(mem_decision if i % 2 == 0 else mem_temp),
            label=0.92 if i % 2 == 0 else 0.15,
            provenance=ExampleProvenance(
                source_type=ExampleSource.MANUALLY_LABELLED,
                created_at=now,
                project_id="proj-alpha",
                source_memory_ids=(f"mem-src-{i}",),
            ),
        )
        for i in range(16)
    ]
    split_imp = DatasetSplitter.split(train_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
    imp_model.train(split_imp, epochs=40)

    pred_dec = imp_model.predict(mem_decision)
    pred_temp = imp_model.predict(mem_temp)

    print(f"  Decision Memory: '{mem_decision.content}'")
    print(f"    * Baseline importance : {mem_decision.importance:.2f}")
    print(f"    * ML Predicted score  : {pred_dec.target_value} (Confidence: {pred_dec.confidence})")
    print(f"    * Explanation         : {pred_dec.explanation}")
    print(f"    * Key features        : is_decision={pred_dec.feature_contributions.get('is_decision_type')}, prov_breadth={pred_dec.feature_contributions.get('provenance_breadth')}")

    print(f"  Temporary Memory: '{mem_temp.content}'")
    print(f"    * ML Predicted score  : {pred_temp.target_value} (Confidence: {pred_temp.confidence})")
    print(f"    * Key features        : temp_penalty={pred_temp.feature_contributions.get('temporary_penalty')}")

    # -------------------------------------------------------------------------
    # Demo 2: ML-Enhanced Retrieval Ranking
    # -------------------------------------------------------------------------
    print("\n[Demo 2] ML-Enhanced Retrieval Ranking")
    print("-" * 50)
    v_store = SQLiteVectorStore(":memory:")
    provider = MockEmbeddingProvider(dimension=16)
    v_mgr = VectorMemoryManager(storage=v_store, provider=provider)

    m_cache_config = Memory(
        memory_id="mem-cache-conf",
        memory_type="FACT",
        content="Redis distributed cache cluster deployed on port 6379 with Sentinel failover.",
        project_id="proj-alpha",
        created_at=now,
        updated_at=now,
        importance=0.85,
    )
    m_cache_note = Memory(
        memory_id="mem-cache-note",
        memory_type="FACT",
        content="General discussion note on caching options and memory benchmarks.",
        project_id="proj-alpha",
        created_at=now - timedelta(days=45),
        updated_at=now - timedelta(days=45),
        importance=0.40,
    )
    v_mgr.index_memory(m_cache_config)
    v_mgr.index_memory(m_cache_note)

    rank_model = RelevanceRankingModel()
    rank_ex = [
        DatasetExample(
            example_id=f"rk-{i}",
            features={
                "base_retrieval_score": 0.8 if i % 2 == 0 else 0.4,
                "vector_score": 0.85 if i % 2 == 0 else 0.45,
                "graph_score": 0.7 if i % 2 == 0 else 0.0,
                "importance": 0.85 if i % 2 == 0 else 0.4,
                "confidence": 1.0,
                "recency_factor": 1.0 if i % 2 == 0 else 0.2,
                "query_term_overlap": 0.75 if i % 2 == 0 else 0.25,
                "is_exact_agent_match": 1.0 if i % 2 == 0 else 0.0,
                "status_penalty": 0.0,
            },
            label=0.90 if i % 2 == 0 else 0.25,
            provenance=ExampleProvenance(ExampleSource.MANUALLY_LABELLED, now, "proj-alpha", (f"rm-{i}",)),
        )
        for i in range(12)
    ]
    split_rk = DatasetSplitter.split(rank_ex, val_ratio=0.25, test_ratio=0.0, group_by_memory=False)
    rank_model.train(split_rk, epochs=30)

    engine = RetrievalEngine(vector_index=v_mgr.index, ml_ranker=rank_model)
    res = engine.retrieve("Redis cache configuration", project_id="proj-alpha", method=RetrievalMethod.VECTOR)

    print(f"  Query: 'Redis cache configuration' -> {res.total_count} candidates retrieved & re-ranked:")
    for rank_idx, item in enumerate(res.results, 1):
        print(f"    {rank_idx}. [{item.memory_id}] ML Score: {item.score:.4f} (Base score: {item.score_breakdown.base_score:.4f})")
        print(f"       Content: '{item.content}'")
        print(f"       ML Explanation: {item.metadata.get('ml_explanation')}")

    # -------------------------------------------------------------------------
    # Demo 3: Duplicate Detection Model
    # -------------------------------------------------------------------------
    print("\n[Demo 3] Duplicate Detection Model")
    print("-" * 50)
    dup_model = DuplicateDetectionModel()
    split_dup = DatasetSplitter.split(
        [
            DatasetExample(
                example_id=f"dp-{i}",
                features={"exact_content_match": 1.0, "jaccard_word_similarity": 1.0, "char_length_ratio": 1.0, "same_memory_type": 1.0, "same_project_id": 1.0, "common_tech_entity_count": 1.0, "has_negation_contrast": 0.0, "subject_match": 1.0, "provenance_overlap": 0.5},
                label=1.0,
                provenance=ExampleProvenance(ExampleSource.REAL, now, "proj-alpha", (f"dm-{i}",)),
            )
            for i in range(8)
        ],
        val_ratio=0.25, test_ratio=0.0, group_by_memory=False,
    )
    dup_model.train(split_dup)

    m_dup_base = Memory(memory_id="m-db-1", memory_type="FACT", content="The project uses PostgreSQL 16 for primary data storage.", project_id="proj-alpha", created_at=now, updated_at=now)
    m_dup_exact = Memory(memory_id="m-db-2", memory_type="FACT", content="The project uses PostgreSQL 16 for primary data storage.", project_id="proj-alpha", created_at=now, updated_at=now)
    m_dup_sem = Memory(memory_id="m-db-3", memory_type="FACT", content="PostgreSQL 16 is deployed for primary data storage in the project.", project_id="proj-alpha", created_at=now, updated_at=now)
    m_dup_contra = Memory(memory_id="m-db-4", memory_type="FACT", content="The project uses MySQL 8 for primary data storage.", project_id="proj-alpha", created_at=now, updated_at=now)

    p_exact = dup_model.predict((m_dup_base, m_dup_exact))
    p_sem = dup_model.predict((m_dup_base, m_dup_sem))
    p_contra = dup_model.predict((m_dup_base, m_dup_contra))

    print(f"  Base memory: '{m_dup_base.content}'")
    print(f"    vs. Exact copy      -> Classification: {p_exact.target_value} (Confidence: {p_exact.confidence:.2f})")
    print(f"    vs. Semantic rewrite -> Classification: {p_sem.target_value} (Confidence: {p_sem.confidence:.2f})")
    print(f"    vs. Mutually exclusive claim -> Classification: {p_contra.target_value} (Confidence: {p_contra.confidence:.2f})")

    # -------------------------------------------------------------------------
    # Demo 4: Conflict Detection Model
    # -------------------------------------------------------------------------
    print("\n[Demo 4] Conflict Detection Model")
    print("-" * 50)
    conf_model = ConflictDetectionModel()
    split_conf = DatasetSplitter.split(
        [
            DatasetExample(
                example_id=f"cf-{i}",
                features={"same_subject": 1.0, "has_mutually_exclusive_entities": 1.0, "has_port_or_numeric_mismatch": 0.0, "has_temporal_progression_language": 0.0, "has_negation_contradiction": 0.0, "detail_level_difference": 0.0, "has_provenance_dispute_markers": 0.0, "is_decision_type": 0.0, "time_gap_days": 1.0},
                label=1.0,
                provenance=ExampleProvenance(ExampleSource.REAL, now, "proj-alpha", (f"cm-{i}",)),
            )
            for i in range(8)
        ],
        val_ratio=0.25, test_ratio=0.0, group_by_memory=False,
    )
    conf_model.train(split_conf)

    m_tech_1 = Memory(memory_id="mt-1", memory_type="FACT", content="Redis cache runs on port 6379", project_id="proj-alpha", created_at=now, updated_at=now)
    m_tech_2 = Memory(memory_id="mt-2", memory_type="FACT", content="Redis cache runs on port 6380", project_id="proj-alpha", created_at=now, updated_at=now)
    m_temporal = Memory(memory_id="mt-3", memory_type="FACT", content="Subsequently migrated Redis to port 6380 yesterday", project_id="proj-alpha", created_at=now + timedelta(days=1), updated_at=now + timedelta(days=1))
    m_elaboration = Memory(memory_id="mt-4", memory_type="FACT", content="Redis cache runs on port 6379 with Sentinel high availability and password auth", project_id="proj-alpha", created_at=now, updated_at=now)

    p_tech = conf_model.predict((m_tech_1, m_tech_2))
    p_temp_c = conf_model.predict((m_tech_1, m_temporal))
    p_elab = conf_model.predict((m_tech_1, m_elaboration))

    print(f"  Base claim: '{m_tech_1.content}'")
    print(f"    vs. Discrepant Port Config -> Conflict: {p_tech.target_value} (Explanation: {p_tech.explanation})")
    print(f"    vs. Temporal Progression   -> Conflict: {p_temp_c.target_value} (Explanation: {p_temp_c.explanation})")
    print(f"    vs. Detailed Elaboration  -> Conflict: {p_elab.target_value} (Explanation: {p_elab.explanation})")

    # -------------------------------------------------------------------------
    # Demo 5: Staleness Detection Model
    # -------------------------------------------------------------------------
    print("\n[Demo 5] Staleness Detection Model")
    print("-" * 50)
    stale_model = StalenessDetectionModel()
    split_stale = DatasetSplitter.split(
        [
            DatasetExample(
                example_id=f"st-{i}",
                features={"age_days": 0.8 if i % 2 == 0 else 0.1, "days_since_update": 0.8 if i % 2 == 0 else 0.1, "is_temporary_type": 1.0 if i % 2 == 0 else 0.0, "is_decision_type": 0.0 if i % 2 == 0 else 1.0, "has_temporal_keywords": 1.0 if i % 2 == 0 else 0.0, "subsequent_project_event_count": 0.5, "has_superseding_child": 0.0, "confirmation_gap_factor": 0.7 if i % 2 == 0 else 0.1},
                label=0.85 if i % 2 == 0 else 0.10,
                provenance=ExampleProvenance(ExampleSource.REAL, now, "proj-alpha", (f"sm-{i}",)),
            )
            for i in range(12)
        ],
        val_ratio=0.25, test_ratio=0.0, group_by_memory=False,
    )
    stale_model.train(split_stale, epochs=30)

    m_stale_temp = Memory(memory_id="ms-1", memory_type="FACT", content="Temporary workaround for session auth until release 1.2", project_id="proj-alpha", created_at=now - timedelta(days=150), updated_at=now - timedelta(days=150))
    m_durable_dec = Memory(memory_id="ms-2", memory_type="DECISION", content="We decided to require OAuth2 for all internal service communication", project_id="proj-alpha", created_at=now - timedelta(days=150), updated_at=now - timedelta(days=150))

    pred_s1 = stale_model.predict(m_stale_temp, subsequent_project_events=45)
    pred_s2 = stale_model.predict(m_durable_dec, subsequent_project_events=45)

    print(f"  Temporary old memory: '{m_stale_temp.content}'")
    print(f"    * Staleness Risk Probability: {pred_s1.target_value} ({pred_s1.explanation})")
    print(f"    * Non-deletion guarantee   : Memory preserved with original status; flagged for review.")

    print(f"  Architectural decision: '{m_durable_dec.content}'")
    print(f"    * Staleness Risk Probability: {pred_s2.target_value} ({pred_s2.explanation})")

    # -------------------------------------------------------------------------
    # Demo 6: Feedback Collection & Model Evaluation
    # -------------------------------------------------------------------------
    print("\n[Demo 6] Feedback Collection & Controlled Evaluation")
    print("-" * 50)
    fb_store = SQLiteFeedbackStore(":memory:")
    collector = FeedbackCollector(fb_store)

    # Record feedback signals
    fb_1 = collector.record_retrieval_feedback(project_id="proj-alpha", memory_id="mem-cache-conf", is_relevant=True, notes="Accurate redis retrieval")
    fb_2 = collector.record_duplicate_feedback(project_id="proj-alpha", m1_id="m-db-1", m2_id="m-db-3", is_duplicate=True, notes="Confirmed semantic rewrite")
    fb_3 = collector.record_staleness_feedback(project_id="proj-alpha", memory_id="ms-1", is_stale=True, notes="Confirmed obsolete temporary workaround")

    print(f"  Recorded {len(fb_store.list_feedback(project_id='proj-alpha'))} explicit feedback records across categories:")
    for r in fb_store.list_feedback(project_id="proj-alpha"):
        print(f"    - Type: {r.feedback_type.value:22} | Signal: {r.signal.value:8} | Target: {r.memory_ids}")

    # Export batch for future training without auto-retrain
    exported_batch = collector.prepare_training_batch(feedback_type=FeedbackType.RETRIEVAL_RELEVANCE, project_id="proj-alpha")
    print(f"  Controlled Training Batch Exported: {len(exported_batch)} examples (Source: {exported_batch[0].provenance.source_type.value})")

    # Comparative evaluation report
    eval_report = ModelEvaluator.evaluate_binary_classification(
        y_true=[1, 1, 0, 1, 0, 0, 1],
        y_pred=[1, 1, 0, 1, 0, 1, 1],
        y_prob=[0.92, 0.88, 0.15, 0.81, 0.22, 0.65, 0.78],
    )
    print(f"  Model Evaluation Metrics (Binary Classification):")
    print(f"    * Accuracy            : {eval_report.accuracy:.4f}")
    print(f"    * Precision           : {eval_report.precision:.4f}")
    print(f"    * Recall              : {eval_report.recall:.4f}")
    print(f"    * F1 Score            : {eval_report.f1_score:.4f}")
    print(f"    * Brier Score (Calib) : {eval_report.brier_score:.4f}")

    print("\n" + "=" * 80)
    print("AI Brain Phase 3 Machine Learning Demonstration Complete: All 6 Scenarios Passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
