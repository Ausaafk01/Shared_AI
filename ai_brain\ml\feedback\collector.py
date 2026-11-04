"""
FeedbackCollector for recording signals and preparing controlled training batches.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from ai_brain.ml.datasets.schemas import DatasetExample, ExampleProvenance, ExampleSource
from ai_brain.ml.feedback.models import FeedbackRecord, FeedbackSignal, FeedbackType
from ai_brain.ml.feedback.store import FeedbackStore, SQLiteFeedbackStore


class FeedbackCollector:
    """
    Central API for recording human/agent feedback on ML predictions.
    
    Guarantees:
    - Provenance and model versions are strictly attached to all records.
    - Zero automated retraining after individual feedback events.
    - Controlled batch preparation exports feedback as validated DatasetExamples.
    """

    def __init__(self, store: Optional[FeedbackStore] = None) -> None:
        self.store = store or SQLiteFeedbackStore()

    def record_retrieval_feedback(
        self,
        *,
        project_id: str,
        memory_id: str,
        is_relevant: bool,
        model_name: str = "RelevanceRankingModel",
        model_version: str = "v1.0",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.RETRIEVAL_RELEVANCE,
            signal=FeedbackSignal.POSITIVE if is_relevant else FeedbackSignal.NEGATIVE,
            project_id=project_id,
            model_name=model_name,
            model_version=model_version,
            created_at=datetime.now(timezone.utc),
            agent_id=agent_id,
            session_id=session_id,
            memory_ids=(memory_id,),
            predicted_value=1.0 if is_relevant else 0.0,
            notes=notes,
        )
        self.store.record_feedback(record)
        return record

    def record_duplicate_feedback(
        self,
        *,
        project_id: str,
        m1_id: str,
        m2_id: str,
        is_duplicate: bool,
        model_name: str = "DuplicateDetectionModel",
        model_version: str = "v1.0",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.DUPLICATE_DETECTION,
            signal=FeedbackSignal.POSITIVE if is_duplicate else FeedbackSignal.NEGATIVE,
            project_id=project_id,
            model_name=model_name,
            model_version=model_version,
            created_at=datetime.now(timezone.utc),
            agent_id=agent_id,
            session_id=session_id,
            memory_ids=(m1_id, m2_id),
            predicted_value=is_duplicate,
            notes=notes,
        )
        self.store.record_feedback(record)
        return record

    def record_conflict_feedback(
        self,
        *,
        project_id: str,
        m1_id: str,
        m2_id: str,
        is_conflict: bool,
        correct_conflict_type: Optional[str] = None,
        model_name: str = "ConflictDetectionModel",
        model_version: str = "v1.0",
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.CONFLICT_DETECTION,
            signal=FeedbackSignal.POSITIVE if is_conflict else FeedbackSignal.NEGATIVE,
            project_id=project_id,
            model_name=model_name,
            model_version=model_version,
            created_at=datetime.now(timezone.utc),
            agent_id=agent_id,
            session_id=session_id,
            memory_ids=(m1_id, m2_id),
            predicted_value=is_conflict,
            corrected_value=correct_conflict_type,
            notes=notes,
        )
        self.store.record_feedback(record)
        return record

    def record_importance_feedback(
        self,
        *,
        project_id: str,
        memory_id: str,
        is_useful: bool,
        suggested_importance: Optional[float] = None,
        model_name: str = "MemoryImportanceModel",
        model_version: str = "v1.0",
        agent_id: Optional[str] = None,
        notes: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.IMPORTANCE_PREDICTION,
            signal=FeedbackSignal.POSITIVE if is_useful else FeedbackSignal.NEGATIVE,
            project_id=project_id,
            model_name=model_name,
            model_version=model_version,
            created_at=datetime.now(timezone.utc),
            agent_id=agent_id,
            memory_ids=(memory_id,),
            corrected_value=suggested_importance,
            notes=notes,
        )
        self.store.record_feedback(record)
        return record

    def record_staleness_feedback(
        self,
        *,
        project_id: str,
        memory_id: str,
        is_stale: bool,
        model_name: str = "StalenessDetectionModel",
        model_version: str = "v1.0",
        agent_id: Optional[str] = None,
        notes: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.STALENESS_PREDICTION,
            signal=FeedbackSignal.POSITIVE if is_stale else FeedbackSignal.NEGATIVE,
            project_id=project_id,
            model_name=model_name,
            model_version=model_version,
            created_at=datetime.now(timezone.utc),
            agent_id=agent_id,
            memory_ids=(memory_id,),
            predicted_value=1.0 if is_stale else 0.0,
            notes=notes,
        )
        self.store.record_feedback(record)
        return record

    def prepare_training_batch(
        self,
        *,
        feedback_type: FeedbackType,
        project_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[DatasetExample]:
        """
        Controlled export of accumulated feedback records into validated DatasetExamples.
        Does not automatically trigger model retraining.
        """
        records = self.store.list_feedback(
            project_id=project_id,
            feedback_type=feedback_type,
            limit=limit,
        )

        examples: List[DatasetExample] = []
        for r in records:
            # Derive target label from signal
            if r.signal == FeedbackSignal.POSITIVE:
                label_val = 1.0
            elif r.signal == FeedbackSignal.NEGATIVE:
                label_val = 0.0
            else:
                label_val = float(r.corrected_value) if r.corrected_value is not None else 0.5

            prov = ExampleProvenance(
                source_type=ExampleSource.FEEDBACK_DERIVED,
                created_at=r.created_at,
                project_id=r.project_id,
                source_memory_ids=r.memory_ids,
                collector="FeedbackCollector",
                notes=f"Feedback signal: {r.signal.value} on model {r.model_name}:{r.model_version}",
            )

            # Features from snapshot or default dummy
            feats = dict(r.features_snapshot) if r.features_snapshot else {"signal_weight": 1.0}

            ex = DatasetExample(
                example_id=f"fb-{r.feedback_id}",
                features=feats,
                label=label_val,
                provenance=prov,
                metadata={"original_feedback_id": r.feedback_id},
            )
            examples.append(ex)

        return examples
