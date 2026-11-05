"""
Brain Loop orchestrator coordinating the complete interaction lifecycle.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import uuid

from ai_brain.context import ContextBuilder
from ai_brain.context.models import ContextPackage
from ai_brain.events.models import Event
from ai_brain.events.storage.base import EventStore
from ai_brain.events.storage.sqlite import SQLiteEventStore
from ai_brain.graph.graph import KnowledgeGraph
from ai_brain.graph.integration import GraphIntegrator
from ai_brain.loop.event_handler import LoopEventHandler
from ai_brain.loop.exceptions import (
    BrainLoopError,
    ContextStageError,
    EventPersistenceError,
    ExtractionStageError,
    GraphIntegrationStageError,
    InputValidationError,
    InteractionNotFoundError,
    InteractionStateError,
    LoopProjectIsolationError,
    MemoryUpdateStageError,
    ResponseValidationError,
    RetrievalStageError,
    VectorIntegrationStageError,
)
from ai_brain.loop.input_processor import InputProcessor
from ai_brain.loop.memory_handler import LoopMemoryHandler, LoopMemoryProcessingResult
from ai_brain.loop.models import (
    AgentInteractionInput,
    AgentResponseInput,
    InteractionCompletionResult,
    InteractionContextResult,
    InteractionRecord,
    InteractionStatus,
    LoopStage,
)
from ai_brain.loop.response_capture import ResponseCapture
from ai_brain.loop.storage.base import InteractionStore
from ai_brain.loop.storage.sqlite import SQLiteInteractionStore
from ai_brain.memories.conflicts.detector import ConflictDetector
from ai_brain.memories.conflicts.manager import ConflictManager
from ai_brain.memories.extraction.extractor import MemoryExtractor
from ai_brain.memories.models import Memory
from ai_brain.memories.updating.updater import MemoryUpdater
from ai_brain.retrieval.engine import RetrievalEngine
from ai_brain.retrieval.models import RetrievalResponse
from ai_brain.vectors.manager import VectorMemoryManager

logger = logging.getLogger(__name__)


class BrainLoop:
    """
    Central orchestration facade managing the complete lifecycle of agent interactions.
    
    Stages:
    1. Input validation & project/session isolation check.
    2. Input event persistence (idempotent, UUID v5).
    3. Multi-strategy memory retrieval.
    4. Prioritized, budgeted context package assembly.
    5. Context delivery to agent caller.
    6. Agent response receipt & validation.
    7. Response event persistence.
    8. Rule-based memory extraction & deduplication.
    9. Memory updating, version chain management & conflict detection.
    10. Downstream synchronization to Knowledge Graph and Vector Memory.
    11. Interaction completion event recording & status reporting.
    """

    def __init__(
        self,
        *,
        event_store: EventStore,
        interaction_store: Optional[InteractionStore] = None,
        retrieval_engine: Optional[RetrievalEngine] = None,
        context_builder: Optional[ContextBuilder] = None,
        memory_extractor: Optional[MemoryExtractor] = None,
        memory_updater: Optional[MemoryUpdater] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        conflict_manager: Optional[ConflictManager] = None,
        graph: Optional[KnowledgeGraph] = None,
        graph_integrator: Optional[GraphIntegrator] = None,
        vector_manager: Optional[VectorMemoryManager] = None,
        input_processor: Optional[InputProcessor] = None,
        response_capture: Optional[ResponseCapture] = None,
        event_handler: Optional[LoopEventHandler] = None,
        memory_handler: Optional[LoopMemoryHandler] = None,
        default_token_budget: Optional[int] = None,
        ml_service: Optional[Any] = None,
    ) -> None:
        if event_store is None:
            raise ValueError("An EventStore instance is required for BrainLoop")
        self._closed = False
        self.event_store = event_store
        self.ml_service = ml_service

        # Interaction storage resolution
        if interaction_store is not None:
            self.interaction_store = interaction_store
        elif isinstance(event_store, SQLiteEventStore) and event_store.database_path != ":memory:":
            self.interaction_store = SQLiteInteractionStore(event_store.database_path)
        else:
            self.interaction_store = SQLiteInteractionStore(":memory:")

        self.retrieval_engine = retrieval_engine
        if self.retrieval_engine is not None and getattr(self.retrieval_engine, "ml_ranker", None) is None:
            if self.ml_service is not None and getattr(self.ml_service.config, "enable_ranking_ml", False):
                self.retrieval_engine.ml_ranker = self.ml_service.registry.get_ranking_model()
        self.context_builder = context_builder or ContextBuilder(default_max_tokens=default_token_budget or 1000)
        self.default_token_budget = default_token_budget

        self.input_processor = input_processor or InputProcessor()
        self.response_capture = response_capture or ResponseCapture()
        self.event_handler = event_handler or LoopEventHandler(event_store=self.event_store)

        # Graph integration resolution
        resolved_graph_integrator = graph_integrator
        if resolved_graph_integrator is None and graph is not None:
            resolved_graph_integrator = GraphIntegrator(graph)

        self.memory_handler = memory_handler or LoopMemoryHandler(
            memory_extractor=memory_extractor,
            memory_updater=memory_updater,
            conflict_detector=conflict_detector,
            conflict_manager=conflict_manager,
            graph_integrator=resolved_graph_integrator,
            vector_manager=vector_manager,
            ml_service=self.ml_service,
        )

    @property
    def memory_extractor(self) -> MemoryExtractor:
        return self.memory_handler.memory_extractor

    @property
    def memory_updater(self) -> MemoryUpdater:
        return self.memory_handler.memory_updater

    @property
    def conflict_detector(self) -> ConflictDetector:
        return self.memory_handler.conflict_detector

    @property
    def conflict_manager(self) -> ConflictManager:
        return self.memory_handler.conflict_manager

    @property
    def graph_integrator(self) -> Optional[GraphIntegrator]:
        return self.memory_handler.graph_integrator

    @property
    def vector_manager(self) -> Optional[VectorMemoryManager]:
        return self.memory_handler.vector_manager

    def close(self) -> None:
        """Close storage connections (idempotent)."""
        if not getattr(self, "_closed", False):
            self._closed = True
            store = getattr(self, "interaction_store", None)
            if store is not None and hasattr(store, "close"):
                try:
                    store.close()
                except Exception:
                    pass

    def __enter__(self) -> "BrainLoop":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_interaction(self, interaction_id: str) -> Optional[InteractionRecord]:
        """Look up an interaction record by ID from persistent storage."""
        return self.interaction_store.get(interaction_id)

    def list_interactions(
        self,
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[Union[str, InteractionStatus]] = None,
    ) -> List[InteractionRecord]:
        """Filter interaction records by project, session, or status from persistent storage."""
        return self.interaction_store.list(
            project_id=project_id,
            session_id=session_id,
            status=status,
        )

    def register_memories(self, memories: Sequence[Memory]) -> None:
        """Seed known existing memories into the memory handler."""
        self.memory_handler.register_memories(memories)

    def _mark_failed(
        self,
        record: InteractionRecord,
        stage: str,
        error_message: str,
    ) -> InteractionRecord:
        """Persist stage failure on the interaction record."""
        failed_record = InteractionRecord(
            interaction_id=record.interaction_id,
            idempotency_key=record.idempotency_key,
            project_id=record.project_id,
            agent_id=record.agent_id,
            session_id=record.session_id,
            status=InteractionStatus.FAILED,
            current_stage=stage,
            stages_completed=record.stages_completed,
            input=record.input,
            input_event_id=record.input_event_id,
            context_event_id=record.context_event_id,
            context_result=record.context_result,
            response=record.response,
            response_event_id=record.response_event_id,
            completion_event_id=record.completion_event_id,
            completion_result=record.completion_result,
            error_message=error_message,
            warnings=record.warnings,
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self.interaction_store.save(failed_record)
        return failed_record

    def prepare_context(
        self,
        interaction_input: AgentInteractionInput,
    ) -> InteractionContextResult:
        """
        Stage A: Process agent input, retrieve relevant memories, and deliver context.
        """
        # Step 1: Validate input
        processed_input = self.input_processor.process(interaction_input)

        # Idempotency lookup
        existing_record: Optional[InteractionRecord] = None
        if processed_input.idempotency_key:
            existing_record = self.interaction_store.get_by_idempotency_key(processed_input.idempotency_key)
        elif processed_input.interaction_id:
            existing_record = self.interaction_store.get(processed_input.interaction_id)

        if existing_record is not None:
            # Check payload match
            if (
                existing_record.project_id != processed_input.project_id
                or existing_record.agent_id != processed_input.agent_id
                or existing_record.session_id != processed_input.session_id
                or existing_record.input.content != processed_input.content
            ):
                id_desc = (
                    f"Idempotency key '{processed_input.idempotency_key}'"
                    if processed_input.idempotency_key
                    else f"Interaction ID '{processed_input.interaction_id}'"
                )
                raise InputValidationError(f"{id_desc} reused with mismatched input payload")

            # If already prepared and valid, return cached context result
            if existing_record.context_result is not None:
                return existing_record.context_result

            # If interrupted/failed previously, resume from existing record
            record = existing_record
            iid = record.interaction_id
        else:
            iid = processed_input.interaction_id
            record = InteractionRecord(
                interaction_id=iid,
                idempotency_key=processed_input.idempotency_key,
                project_id=processed_input.project_id,
                agent_id=processed_input.agent_id,
                session_id=processed_input.session_id,
                input=processed_input,
                status=InteractionStatus.INITIALIZED,
                current_stage=LoopStage.INITIALIZED.value,
                stages_completed=(),
                created_at=processed_input.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self.interaction_store.save(record)

        # Stage 1: Record input event in persistent store
        input_event_id = record.input_event_id
        if LoopStage.INPUT_PERSISTED.value not in record.stages_completed:
            try:
                input_event = self.event_handler.record_input_event(processed_input)
                input_event_id = input_event.event_id
                stages = tuple(record.stages_completed) + (LoopStage.INPUT_PERSISTED.value,)
                record = InteractionRecord(
                    interaction_id=iid,
                    idempotency_key=record.idempotency_key,
                    project_id=record.project_id,
                    agent_id=record.agent_id,
                    session_id=record.session_id,
                    input=record.input,
                    status=InteractionStatus.INPUT_RECEIVED,
                    current_stage=LoopStage.INPUT_PERSISTED.value,
                    stages_completed=stages,
                    input_event_id=input_event_id,
                    context_event_id=record.context_event_id,
                    context_result=record.context_result,
                    created_at=record.created_at,
                    updated_at=datetime.now(timezone.utc),
                )
                self.interaction_store.save(record)
            except Exception as e:
                self._mark_failed(record, LoopStage.INPUT_PERSISTED.value, str(e))
                raise

        # Stage 2: Retrieve relevant memories
        retrieval_response: Optional[RetrievalResponse] = None
        if self.retrieval_engine is not None:
            try:
                retrieval_response = self.retrieval_engine.retrieve(
                    query=processed_input.query,
                    project_id=processed_input.project_id,
                    method=processed_input.retrieval_method,
                    top_k=processed_input.top_k,
                    filters=processed_input.retrieval_filters,
                )
            except Exception as e:
                self.event_handler.record_error_event(
                    interaction_id=iid,
                    project_id=processed_input.project_id,
                    agent_id=processed_input.agent_id,
                    session_id=processed_input.session_id,
                    stage="retrieval",
                    error_message=str(e),
                )
                self._mark_failed(record, LoopStage.CONTEXT_RETRIEVED.value, str(e))
                raise RetrievalStageError(
                    f"Retrieval failed for interaction '{iid}': {e}"
                ) from e
        else:
            retrieval_response = RetrievalResponse(
                query=processed_input.query or "",
                normalized_query=processed_input.query or "",
                project_id=processed_input.project_id,
                method=processed_input.retrieval_method,
                results=(),
            )

        if LoopStage.CONTEXT_RETRIEVED.value not in record.stages_completed:
            stages = tuple(record.stages_completed) + (LoopStage.CONTEXT_RETRIEVED.value,)
            record = InteractionRecord(
                interaction_id=iid,
                idempotency_key=record.idempotency_key,
                project_id=record.project_id,
                agent_id=record.agent_id,
                session_id=record.session_id,
                input=record.input,
                status=record.status,
                current_stage=LoopStage.CONTEXT_RETRIEVED.value,
                stages_completed=stages,
                input_event_id=input_event_id,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self.interaction_store.save(record)

        # Stage 3: Assemble Context Package within token budget & record delivery event
        try:
            budget = processed_input.token_budget or self.default_token_budget
            context_package = self.context_builder.build_context(
                retrieval_response,
                project_id=processed_input.project_id,
                query=processed_input.query,
                max_tokens=budget,
            )
            context_event = self.event_handler.record_context_event(
                interaction_input=processed_input,
                parent_event_id=input_event_id or "",
                tokens_used=context_package.tokens_used,
                token_budget=context_package.token_budget,
                items_count=len(context_package.items),
                has_disputes=context_package.has_disputes,
                has_omitted_disputes=context_package.has_omitted_disputes,
            )
        except Exception as e:
            self.event_handler.record_error_event(
                interaction_id=iid,
                project_id=processed_input.project_id,
                agent_id=processed_input.agent_id,
                session_id=processed_input.session_id,
                stage="context_builder",
                error_message=str(e),
            )
            self._mark_failed(record, LoopStage.CONTEXT_DELIVERED.value, str(e))
            raise ContextStageError(
                f"Context construction failed for interaction '{iid}': {e}"
            ) from e

        context_result = InteractionContextResult(
            interaction_id=iid,
            project_id=processed_input.project_id,
            agent_id=processed_input.agent_id,
            session_id=processed_input.session_id,
            input_event_id=input_event_id or "",
            context_event_id=context_event.event_id,
            context_package=context_package,
            retrieval_response=retrieval_response,
            status=InteractionStatus.CONTEXT_PREPARED,
            metadata={"retrieval_method": processed_input.retrieval_method},
            created_at=datetime.now(timezone.utc),
        )

        stages = tuple(record.stages_completed) + (LoopStage.CONTEXT_DELIVERED.value,)
        prepared_record = InteractionRecord(
            interaction_id=iid,
            idempotency_key=record.idempotency_key,
            project_id=processed_input.project_id,
            agent_id=processed_input.agent_id,
            session_id=processed_input.session_id,
            input=processed_input,
            status=InteractionStatus.CONTEXT_PREPARED,
            current_stage=LoopStage.CONTEXT_DELIVERED.value,
            stages_completed=stages,
            input_event_id=input_event_id,
            context_event_id=context_event.event_id,
            context_result=context_result,
            created_at=processed_input.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self.interaction_store.save(prepared_record)

        return context_result

    def complete_interaction(
        self,
        response_input: AgentResponseInput,
        *,
        raise_on_sync_failure: bool = False,
    ) -> InteractionCompletionResult:
        """
        Stage B: Accept agent response, persist events, extract memories, and update state.
        """
        iid = response_input.interaction_id
        record = self.interaction_store.get(iid)

        # Idempotency check: if already completed, verify payload and return existing result
        if record and record.status in (InteractionStatus.COMPLETED, InteractionStatus.PARTIAL_SUCCESS):
            if record.response is not None and record.response.content != response_input.content:
                raise ResponseValidationError(
                    f"Interaction '{iid}' was already completed with different response content"
                )
            if record.completion_result is not None:
                return record.completion_result

        # Step 1: Validate response against originating interaction
        validated_response = self.response_capture.validate_response(response_input, record)

        # Stage 4: Record response event in persistent store
        response_event_id = record.response_event_id if record else None
        if not record or LoopStage.RESPONSE_PERSISTED.value not in record.stages_completed:
            parent_id = record.context_event_id or record.input_event_id or "" if record else ""
            try:
                response_event = self.event_handler.record_response_event(
                    response_input=validated_response,
                    parent_event_id=parent_id,
                )
                response_event_id = response_event.event_id
                stages = tuple(record.stages_completed) + (LoopStage.RESPONSE_PERSISTED.value,)
                record = InteractionRecord(
                    interaction_id=iid,
                    idempotency_key=record.idempotency_key,
                    project_id=validated_response.project_id,
                    agent_id=validated_response.agent_id,
                    session_id=validated_response.session_id,
                    input=record.input,
                    status=InteractionStatus.RESPONSE_RECEIVED,
                    current_stage=LoopStage.RESPONSE_PERSISTED.value,
                    stages_completed=stages,
                    input_event_id=record.input_event_id,
                    context_event_id=record.context_event_id,
                    context_result=record.context_result,
                    response=validated_response,
                    response_event_id=response_event_id,
                    created_at=record.created_at,
                    updated_at=datetime.now(timezone.utc),
                )
                self.interaction_store.save(record)
            except Exception as e:
                if record:
                    self._mark_failed(record, LoopStage.RESPONSE_PERSISTED.value, str(e))
                raise

        # Stage 5-9: Memory Extraction, Updating, and Downstream Synchronization
        events_to_process = []
        if record.input_event_id:
            in_ev = self.event_store.get_event_by_id(record.input_event_id)
            if in_ev:
                events_to_process.append(in_ev)
        if response_event_id:
            resp_ev = self.event_store.get_event_by_id(response_event_id)
            if resp_ev:
                events_to_process.append(resp_ev)

        # Step 5: Memory extraction
        try:
            extracted_memories = self.memory_handler.extract_memories(
                events_to_process, project_id=validated_response.project_id
            )
            stages = tuple(record.stages_completed)
            if LoopStage.MEMORY_EXTRACTED.value not in stages:
                stages = stages + (LoopStage.MEMORY_EXTRACTED.value,)
            record = InteractionRecord(
                interaction_id=iid,
                idempotency_key=record.idempotency_key,
                project_id=record.project_id,
                agent_id=record.agent_id,
                session_id=record.session_id,
                input=record.input,
                status=record.status,
                current_stage=LoopStage.MEMORY_EXTRACTED.value,
                stages_completed=stages,
                input_event_id=record.input_event_id,
                context_event_id=record.context_event_id,
                context_result=record.context_result,
                response=record.response,
                response_event_id=record.response_event_id,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self.interaction_store.save(record)
        except Exception as e:
            self._mark_failed(record, LoopStage.MEMORY_EXTRACTED.value, str(e))
            raise

        # Step 6 & 7: Memory Updating and Conflict Detection
        try:
            update_results, conflicts, update_warnings = self.memory_handler.update_memories_and_conflicts(
                extracted_memories, project_id=validated_response.project_id
            )
            stages = tuple(record.stages_completed)
            if LoopStage.MEMORY_UPDATED.value not in stages:
                stages = stages + (LoopStage.MEMORY_UPDATED.value,)
            if LoopStage.CONFLICTS_CHECKED.value not in stages:
                stages = stages + (LoopStage.CONFLICTS_CHECKED.value,)
            record = InteractionRecord(
                interaction_id=iid,
                idempotency_key=record.idempotency_key,
                project_id=record.project_id,
                agent_id=record.agent_id,
                session_id=record.session_id,
                input=record.input,
                status=record.status,
                current_stage=LoopStage.CONFLICTS_CHECKED.value,
                stages_completed=stages,
                input_event_id=record.input_event_id,
                context_event_id=record.context_event_id,
                context_result=record.context_result,
                response=record.response,
                response_event_id=record.response_event_id,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self.interaction_store.save(record)
        except Exception as e:
            self._mark_failed(record, LoopStage.MEMORY_UPDATED.value, str(e))
            raise

        # Step 8: Graph Integration
        try:
            nodes_integrated, graph_warnings = self.memory_handler.sync_graph(
                update_results, conflicts, raise_on_failure=raise_on_sync_failure
            )
            stages = tuple(record.stages_completed)
            if LoopStage.GRAPH_SYNCED.value not in stages:
                stages = stages + (LoopStage.GRAPH_SYNCED.value,)
            record = InteractionRecord(
                interaction_id=iid,
                idempotency_key=record.idempotency_key,
                project_id=record.project_id,
                agent_id=record.agent_id,
                session_id=record.session_id,
                input=record.input,
                status=record.status,
                current_stage=LoopStage.GRAPH_SYNCED.value,
                stages_completed=stages,
                input_event_id=record.input_event_id,
                context_event_id=record.context_event_id,
                context_result=record.context_result,
                response=record.response,
                response_event_id=record.response_event_id,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self.interaction_store.save(record)
        except Exception as e:
            self._mark_failed(record, LoopStage.GRAPH_SYNCED.value, str(e))
            raise

        # Step 9: Vector Memory Indexing
        try:
            vectors_indexed, vector_warnings = self.memory_handler.sync_vectors(
                update_results, raise_on_failure=raise_on_sync_failure
            )
            stages = tuple(record.stages_completed)
            if LoopStage.VECTOR_SYNCED.value not in stages:
                stages = stages + (LoopStage.VECTOR_SYNCED.value,)
            record = InteractionRecord(
                interaction_id=iid,
                idempotency_key=record.idempotency_key,
                project_id=record.project_id,
                agent_id=record.agent_id,
                session_id=record.session_id,
                input=record.input,
                status=record.status,
                current_stage=LoopStage.VECTOR_SYNCED.value,
                stages_completed=stages,
                input_event_id=record.input_event_id,
                context_event_id=record.context_event_id,
                context_result=record.context_result,
                response=record.response,
                response_event_id=record.response_event_id,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            self.interaction_store.save(record)
        except Exception as e:
            self._mark_failed(record, LoopStage.VECTOR_SYNCED.value, str(e))
            raise

        all_warnings = list(update_warnings) + list(graph_warnings) + list(vector_warnings)
        is_partial = len(all_warnings) > 0
        final_status = InteractionStatus.PARTIAL_SUCCESS if is_partial else InteractionStatus.COMPLETED

        # Step 10: Record completion event in persistent event store
        try:
            completion_event = self.event_handler.record_completion_event(
                interaction_id=iid,
                project_id=validated_response.project_id,
                agent_id=validated_response.agent_id,
                session_id=validated_response.session_id,
                parent_event_id=response_event_id or "",
                memories_count=len(extracted_memories),
                conflicts_count=len(conflicts),
                is_partial=is_partial,
                warnings=all_warnings,
            )
        except Exception as e:
            self._mark_failed(record, LoopStage.COMPLETED.value, str(e))
            raise

        completion_result = InteractionCompletionResult(
            interaction_id=iid,
            project_id=validated_response.project_id,
            agent_id=validated_response.agent_id,
            session_id=validated_response.session_id,
            status=final_status,
            response_event_id=response_event_id or "",
            completion_event_id=completion_event.event_id,
            extracted_memories=extracted_memories,
            update_results=update_results,
            conflicts_detected=conflicts,
            graph_nodes_integrated=nodes_integrated,
            vectors_indexed=vectors_indexed,
            warnings=tuple(all_warnings),
            is_partial_success=is_partial,
            metadata={"events_processed_count": len(events_to_process)},
            created_at=datetime.now(timezone.utc),
        )

        stages = tuple(record.stages_completed) + (LoopStage.COMPLETED.value,)
        completed_record = InteractionRecord(
            interaction_id=iid,
            idempotency_key=record.idempotency_key,
            project_id=validated_response.project_id,
            agent_id=validated_response.agent_id,
            session_id=validated_response.session_id,
            input=record.input,
            status=final_status,
            current_stage=LoopStage.COMPLETED.value,
            stages_completed=stages,
            input_event_id=record.input_event_id,
            context_event_id=record.context_event_id,
            context_result=record.context_result,
            response=validated_response,
            response_event_id=response_event_id,
            completion_event_id=completion_event.event_id,
            completion_result=completion_result,
            warnings=tuple(all_warnings),
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self.interaction_store.save(completed_record)

        # Automated feedback recording: record observed context utilization upon successful completion
        if self.ml_service is not None and getattr(self.ml_service, "feedback_collector", None) is not None:
            try:
                if record.context_result and record.context_result.context_package:
                    pkg = record.context_result.context_package
                    for item in pkg.items:
                        self.ml_service.feedback_collector.record_retrieval_feedback(
                            project_id=validated_response.project_id,
                            memory_id=item.memory_id,
                            is_relevant=True,
                            agent_id=validated_response.agent_id,
                            session_id=validated_response.session_id,
                            notes="Observed context utilization in completed BrainLoop interaction",
                        )
            except Exception as fb_err:
                logger.debug("Automated feedback collection skipped: %s", fb_err)

        return completion_result

    def assess_memory_staleness(self, memory: Memory, **context: Any) -> Any:
        """Assess obsolescence risk for a memory without deleting it."""
        if self.ml_service is not None:
            return self.ml_service.assess_staleness(memory, **context)
        status = getattr(memory, "status", "ACTIVE")
        return 1.0 if status == "SUPERSEDED" else 0.0

    def assess_project_staleness(self, project_id: str, **context: Any) -> List[Tuple[Memory, Any]]:
        """Assess obsolescence risk for all active project memories without deleting any data."""
        active_mems = self.memory_handler.get_active_memories(project_id)
        results = []
        for m in active_mems:
            pred = self.assess_memory_staleness(m, **context)
            results.append((m, pred))
        return results

    def record_feedback(self, **kwargs: Any) -> Any:
        """Record manual human or agent feedback signal with explicit provenance."""
        if self.ml_service is not None:
            fb_type = kwargs.get("feedback_type")
            if fb_type == "RETRIEVAL_RELEVANCE" or kwargs.get("is_relevant") is not None:
                return self.ml_service.feedback_collector.record_retrieval_feedback(**kwargs)
            elif fb_type == "DUPLICATE_DETECTION" or kwargs.get("is_duplicate") is not None:
                return self.ml_service.feedback_collector.record_duplicate_feedback(**kwargs)
            elif fb_type == "CONFLICT_DETECTION" or kwargs.get("is_conflict") is not None:
                return self.ml_service.feedback_collector.record_conflict_feedback(**kwargs)
            elif fb_type == "IMPORTANCE_PREDICTION" or kwargs.get("is_useful") is not None:
                return self.ml_service.feedback_collector.record_importance_feedback(**kwargs)
            elif fb_type == "STALENESS_PREDICTION" or kwargs.get("is_stale") is not None:
                return self.ml_service.feedback_collector.record_staleness_feedback(**kwargs)
        return None

    def resume_interaction(
        self,
        interaction_id: str,
        *,
        response_input: Optional[AgentResponseInput] = None,
    ) -> Union[InteractionContextResult, InteractionCompletionResult]:
        """
        Diagnose and resume an interrupted or failed interaction from its last persisted stage.
        """
        record = self.interaction_store.get(interaction_id)
        if record is None:
            raise InteractionNotFoundError(f"Interaction '{interaction_id}' not found for resumption")

        # Case 1: Interrupted during prepare_context (before CONTEXT_DELIVERED)
        if LoopStage.CONTEXT_DELIVERED.value not in record.stages_completed:
            return self.prepare_context(record.input)

        # Case 2: Already prepared, now resuming completion
        resp = response_input or record.response
        if resp is None:
            if record.context_result is not None:
                return record.context_result
            raise InteractionStateError(
                f"Interaction '{interaction_id}' was prepared but no response input is available to resume completion"
            )

        return self.complete_interaction(resp)

    def retry_failed_sync(self, interaction_id: str) -> InteractionCompletionResult:
        """
        Retry downstream graph and vector synchronization for an interaction with warnings.
        """
        record = self.interaction_store.get(interaction_id)
        if record is None:
            raise InteractionNotFoundError(f"Interaction '{interaction_id}' not found")

        if record.completion_result is None or not record.completion_result.update_results:
            raise InteractionStateError(f"No update results available to resync for interaction '{interaction_id}'")

        res = record.completion_result
        nodes, g_warn = self.memory_handler.sync_graph(res.update_results, res.conflicts_detected)
        vecs, v_warn = self.memory_handler.sync_vectors(res.update_results)

        new_warnings = list(g_warn) + list(v_warn)
        is_partial = len(new_warnings) > 0
        new_status = InteractionStatus.PARTIAL_SUCCESS if is_partial else InteractionStatus.COMPLETED

        updated_res = InteractionCompletionResult(
            interaction_id=res.interaction_id,
            project_id=res.project_id,
            agent_id=res.agent_id,
            session_id=res.session_id,
            status=new_status,
            response_event_id=res.response_event_id,
            completion_event_id=res.completion_event_id,
            extracted_memories=res.extracted_memories,
            update_results=res.update_results,
            conflicts_detected=res.conflicts_detected,
            graph_nodes_integrated=res.graph_nodes_integrated + nodes,
            vectors_indexed=res.vectors_indexed + vecs,
            warnings=tuple(new_warnings),
            is_partial_success=is_partial,
            metadata=dict(res.metadata),
            created_at=datetime.now(timezone.utc),
        )

        updated_record = InteractionRecord(
            interaction_id=record.interaction_id,
            idempotency_key=record.idempotency_key,
            project_id=record.project_id,
            agent_id=record.agent_id,
            session_id=record.session_id,
            input=record.input,
            status=new_status,
            current_stage=record.current_stage,
            stages_completed=record.stages_completed,
            input_event_id=record.input_event_id,
            context_event_id=record.context_event_id,
            context_result=record.context_result,
            response=record.response,
            response_event_id=record.response_event_id,
            completion_event_id=record.completion_event_id,
            completion_result=updated_res,
            warnings=tuple(new_warnings),
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self.interaction_store.save(updated_record)
        return updated_res

    def run_interaction(
        self,
        interaction_input: AgentInteractionInput,
        agent_callback: Callable[[InteractionContextResult], str],
    ) -> tuple[InteractionContextResult, InteractionCompletionResult]:
        """
        Convenience end-to-end execution helper invoking a simulated agent callable.
        """
        context_res = self.prepare_context(interaction_input)
        agent_output = agent_callback(context_res)

        response_input = AgentResponseInput(
            interaction_id=context_res.interaction_id,
            project_id=interaction_input.project_id,
            agent_id=interaction_input.agent_id,
            session_id=interaction_input.session_id,
            content=agent_output,
        )
        completion_res = self.complete_interaction(response_input)
        return context_res, completion_res
