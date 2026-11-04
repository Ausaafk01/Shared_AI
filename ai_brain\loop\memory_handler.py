"""
Memory extraction, updating, conflict management, and integration handler for Brain Loop.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_brain.events.models import Event
from ai_brain.graph.integration import GraphIntegrator
from ai_brain.loop.exceptions import (
    ExtractionStageError,
    GraphIntegrationStageError,
    MemoryUpdateStageError,
    VectorIntegrationStageError,
)
from ai_brain.memories.conflicts.detector import ConflictDetector
from ai_brain.memories.conflicts.manager import ConflictManager
from ai_brain.memories.conflicts.models import MemoryConflict
from ai_brain.memories.extraction.extractor import MemoryExtractor
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus
from ai_brain.memories.updating.updater import MemoryUpdater, UpdateAction, UpdateResult
from ai_brain.vectors.manager import VectorMemoryManager

logger = logging.getLogger(__name__)

# Allowed interaction event types that carry informative domain knowledge
INFORMATIVE_EVENT_TYPES = frozenset({
    "INTERACTION_INPUT",
    "USER_MESSAGE",
    "INTERACTION_RESPONSE",
    "AGENT_MESSAGE",
    "DECISION",
    "TASK_CREATED",
    "TASK_COMPLETED",
    "CODE_CHANGE",
    "FILE_CHANGE",
})


@dataclass(frozen=True)
class LoopMemoryProcessingResult:
    """
    Outcome of memory extraction, updating, conflict handling, and downstream sync.
    """
    extracted_memories: Tuple[Memory, ...]
    update_results: Tuple[UpdateResult, ...]
    conflicts_detected: Tuple[MemoryConflict, ...]
    graph_nodes_integrated: int
    vectors_indexed: int
    warnings: Tuple[str, ...]
    is_partial_success: bool


class LoopMemoryHandler:
    """
    Coordinates extraction of memories from events, manages versioning via MemoryUpdater,
    detects conflicts via ConflictDetector / ConflictManager, and synchronizes memories
    to Knowledge Graph and Vector Memory.
    
    Guarantees:
    - Only passes appropriate interaction events to MemoryExtractor (filters out telemetry).
    - Never mutates original Memory or Event objects.
    - Preserves project isolation across all operations.
    - Idempotent and version-preserving: duplicate claims do not create redundant versions.
    - Contradictory information is captured as DISPUTED without silent resolution.
    - Downstream integration failures are surfaced as warnings/partial success without data loss.
    """

    def __init__(
        self,
        *,
        memory_extractor: Optional[MemoryExtractor] = None,
        memory_updater: Optional[MemoryUpdater] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        conflict_manager: Optional[ConflictManager] = None,
        graph_integrator: Optional[GraphIntegrator] = None,
        vector_manager: Optional[VectorMemoryManager] = None,
        ml_service: Optional[Any] = None,
    ) -> None:
        self.memory_extractor = memory_extractor or MemoryExtractor()
        self.memory_updater = memory_updater or MemoryUpdater()
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.conflict_manager = conflict_manager or ConflictManager(updater=self.memory_updater)
        self.graph_integrator = graph_integrator
        self.vector_manager = vector_manager
        self.ml_service = ml_service

        # In-memory index of current active memories per project
        # project_id -> {memory_id: Memory}
        self._project_memories: Dict[str, Dict[str, Memory]] = {}

    def register_memory(self, memory: Memory) -> None:
        """Register an existing memory into the handler's project store."""
        proj = self._project_memories.setdefault(memory.project_id, {})
        proj[memory.memory_id] = memory

    def register_memories(self, memories: Sequence[Memory]) -> None:
        """Register a collection of existing memories."""
        for m in memories:
            self.register_memory(m)

    def get_active_memories(self, project_id: str) -> List[Memory]:
        """Return all known active or disputed memories for a project."""
        proj = self._project_memories.get(project_id, {})
        return [
            m for m in proj.values()
            if m.status in (MemoryStatus.ACTIVE.value, MemoryStatus.DISPUTED.value, "ACTIVE", "DISPUTED")
        ]

    def filter_appropriate_events(self, events: Sequence[Event]) -> List[Event]:
        """Filter out internal telemetry (e.g. CONTEXT_DELIVERED, INTERACTION_COMPLETED)."""
        return [
            e for e in events
            if e.event_type in INFORMATIVE_EVENT_TYPES
        ]

    def extract_memories(
        self,
        events: Sequence[Event],
        *,
        project_id: str,
    ) -> Tuple[Memory, ...]:
        """
        Step 1: Extract domain memories from appropriate interaction events.
        """
        appropriate_events = self.filter_appropriate_events(events)
        if not appropriate_events:
            return ()

        try:
            extracted_memories = self.memory_extractor.extract_many(appropriate_events)
        except Exception as e:
            raise ExtractionStageError(f"Memory extraction failed on interaction events: {e}") from e

        # Strictly enforce project isolation
        project_mems = [m for m in extracted_memories if m.project_id == project_id]
        if self.ml_service is not None and getattr(self.ml_service.config, "enable_importance_ml", False):
            enriched = []
            for m in project_mems:
                pred = self.ml_service.predict_importance(m)
                if not pred.fallback_used and pred.confidence >= self.ml_service.config.min_confidence_threshold:
                    m_new = Memory(
                        memory_id=m.memory_id,
                        memory_type=m.memory_type,
                        content=m.content,
                        project_id=m.project_id,
                        created_at=m.created_at,
                        updated_at=m.updated_at,
                        source_event_ids=m.source_event_ids,
                        source_agent_ids=m.source_agent_ids,
                        importance=float(pred.target_value),
                        confidence=m.confidence,
                        status=m.status,
                        relationships=m.relationships,
                        metadata=m.metadata,
                        root_memory_id=m.root_memory_id,
                        parent_memory_id=m.parent_memory_id,
                        version=m.version,
                    )
                    enriched.append(m_new)
                else:
                    enriched.append(m)
            return tuple(enriched)
        return tuple(project_mems)

    def update_memories_and_conflicts(
        self,
        incoming_memories: Sequence[Memory],
        *,
        project_id: str,
    ) -> Tuple[Tuple[UpdateResult, ...], Tuple[MemoryConflict, ...], Tuple[str, ...]]:
        """
        Step 2: Update version chains and detect memory conflicts.
        """
        warnings: List[str] = []
        conflicts_detected: List[MemoryConflict] = []
        update_results: List[UpdateResult] = []

        for incoming in incoming_memories:
            existing = self.get_active_memories(project_id)
            conf_target_mem = None

            # Detect conflicts against existing memories
            try:
                for existing_mem in existing:
                    conf = self.conflict_detector.detect_pair(incoming, existing_mem)
                    if conf:
                        self.conflict_manager.track(conf)
                        conflicts_detected.append(conf)
                        conf_target_mem = existing_mem
                        break
            except Exception as e:
                warnings.append(f"Conflict detection warning for memory '{incoming.memory_id}': {e}")

            # Apply Memory Updater (use dispute strategy if conflict detected)
            try:
                if conf_target_mem is not None:
                    res = self.memory_updater.update(conf_target_mem, incoming, strategy="dispute")
                else:
                    batch_res = self.memory_updater.update_many([incoming], existing_memories=existing)
                    res = batch_res[0]
                update_results.append(res)
            except Exception as e:
                raise MemoryUpdateStageError(
                    f"Memory update failed for incoming memory '{incoming.memory_id}': {e}"
                ) from e

            # Update project memory cache with affected and current memories
            for affected in res.affected_memories:
                self.register_memory(affected)
            if res.current_memory:
                self.register_memory(res.current_memory)

        return tuple(update_results), tuple(conflicts_detected), tuple(warnings)

    def sync_graph(
        self,
        update_results: Sequence[UpdateResult],
        conflicts: Sequence[MemoryConflict] = (),
        *,
        raise_on_failure: bool = False,
    ) -> Tuple[int, Tuple[str, ...]]:
        """
        Step 3: Synchronize updated memories and conflicts to Knowledge Graph.
        """
        if self.graph_integrator is None:
            return 0, ()

        nodes_integrated = 0
        warnings: List[str] = []

        for res in update_results:
            memories_to_graph = list(res.affected_memories)
            if res.current_memory and res.current_memory not in memories_to_graph:
                memories_to_graph.append(res.current_memory)

            for mem in memories_to_graph:
                try:
                    self.graph_integrator.integrate_memory(mem)
                    nodes_integrated += 1
                except Exception as e:
                    msg = f"Graph integration failed for memory '{mem.memory_id}': {e}"
                    logger.warning(msg)
                    warnings.append(msg)
                    if raise_on_failure:
                        raise GraphIntegrationStageError(msg) from e

        for conf in conflicts:
            try:
                self.graph_integrator.integrate_conflict(conf)
            except Exception as e:
                msg = f"Graph conflict integration failed for '{conf.conflict_id}': {e}"
                logger.warning(msg)
                warnings.append(msg)
                if raise_on_failure:
                    raise GraphIntegrationStageError(msg) from e

        return nodes_integrated, tuple(warnings)

    def sync_vectors(
        self,
        update_results: Sequence[UpdateResult],
        *,
        raise_on_failure: bool = False,
    ) -> Tuple[int, Tuple[str, ...]]:
        """
        Step 4: Synchronize updated memories to Vector Memory.
        """
        if self.vector_manager is None:
            return 0, ()

        vectors_indexed = 0
        warnings: List[str] = []

        for res in update_results:
            memories_to_vector = list(res.affected_memories)
            if res.current_memory and res.current_memory not in memories_to_vector:
                memories_to_vector.append(res.current_memory)

            for mem in memories_to_vector:
                try:
                    self.vector_manager.index_memory(mem)
                    vectors_indexed += 1
                except Exception as e:
                    msg = f"Vector indexing failed for memory '{mem.memory_id}': {e}"
                    logger.warning(msg)
                    warnings.append(msg)
                    if raise_on_failure:
                        raise VectorIntegrationStageError(msg) from e

        return vectors_indexed, tuple(warnings)

    def process_interaction_events(
        self,
        events: Sequence[Event],
        *,
        project_id: str,
        raise_on_sync_failure: bool = False,
    ) -> LoopMemoryProcessingResult:
        """
        End-to-end memory handling pipeline coordinating extraction, updating,
        conflict tracking, and downstream graph and vector synchronization.
        """
        # Step 1: Extraction
        extracted_memories = self.extract_memories(events, project_id=project_id)

        # Step 2: Updating & Conflicts
        update_results, conflicts, update_warnings = self.update_memories_and_conflicts(
            extracted_memories, project_id=project_id
        )

        # Step 3: Graph Sync
        nodes_integrated, graph_warnings = self.sync_graph(
            update_results, conflicts, raise_on_failure=raise_on_sync_failure
        )

        # Step 4: Vector Sync
        vectors_indexed, vector_warnings = self.sync_vectors(
            update_results, raise_on_failure=raise_on_sync_failure
        )

        all_warnings = list(update_warnings) + list(graph_warnings) + list(vector_warnings)
        is_partial = len(all_warnings) > 0

        return LoopMemoryProcessingResult(
            extracted_memories=extracted_memories,
            update_results=update_results,
            conflicts_detected=conflicts,
            graph_nodes_integrated=nodes_integrated,
            vectors_indexed=vectors_indexed,
            warnings=tuple(all_warnings),
            is_partial_success=is_partial,
        )
