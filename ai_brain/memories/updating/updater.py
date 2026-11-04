"""
Memory Updater for AI Brain.
Manages deterministic memory updates, version transitions, duplicate handling, and conflict preservation.
"""

from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import List, Optional, Sequence, Tuple
import uuid

from ai_brain.memories.exceptions import InvalidUpdateOperationError
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus
from .comparison import MemoryComparator, MemoryRelation
from .versioning import VersionTracker

logger = logging.getLogger(__name__)


class UpdateAction(StrEnum):
    """Categorical action taken by the updater."""
    SUPERSEDED = "SUPERSEDED"          # Replaced existing memory with a new version (Rule B)
    CREATED_NEW = "CREATED_NEW"        # Created as a separate distinct memory (Rule A)
    DUPLICATE_NOOP = "DUPLICATE_NOOP"  # Duplicate or unchanged; no version created (Rules C & D)
    DISPUTED = "DISPUTED"              # Contradictory claims; both marked DISPUTED (Rule E)


@dataclass(frozen=True)
class UpdateResult:
    """
    Structured outcome of a memory update operation.
    """
    action: UpdateAction
    created: bool
    superseded: bool
    duplicate: bool
    conflict: bool
    current_memory: Optional[Memory]
    previous_memory: Optional[Memory]
    affected_memories: Tuple[Memory, ...]
    message: str


class MemoryUpdater:
    """
    Orchestrates memory updating, version creation, and lifecycle status transitions.
    
    Guarantees:
    - Never mutates existing Memory objects in-place.
    - Creates explicit version chains with root_memory_id, parent_memory_id, and version number.
    - Preserves historical records when memories are superseded or disputed.
    - Suppresses duplicate or unchanged information without creating redundant versions.
    - Strictly isolates memories by project.
    """

    def __init__(
        self,
        tracker: Optional[VersionTracker] = None,
        comparator: Optional[MemoryComparator] = None,
    ) -> None:
        self.tracker = tracker or VersionTracker()
        self.comparator = comparator or MemoryComparator()

    def update(
        self,
        existing_memory: Memory,
        incoming_memory: Memory,
        *,
        strategy: Optional[str] = None,
    ) -> UpdateResult:
        """
        Update an existing memory with incoming memory information.
        
        Args:
            existing_memory: Currently recorded memory.
            incoming_memory: New or candidate memory.
            strategy: Optional explicit directive ('supersede', 'dispute', 'duplicate', 'new').
                      If omitted, the comparator determines the relation automatically.
                      
        Returns:
            UpdateResult detailing actions taken and affected memory versions.
            
        Raises:
            InvalidUpdateOperationError: If inputs are invalid or belong to different projects.
        """
        self._validate_memories(existing_memory, incoming_memory)

        # Register existing memory with tracker if not already tracked
        self.tracker.track(existing_memory)

        # Determine relation
        if strategy:
            strat_lower = strategy.strip().lower()
            if strat_lower == "supersede":
                relation = MemoryRelation.UPDATE
            elif strat_lower == "dispute":
                relation = MemoryRelation.CONTRADICTORY
            elif strat_lower == "duplicate":
                relation = MemoryRelation.DUPLICATE
            elif strat_lower == "new":
                relation = MemoryRelation.DISTINCT
            else:
                raise InvalidUpdateOperationError(f"Unknown update strategy '{strategy}'")
        else:
            comparison = self.comparator.compare(existing_memory, incoming_memory)
            relation = comparison.relation

        # Apply appropriate update rule
        if relation in (MemoryRelation.IDENTICAL, MemoryRelation.DUPLICATE):
            return self._handle_duplicate(existing_memory, incoming_memory)
        elif relation == MemoryRelation.UPDATE:
            return self._handle_update_supersede(existing_memory, incoming_memory)
        elif relation == MemoryRelation.CONTRADICTORY:
            return self._handle_contradictory_dispute(existing_memory, incoming_memory)
        else:
            return self._handle_distinct_new(existing_memory, incoming_memory)

    def _validate_memories(self, existing: Memory, incoming: Memory) -> None:
        """Ensure inputs are valid Memory instances from the same project."""
        if not isinstance(existing, Memory):
            raise InvalidUpdateOperationError(
                f"existing_memory must be a Memory instance, got {type(existing).__name__}"
            )
        if not isinstance(incoming, Memory):
            raise InvalidUpdateOperationError(
                f"incoming_memory must be a Memory instance, got {type(incoming).__name__}"
            )
        if existing.project_id != incoming.project_id:
            raise InvalidUpdateOperationError(
                f"Cannot update across different projects: '{existing.project_id}' vs '{incoming.project_id}'"
            )

    def _handle_duplicate(self, existing: Memory, incoming: Memory) -> UpdateResult:
        """Rule C & D: Handle duplicate or unchanged information."""
        return UpdateResult(
            action=UpdateAction.DUPLICATE_NOOP,
            created=False,
            superseded=False,
            duplicate=True,
            conflict=False,
            current_memory=existing,
            previous_memory=None,
            affected_memories=(existing,),
            message="Incoming information is identical or unchanged; no new version created.",
        )

    def _handle_update_supersede(self, existing: Memory, incoming: Memory) -> UpdateResult:
        """Rule B: Supersede existing memory with a new version."""
        new_memory_id = str(uuid.uuid4())

        # 1. Create superseded immutable snapshot of existing memory
        superseded_rels = dict(existing.relationships)
        superseded_rels["superseded_by"] = new_memory_id

        superseded_prev = existing.with_updates(
            status=MemoryStatus.SUPERSEDED.value,
            relationships=superseded_rels,
            memory_id=existing.memory_id,
            updated_at=max(existing.updated_at, incoming.updated_at),
        )

        # 2. Combine provenance
        combined_events = list(existing.source_event_ids)
        for eid in incoming.source_event_ids:
            if eid not in combined_events:
                combined_events.append(eid)

        combined_agents = list(existing.source_agent_ids)
        for aid in incoming.source_agent_ids:
            if aid not in combined_agents:
                combined_agents.append(aid)

        # 3. Create new active version linked to predecessor
        new_rels = dict(incoming.relationships)
        new_rels["supersedes"] = existing.memory_id

        new_version = incoming.with_updates(
            memory_id=new_memory_id,
            root_memory_id=existing.root_memory_id,
            status=MemoryStatus.ACTIVE.value,
            relationships=new_rels,
            parent_memory_id=existing.memory_id,
            version=existing.version + 1,
            source_event_ids=combined_events,
            source_agent_ids=combined_agents,
            updated_at=max(existing.updated_at, incoming.updated_at),
        )

        # 4. Register new version in tracker
        self.tracker.track(new_version)

        return UpdateResult(
            action=UpdateAction.SUPERSEDED,
            created=True,
            superseded=True,
            duplicate=False,
            conflict=False,
            current_memory=new_version,
            previous_memory=superseded_prev,
            affected_memories=(superseded_prev, new_version),
            message=f"Memory version {existing.version} ({existing.memory_id}) superseded by version {new_version.version} ({new_version.memory_id}).",
        )

    def _handle_contradictory_dispute(self, existing: Memory, incoming: Memory) -> UpdateResult:
        """Rule E: Handle contradictory information without silently choosing a side."""
        # 1. Immutable replacement for existing marked DISPUTED
        exist_rels = dict(existing.relationships)
        existing_conflicts = list(exist_rels.get("conflicts_with", []))
        if incoming.memory_id not in existing_conflicts:
            existing_conflicts.append(incoming.memory_id)
        exist_rels["conflicts_with"] = existing_conflicts

        disputed_existing = existing.with_updates(
            status=MemoryStatus.DISPUTED.value,
            confidence=min(existing.confidence, 0.5),
            relationships=exist_rels,
            memory_id=existing.memory_id,
        )

        # 2. Immutable incoming memory marked DISPUTED
        incom_rels = dict(incoming.relationships)
        incoming_conflicts = list(incom_rels.get("conflicts_with", []))
        if existing.memory_id not in incoming_conflicts:
            incoming_conflicts.append(existing.memory_id)
        incom_rels["conflicts_with"] = incoming_conflicts

        disputed_incoming = incoming.with_updates(
            status=MemoryStatus.DISPUTED.value,
            confidence=min(incoming.confidence, 0.5),
            relationships=incom_rels,
            memory_id=incoming.memory_id,
        )

        self.tracker.track(disputed_incoming)

        return UpdateResult(
            action=UpdateAction.DISPUTED,
            created=True,
            superseded=False,
            duplicate=False,
            conflict=True,
            current_memory=disputed_incoming,
            previous_memory=disputed_existing,
            affected_memories=(disputed_existing, disputed_incoming),
            message="Contradictory assertions detected; both memories preserved with DISPUTED status.",
        )

    def _handle_distinct_new(self, existing: Memory, incoming: Memory) -> UpdateResult:
        """Rule A: Separate distinct knowledge."""
        self.tracker.track(incoming)
        return UpdateResult(
            action=UpdateAction.CREATED_NEW,
            created=True,
            superseded=False,
            duplicate=False,
            conflict=False,
            current_memory=incoming,
            previous_memory=None,
            affected_memories=(existing, incoming),
            message="Incoming information represents distinct knowledge; created as separate memory.",
        )

    def update_many(
        self,
        incoming_memories: Sequence[Memory],
        existing_memories: Optional[Sequence[Memory]] = None,
    ) -> List[UpdateResult]:
        """
        Process a batch of incoming memories against known existing memories.
        
        Args:
            incoming_memories: Collection of incoming candidate memories.
            existing_memories: Optional known pool of existing memories.
            
        Returns:
            List of UpdateResult instances for each processed memory.
        """
        results: List[UpdateResult] = []
        if not incoming_memories:
            return results

        # Initialize existing pool
        known_pool = list(existing_memories or self.tracker.all_current())

        for incoming in incoming_memories:
            # Find best candidate in known pool by project and subject
            matched_existing: Optional[Memory] = None
            for existing in known_pool:
                if existing.project_id == incoming.project_id:
                    comp = self.comparator.compare(existing, incoming)
                    if comp.relation in (
                        MemoryRelation.IDENTICAL,
                        MemoryRelation.DUPLICATE,
                        MemoryRelation.UPDATE,
                        MemoryRelation.CONTRADICTORY,
                    ):
                        matched_existing = existing
                        break

            if matched_existing:
                res = self.update(matched_existing, incoming)
                results.append(res)
                # Update known pool
                if res.superseded and res.current_memory:
                    known_pool.remove(matched_existing)
                    known_pool.append(res.current_memory)
            else:
                # Rule A: New independent memory
                self.tracker.track(incoming)
                known_pool.append(incoming)
                results.append(
                    UpdateResult(
                        action=UpdateAction.CREATED_NEW,
                        created=True,
                        superseded=False,
                        duplicate=False,
                        conflict=False,
                        current_memory=incoming,
                        previous_memory=None,
                        affected_memories=(incoming,),
                        message="Distinct memory created and tracked.",
                    )
                )

        return results

    def get_history(self, root_memory_id: str) -> List[Memory]:
        """Retrieve the version history for a memory by its root identity."""
        history = self.tracker.get_history(root_memory_id)
        return history.get_history() if history else []

    def get_current(self, root_memory_id: str) -> Optional[Memory]:
        """Retrieve the current active version for a root memory identity."""
        return self.tracker.get_current(root_memory_id)

    def get_all_current(self) -> List[Memory]:
        """Retrieve all currently active memories across all tracked chains."""
        return self.tracker.all_current()
