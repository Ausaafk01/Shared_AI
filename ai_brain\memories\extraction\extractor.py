"""
Orchestrator for AI Brain Memory Extraction.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ai_brain.events.models import Event
from ai_brain.memories.exceptions import (
    ExtractionRuleError,
    InvalidEventError,
    MemoryValidationError,
)
from ai_brain.memories.models import Memory
from ai_brain.memories.types import MemoryStatus, MemoryType
from .helpers import CandidateMemory
from .rules import ExtractionRule, get_default_rules

logger = logging.getLogger(__name__)


def _detect_conflict_key(content: str, memory_type: str | MemoryType) -> Optional[str]:
    """
    Generate a conflict key for subject-based statements to detect contradictory information.
    For example, 'The project uses PostgreSQL as its database' and 'The project uses MySQL as its database'
    both target the key 'project:database'.
    """
    lower = content.lower()
    if "database" in lower or "postgres" in lower or "mysql" in lower or "mongodb" in lower:
        return "tech:database"
    if "authentication" in lower or "jwt" in lower or "cookie" in lower:
        return "tech:authentication"
    if "cache" in lower or "redis" in lower or "memcached" in lower:
        return "tech:caching"
    return None


class MemoryExtractor:
    """
    Extracts structured, immutable Memory objects from Events.
    
    Responsibilities:
    - Analyzes single events and multi-event sequences.
    - Applies deterministic classification rules.
    - Preserves provenance (source_event_ids, source_agent_ids).
    - Prevents intra-batch duplicate candidates.
    - Detects conflicting assertions and marks them DISPUTED.
    - Keeps multiple projects strictly separated.
    - Robust error handling with error collection.
    """

    def __init__(
        self,
        rules: Optional[Sequence[ExtractionRule]] = None,
        *,
        strict: bool = False,
    ) -> None:
        """
        Initialize the MemoryExtractor.
        
        Args:
            rules: Custom sequence of extraction rules. Defaults to standard rules.
            strict: If True, raise exceptions on invalid events or rule errors.
                    If False (default), log and record errors, skipping bad events.
        """
        self.rules: List[ExtractionRule] = list(rules if rules is not None else get_default_rules())
        self.strict = strict
        self.last_errors: List[Exception] = []

    def extract(self, event: Event) -> List[Memory]:
        """
        Extract memories from a single Event.
        
        Args:
            event: An immutable Event instance.
            
        Returns:
            List of generated Memory objects (zero or more).
            
        Raises:
            InvalidEventError: If the event is not a valid Event instance.
            ExtractionRuleError: If a rule fails in strict mode.
        """
        self.last_errors = []
        self._validate_event(event)

        candidates: List[CandidateMemory] = []
        for rule in self.rules:
            try:
                if rule.can_extract(event):
                    extracted = rule.extract_single(event)
                    candidates.extend(extracted)
            except Exception as e:
                err = ExtractionRuleError(f"Rule '{rule.rule_name}' failed on event '{event.event_id}': {e}")
                self.last_errors.append(err)
                if self.strict:
                    raise err from e
                logger.warning(str(err))

        deduped = self._deduplicate_candidates(candidates)
        return self._build_memories(deduped)

    def extract_many(self, events: Sequence[Event]) -> List[Memory]:
        """
        Extract memories from a collection of Events.
        
        Supports multi-event correlation within the same project and session,
        merges duplicates within the batch, and flags conflicting assertions.
        
        Args:
            events: A sequence of Event instances.
            
        Returns:
            List of generated Memory objects.
            
        Raises:
            InvalidEventError: In strict mode if an event is invalid.
        """
        self.last_errors = []
        if not events:
            return []

        # 1. Validate and group events by project_id
        valid_events_by_project: Dict[str, List[Event]] = defaultdict(list)
        for idx, event in enumerate(events):
            try:
                self._validate_event(event)
                valid_events_by_project[event.project_id].append(event)
            except Exception as e:
                self.last_errors.append(e)
                if self.strict:
                    raise
                logger.warning("Skipping invalid event at index %d: %s", idx, e)

        all_memories: List[Memory] = []

        # 2. Process each project independently (projects are strictly separated)
        for project_id, proj_events in valid_events_by_project.items():
            project_candidates: List[CandidateMemory] = []

            # Group by session for multi-event sequence extraction
            events_by_session: Dict[str, List[Event]] = defaultdict(list)
            for ev in proj_events:
                events_by_session[ev.session_id].append(ev)

            # Multi-event rules per session
            for session_id, sess_events in events_by_session.items():
                if len(sess_events) >= 2:
                    for rule in self.rules:
                        try:
                            if rule.can_extract_multi(sess_events):
                                multi_extracted = rule.extract_multi(sess_events)
                                project_candidates.extend(multi_extracted)
                        except Exception as e:
                            err = ExtractionRuleError(f"Multi-event rule '{rule.rule_name}' failed: {e}")
                            self.last_errors.append(err)
                            if self.strict:
                                raise err from e
                            logger.warning(str(err))

            # Single-event rules for all events in the project
            for event in proj_events:
                for rule in self.rules:
                    try:
                        if rule.can_extract(event):
                            single_extracted = rule.extract_single(event)
                            project_candidates.extend(single_extracted)
                    except Exception as e:
                        err = ExtractionRuleError(f"Rule '{rule.rule_name}' failed on event '{event.event_id}': {e}")
                        self.last_errors.append(err)
                        if self.strict:
                            raise err from e
                        logger.warning(str(err))

            # 3. Deduplicate candidates in the same batch
            deduped = self._deduplicate_candidates(project_candidates)

            # 4. Detect and flag conflicting statements
            resolved = self._handle_conflicts(deduped)

            # 5. Build immutable Memory objects
            all_memories.extend(self._build_memories(resolved))

        return all_memories

    def _validate_event(self, event: Any) -> None:
        """Ensure the input object is a valid, well-formed Event."""
        if not isinstance(event, Event):
            raise InvalidEventError(
                f"Expected Event instance, got {type(event).__name__ if event is not None else 'None'}"
            )
        if not event.project_id or not isinstance(event.project_id, str) or not event.project_id.strip():
            raise InvalidEventError(f"Event '{event.event_id}' has invalid or empty project_id")

    def _deduplicate_candidates(
        self, candidates: List[CandidateMemory]
    ) -> List[CandidateMemory]:
        """
        Merge candidate memories that represent identical information within the same extraction.
        Combines source_event_ids, source_agent_ids, and merges timestamps.
        """
        if not candidates:
            return []

        merged_map: Dict[Tuple[str, str, str], CandidateMemory] = {}

        for cand in candidates:
            # Normalized key: (project_id, memory_type, lowercase stripped content)
            type_str = cand.memory_type.value if hasattr(cand.memory_type, "value") else str(cand.memory_type)
            key = (cand.project_id, type_str.upper(), cand.content.strip().lower())

            if key not in merged_map:
                merged_map[key] = cand
            else:
                existing = merged_map[key]
                # Merge provenance
                for eid in cand.source_event_ids:
                    if eid not in existing.source_event_ids:
                        existing.source_event_ids.append(eid)
                for aid in cand.source_agent_ids:
                    if aid not in existing.source_agent_ids:
                        existing.source_agent_ids.append(aid)

                # Merge timestamps
                if cand.created_at and existing.created_at:
                    existing.created_at = min(existing.created_at, cand.created_at)
                if cand.updated_at and existing.updated_at:
                    existing.updated_at = max(existing.updated_at, cand.updated_at)

                # Higher importance and confidence
                existing.importance = max(existing.importance, cand.importance)
                existing.confidence = max(existing.confidence, cand.confidence)

                # Merge metadata
                existing.metadata.update(cand.metadata)

        return list(merged_map.values())

    def _handle_conflicts(
        self, candidates: List[CandidateMemory]
    ) -> List[CandidateMemory]:
        """
        Identify conflicting statements within the same batch.
        Marks conflicting memories as DISPUTED without silently discarding any provenance.
        """
        if len(candidates) < 2:
            return candidates

        # Group by detected conflict subject
        subject_groups: Dict[str, List[CandidateMemory]] = defaultdict(list)
        for cand in candidates:
            key = _detect_conflict_key(cand.content, cand.memory_type)
            if key:
                subject_groups[key].append(cand)

        for key, group in subject_groups.items():
            if len(group) >= 2:
                # Check if contents are different (i.e. contradictory assertions)
                unique_contents = {c.content.strip().lower() for c in group}
                if len(unique_contents) > 1:
                    # Contradiction detected: mark all candidates in group as DISPUTED
                    for c in group:
                        c.status = MemoryStatus.DISPUTED.value
                        c.confidence = min(c.confidence, 0.5)
                        other_contents = [
                            o.content for o in group if o.content.strip().lower() != c.content.strip().lower()
                        ]
                        c.relationships["conflicts_with"] = other_contents
                        c.metadata["disputed_reason"] = f"Contradictory information detected on '{key}'"

        return candidates

    def _build_memories(self, candidates: List[CandidateMemory]) -> List[Memory]:
        """Build candidate memories into immutable validated Memory instances."""
        memories: List[Memory] = []
        for cand in candidates:
            try:
                memories.append(cand.build())
            except MemoryValidationError as e:
                self.last_errors.append(e)
                if self.strict:
                    raise
                logger.warning("Failed to build memory candidate: %s", e)
        return memories

