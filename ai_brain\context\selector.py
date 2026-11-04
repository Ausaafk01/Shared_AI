"""
Memory selection and filtering for Context Builder.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from ai_brain.context.exceptions import ContextProjectIsolationError, SelectionError
from ai_brain.context.models import ExcludedItem
from ai_brain.retrieval.models import RetrievalItem, RetrievalResponse


@dataclass(frozen=True, kw_only=True)
class SelectionCriteria:
    """
    Configuration criteria for selecting memories into context.
    """
    min_relevance_score: float = 0.0
    allowed_statuses: tuple[str, ...] = ("ACTIVE", "DISPUTED")
    include_superseded: bool = False
    include_archived: bool = False
    deduplicate_versions: bool = True
    allowed_memory_types: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        statuses = set(s.strip().upper() for s in self.allowed_statuses if s and isinstance(s, str))
        if self.include_superseded:
            statuses.add("SUPERSEDED")
        if self.include_archived:
            statuses.add("ARCHIVED")
        object.__setattr__(self, "allowed_statuses", tuple(sorted(statuses)) or ("ACTIVE", "DISPUTED"))
        types = tuple(t.strip().upper() for t in self.allowed_memory_types if t and isinstance(t, str))
        object.__setattr__(self, "allowed_memory_types", types)


class MemorySelector:
    """
    Enforces project isolation, deduplication across memory versions,
    status filtering, and relevance gating.
    """

    def __init__(self, criteria: Optional[SelectionCriteria] = None) -> None:
        self.criteria = criteria or SelectionCriteria()

    def select(
        self,
        retrieval_input: Union[RetrievalResponse, Sequence[RetrievalItem]],
        *,
        project_id: str,
    ) -> tuple[list[RetrievalItem], list[ExcludedItem]]:
        """
        Select relevant memories and return both selected candidates and excluded items.
        
        Guarantees:
        - Strict project isolation enforcement.
        - Excludes superseded and archived items unless explicitly permitted.
        - Deduplicates multiple versions of the same root memory.
        - Preserves disputed items as explicitly disputed.
        """
        if not project_id or not isinstance(project_id, str):
            raise SelectionError("A non-empty project_id is required for memory selection")

        if isinstance(retrieval_input, RetrievalResponse):
            items: Sequence[RetrievalItem] = retrieval_input.results
            # Verify response project_id matches if given
            if retrieval_input.project_id and retrieval_input.project_id != project_id:
                raise ContextProjectIsolationError(
                    f"Retrieval response project '{retrieval_input.project_id}' "
                    f"does not match requested project '{project_id}'"
                )
        else:
            items = retrieval_input

        selected: list[RetrievalItem] = []
        excluded: list[ExcludedItem] = []

        # Step 1: Project isolation check and basic filtering
        admissible_items: list[RetrievalItem] = []
        for item in items:
            # Check project isolation
            if item.project_id != project_id:
                raise ContextProjectIsolationError(
                    f"Cross-project memory detected: memory '{item.memory_id}' belongs to "
                    f"project '{item.project_id}', but selection requested '{project_id}'"
                )

            # Check status
            status_norm = item.status.strip().upper()
            item_is_disputed = bool(item.is_disputed or status_norm == "DISPUTED")

            if status_norm == "SUPERSEDED" and not self.criteria.include_superseded:
                excluded.append(
                    ExcludedItem(
                        memory_id=item.memory_id,
                        reason="excluded_status",
                        priority_score=item.score,
                        is_disputed=item_is_disputed,
                        status=item.status,
                        details={"status": item.status, "reason": "superseded_excluded"},
                    )
                )
                continue

            if status_norm == "ARCHIVED" and not self.criteria.include_archived:
                excluded.append(
                    ExcludedItem(
                        memory_id=item.memory_id,
                        reason="excluded_status",
                        priority_score=item.score,
                        is_disputed=item_is_disputed,
                        status=item.status,
                        details={"status": item.status, "reason": "archived_excluded"},
                    )
                )
                continue

            if status_norm not in self.criteria.allowed_statuses:
                excluded.append(
                    ExcludedItem(
                        memory_id=item.memory_id,
                        reason="excluded_status",
                        priority_score=item.score,
                        is_disputed=item_is_disputed,
                        status=item.status,
                        details={"status": item.status, "allowed": list(self.criteria.allowed_statuses)},
                    )
                )
                continue

            # Check memory type if restricted
            if self.criteria.allowed_memory_types:
                mtype = item.memory_type.strip().upper()
                if mtype not in self.criteria.allowed_memory_types:
                    excluded.append(
                        ExcludedItem(
                            memory_id=item.memory_id,
                            reason="excluded_type",
                            priority_score=item.score,
                            is_disputed=item_is_disputed,
                            status=item.status,
                            details={"memory_type": item.memory_type},
                        )
                    )
                    continue

            # Check relevance score gate
            if item.score < self.criteria.min_relevance_score:
                excluded.append(
                    ExcludedItem(
                        memory_id=item.memory_id,
                        reason="low_relevance",
                        priority_score=item.score,
                        is_disputed=item_is_disputed,
                        status=item.status,
                        details={"score": item.score, "threshold": self.criteria.min_relevance_score},
                    )
                )
                continue

            admissible_items.append(item)

        # Step 2: Version deduplication across shared root_memory_id
        if self.criteria.deduplicate_versions:
            root_groups: dict[str, list[RetrievalItem]] = {}
            for item in admissible_items:
                root_id = item.root_memory_id or item.memory_id
                root_groups.setdefault(root_id, []).append(item)

            for root_id, group in root_groups.items():
                if len(group) == 1:
                    selected.append(group[0])
                else:
                    # Sort candidates: active first, higher version first, higher score first
                    sorted_group = sorted(
                        group,
                        key=lambda x: (
                            1 if x.status == "ACTIVE" else (0 if x.status == "DISPUTED" else -1),
                            x.version,
                            x.score,
                        ),
                        reverse=True,
                    )
                    winner = sorted_group[0]
                    selected.append(winner)
                    for loser in sorted_group[1:]:
                        loser_status_norm = loser.status.strip().upper()
                        excluded.append(
                            ExcludedItem(
                                memory_id=loser.memory_id,
                                reason="duplicate_version",
                                priority_score=loser.score,
                                is_disputed=bool(loser.is_disputed or loser_status_norm == "DISPUTED"),
                                status=loser.status,
                                details={
                                    "root_memory_id": root_id,
                                    "version": loser.version,
                                    "superseded_by": winner.memory_id,
                                    "winning_version": winner.version,
                                },
                            )
                        )
        else:
            selected.extend(admissible_items)

        return selected, excluded
