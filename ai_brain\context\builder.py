"""
Context Builder subsystem facade for AI Brain.
"""

from typing import Any, Mapping, Optional, Sequence, Union

from ai_brain.context.budget import TokenBudgetManager
from ai_brain.context.compressor import ContextCompressor
from ai_brain.context.exceptions import ContextBuilderError, SelectionError
from ai_brain.context.formatter import ContextFormatter
from ai_brain.context.models import ContextPackage, ExcludedItem
from ai_brain.context.prioritizer import MemoryPrioritizer
from ai_brain.context.selector import MemorySelector, SelectionCriteria
from ai_brain.context.tokenizer import BaseTokenizer, DeterministicTokenizer
from ai_brain.retrieval.models import RetrievalItem, RetrievalResponse


class ContextBuilder:
    """
    Coordinates selection, prioritization, semantic compression,
    budget enforcement, and agent-ready prompt formatting.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[BaseTokenizer] = None,
        selector: Optional[MemorySelector] = None,
        prioritizer: Optional[MemoryPrioritizer] = None,
        compressor: Optional[ContextCompressor] = None,
        formatter: Optional[ContextFormatter] = None,
        default_max_tokens: int = 2000,
        include_provenance_section: bool = True,
    ) -> None:
        if default_max_tokens <= 0:
            raise ContextBuilderError("default_max_tokens must be positive")
        self.default_max_tokens = default_max_tokens
        self.tokenizer = tokenizer or DeterministicTokenizer()
        self.selector = selector or MemorySelector()
        self.prioritizer = prioritizer or MemoryPrioritizer()
        self.compressor = compressor or ContextCompressor()
        self.formatter = formatter or ContextFormatter(
            include_provenance_section=include_provenance_section
        )

    def build_context(
        self,
        retrieval_input: Union[RetrievalResponse, Sequence[RetrievalItem]],
        *,
        project_id: str,
        query: Optional[str] = None,
        max_tokens: Optional[int] = None,
        min_relevance: Optional[float] = None,
        include_disputed: bool = True,
        include_superseded: bool = False,
        include_archived: bool = False,
        deduplicate_versions: bool = True,
        strict_budget: bool = False,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ContextPackage:
        """
        Transform retrieved items into a compact, prioritized, token-bounded context package.
        
        Args:
            retrieval_input: RetrievalResponse or sequence of RetrievalItem objects.
            project_id: Project identifier for isolation.
            query: Query string or focus topic.
            max_tokens: Maximum token budget (defaults to self.default_max_tokens).
            min_relevance: Minimum relevance score required for candidate selection.
            include_disputed: Whether to admit disputed memories into context.
            include_superseded: Whether to allow superseded memory versions.
            include_archived: Whether to allow archived memories.
            deduplicate_versions: Whether to retain only the latest/active version of each entity.
            strict_budget: Raise TokenBudgetExceededError if budget cannot fit minimum context.
            extra_metadata: Optional extra metadata dictionary.
        """
        if not project_id or not isinstance(project_id, str):
            raise SelectionError("A non-empty project_id is required to build context")

        budget_limit = max_tokens if max_tokens is not None else self.default_max_tokens
        if budget_limit <= 0:
            raise ContextBuilderError(f"Token budget limit must be positive, got {budget_limit}")

        # Resolve query string
        resolved_query = query
        if resolved_query is None and isinstance(retrieval_input, RetrievalResponse):
            resolved_query = retrieval_input.query
        resolved_query = resolved_query or ""

        # Step 1: Memory Selection and Filtering
        allowed_statuses = ["ACTIVE"]
        if include_disputed:
            allowed_statuses.append("DISPUTED")
        if include_archived:
            allowed_statuses.append("ARCHIVED")
        if include_superseded:
            allowed_statuses.append("SUPERSEDED")

        criteria = SelectionCriteria(
            min_relevance_score=min_relevance if min_relevance is not None else self.selector.criteria.min_relevance_score,
            allowed_statuses=tuple(allowed_statuses),
            include_superseded=include_superseded,
            include_archived=include_archived,
            deduplicate_versions=deduplicate_versions,
            allowed_memory_types=self.selector.criteria.allowed_memory_types,
        )
        selector = MemorySelector(criteria=criteria)
        selected_candidates, selection_excluded = selector.select(
            retrieval_input,
            project_id=project_id,
        )

        # Step 2: Prioritization
        ranked_candidates = self.prioritizer.prioritize(selected_candidates)

        # Step 3: Semantic Compression and Claim Consolidation
        compressed_candidates = self.compressor.compress(ranked_candidates)

        # Step 4: Token Budget Management and Packing
        budget_manager = TokenBudgetManager(
            max_tokens=budget_limit,
            tokenizer=self.tokenizer,
            formatter=self.formatter,
            strict_budget=strict_budget,
        )
        (
            included_items,
            sections,
            all_excluded,
            tokens_used,
            remaining_tokens,
        ) = budget_manager.pack(
            compressed_candidates,
            project_id=project_id,
            query=resolved_query,
            pre_excluded=selection_excluded,
        )

        # Step 5: Render agent representations
        rendered_md = self.formatter.render_markdown(
            project_id=project_id,
            query=resolved_query,
            sections=sections,
            items=included_items,
            excluded_items=all_excluded,
        )
        rendered_txt = self.formatter.render_text(
            project_id=project_id,
            query=resolved_query,
            sections=sections,
            items=included_items,
            excluded_items=all_excluded,
        )

        # Assemble package metadata
        has_omitted_disp = any(ex.is_disputed or ex.status == "DISPUTED" for ex in all_excluded)
        omitted_disp_count = sum(1 for ex in all_excluded if ex.is_disputed or ex.status == "DISPUTED")

        meta = {
            "retrieval_total_candidates": len(retrieval_input.results) if isinstance(retrieval_input, RetrievalResponse) else len(retrieval_input),
            "selection_retained_candidates": len(selected_candidates),
            "compression_retained_candidates": len(compressed_candidates),
            "budget_retained_items": len(included_items),
            "total_excluded_items": len(all_excluded),
            "has_disputes": any(it.is_disputed for it in included_items),
            "has_omitted_disputes": has_omitted_disp,
            "omitted_disputes_count": omitted_disp_count,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return ContextPackage(
            project_id=project_id,
            query=resolved_query,
            sections=tuple(sections),
            items=tuple(included_items),
            token_budget=budget_limit,
            tokens_used=tokens_used,
            remaining_tokens=remaining_tokens,
            excluded_items=tuple(all_excluded),
            metadata=meta,
            rendered_markdown=rendered_md,
            rendered_text=rendered_txt,
        )

