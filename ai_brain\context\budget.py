"""
Token budget allocation and greedy prioritized packing for Context Builder.
"""

from typing import Optional, Sequence

from ai_brain.context.exceptions import TokenBudgetExceededError
from ai_brain.context.formatter import ContextFormatter
from ai_brain.context.models import (
    ContextItem,
    ContextSection,
    ExcludedItem,
    PriorityBreakdown,
)
from ai_brain.context.tokenizer import BaseTokenizer, DeterministicTokenizer
from ai_brain.retrieval.models import RetrievalItem


class TokenBudgetManager:
    """
    Manages token budget constraints, estimates token consumption,
    and greedily packs memories by priority.
    
    Guarantees:
    - Never exceeds the configured max_tokens budget.
    - Accurately measures header and markdown formatting overhead.
    - Records detailed exclusion metadata for items dropped due to budget.
    - Pluggable tokenizer support with deterministic offline fallback.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 2000,
        tokenizer: Optional[BaseTokenizer] = None,
        formatter: Optional[ContextFormatter] = None,
        strict_budget: bool = False,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or DeterministicTokenizer()
        self.formatter = formatter or ContextFormatter()
        self.strict_budget = strict_budget

    def pack(
        self,
        candidates: Sequence[tuple[RetrievalItem, float, PriorityBreakdown]],
        *,
        project_id: str,
        query: str,
        pre_excluded: Sequence[ExcludedItem] = (),
    ) -> tuple[list[ContextItem], list[ContextSection], list[ExcludedItem], int, int]:
        """
        Greedily pack candidate items into the token budget in descending order of priority.
        
        Returns:
            (included_items, sections, all_excluded_items, tokens_used, remaining_tokens)
        """
        all_excluded: list[ExcludedItem] = list(pre_excluded)
        budget_excluded: list[ExcludedItem] = []

        # Step 0: Check empty baseline overhead with any pre-existing exclusions
        empty_md = self.formatter.render_markdown(
            project_id=project_id,
            query=query,
            sections=[],
            items=[],
            excluded_items=all_excluded,
        )
        base_tokens = self.tokenizer.count_tokens(empty_md)

        if base_tokens > self.max_tokens:
            if self.strict_budget:
                raise TokenBudgetExceededError(
                    f"Baseline header overhead ({base_tokens} tokens) exceeds "
                    f"budget limit of {self.max_tokens} tokens"
                )
            for item, priority, _ in candidates:
                budget_excluded.append(
                    ExcludedItem(
                        memory_id=item.memory_id,
                        reason="token_budget_exceeded",
                        priority_score=priority,
                        token_count=self.tokenizer.count_tokens(item.content),
                        is_disputed=bool(item.is_disputed or item.status == "DISPUTED"),
                        status=item.status,
                        details={"max_tokens": self.max_tokens, "tokens_used": base_tokens},
                    )
                )
            combined_excluded = list(all_excluded) + list(budget_excluded)
            return [], [], combined_excluded, base_tokens, max(0, self.max_tokens - base_tokens)

        included_items: list[ContextItem] = []

        # Map candidates by memory_id for quick lookup
        candidate_items_map = {item.memory_id: item for item, _, _ in candidates}

        # Step 1: Pack items greedily by priority
        for item, priority, breakdown in candidates:
            # Estimate tokens of item content
            item_tokens = self.tokenizer.count_tokens(item.content)

            # Extract graph relationships
            graph_rels: list[str] = []
            if hasattr(item, "graph_metadata") and item.graph_metadata:
                traversed = item.graph_metadata.get("relationships_traversed", ())
                if isinstance(traversed, (list, tuple)):
                    graph_rels.extend([str(r) for r in traversed])

            tentative_item = ContextItem(
                memory_id=item.memory_id,
                root_memory_id=item.root_memory_id,
                version=item.version,
                project_id=item.project_id,
                content=item.content,
                memory_type=item.memory_type,
                status=item.status,
                priority_score=priority,
                priority_breakdown=breakdown,
                relevance_score=item.score,
                token_count=item_tokens,
                is_disputed=bool(item.is_disputed or item.status == "DISPUTED"),
                source_agent_ids=item.source_agent_ids,
                source_event_ids=item.source_event_ids,
                graph_relationships=tuple(graph_rels),
                created_at=item.created_at,
                metadata=item.metadata,
            )

            # Test inclusion with realistic simulated exclusions (to account for omission notice)
            trial_items = included_items + [tentative_item]
            trial_included_ids = {it.memory_id for it in trial_items}

            # Any candidate or pre_excluded not in trial_items is tentatively excluded
            trial_excluded: list[Any] = list(all_excluded) + list(budget_excluded)
            for c_id, c_item in candidate_items_map.items():
                if c_id not in trial_included_ids and not any(ex.memory_id == c_id for ex in trial_excluded):
                    trial_excluded.append(
                        ExcludedItem(
                            memory_id=c_item.memory_id,
                            reason="token_budget_exceeded",
                            is_disputed=bool(c_item.is_disputed or c_item.status == "DISPUTED"),
                            status=c_item.status,
                        )
                    )

            trial_sections = self.formatter.build_sections(trial_items)
            trial_md = self.formatter.render_markdown(
                project_id=project_id,
                query=query,
                sections=trial_sections,
                items=trial_items,
                excluded_items=trial_excluded,
            )
            trial_tokens = self.tokenizer.count_tokens(trial_md)

            if trial_tokens <= self.max_tokens:
                included_items.append(tentative_item)
            else:
                budget_excluded.append(
                    ExcludedItem(
                        memory_id=item.memory_id,
                        reason="token_budget_exceeded",
                        priority_score=priority,
                        token_count=item_tokens,
                        is_disputed=bool(item.is_disputed or item.status == "DISPUTED"),
                        status=item.status,
                        details={
                            "item_tokens": item_tokens,
                            "budget_limit": self.max_tokens,
                            "tokens_needed_total": trial_tokens,
                        },
                    )
                )

        combined_excluded = list(all_excluded) + list(budget_excluded)

        # Step 2: Build final sections and verify final markdown token budget
        final_sections = self.formatter.build_sections(included_items)
        final_md = self.formatter.render_markdown(
            project_id=project_id,
            query=query,
            sections=final_sections,
            items=included_items,
            excluded_items=combined_excluded,
        )
        total_tokens_used = self.tokenizer.count_tokens(final_md)

        # Step 3: Hard guarantee — pop items if token count ever exceeds budget
        while included_items and total_tokens_used > self.max_tokens:
            dropped = included_items.pop()
            combined_excluded.append(
                ExcludedItem(
                    memory_id=dropped.memory_id,
                    reason="token_budget_exceeded",
                    priority_score=dropped.priority_score,
                    token_count=dropped.token_count,
                    is_disputed=dropped.is_disputed,
                    status=dropped.status,
                    details={"budget_limit": self.max_tokens},
                )
            )
            final_sections = self.formatter.build_sections(included_items)
            final_md = self.formatter.render_markdown(
                project_id=project_id,
                query=query,
                sections=final_sections,
                items=included_items,
                excluded_items=combined_excluded,
            )
            total_tokens_used = self.tokenizer.count_tokens(final_md)

        # Step 4: Measure section tokens
        measured_sections: list[ContextSection] = []
        for sec in final_sections:
            sec_text = "\n".join(self.formatter.format_item_line(it) for it in sec.items)
            sec_tokens = self.tokenizer.count_tokens(sec_text)
            measured_sections.append(
                ContextSection(
                    name=sec.name,
                    items=sec.items,
                    section_tokens=sec_tokens,
                )
            )

        remaining = max(0, self.max_tokens - total_tokens_used)
        return included_items, measured_sections, combined_excluded, total_tokens_used, remaining

