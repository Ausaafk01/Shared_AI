"""
Context compression and deduplication for Context Builder.
"""

import re
from typing import Sequence

from ai_brain.context.exceptions import CompressionError
from ai_brain.context.models import PriorityBreakdown
from ai_brain.retrieval.models import RetrievalItem


def _normalize_text(text: str) -> str:
    """Normalize whitespace and strip extraneous characters."""
    # Collapse multiple whitespaces/tabs
    s = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


class ContextCompressor:
    """
    Safely compresses and deduplicates context items without losing factual integrity.
    
    Guarantees:
    - Normalizes whitespace and removes redundant formatting noise.
    - Consolidates identical or compatible claims by merging provenance without duplicating content.
    - Strictly preserves contradictory and disputed claims as distinct entries.
    - Never hallucinates new facts or resolves disagreements silently.
    - Original RetrievalItem objects are never mutated.
    """

    def __init__(self, *, enable_claim_merging: bool = True) -> None:
        self.enable_claim_merging = enable_claim_merging

    def compress(
        self,
        ranked_items: Sequence[tuple[RetrievalItem, float, PriorityBreakdown]],
    ) -> list[tuple[RetrievalItem, float, PriorityBreakdown]]:
        """
        Compress and optionally merge compatible claims from ranked candidates.
        """
        if not ranked_items:
            return []

        # Step 1: Normalize content without mutating original objects
        cleaned_items: list[tuple[RetrievalItem, float, PriorityBreakdown]] = []
        for item, priority, breakdown in ranked_items:
            norm_content = _normalize_text(item.content)
            if norm_content != item.content:
                # Create a shallow updated copy with normalized content
                cleaned_item = RetrievalItem(
                    memory_id=item.memory_id,
                    root_memory_id=item.root_memory_id,
                    version=item.version,
                    project_id=item.project_id,
                    content=norm_content,
                    memory_type=item.memory_type,
                    status=item.status,
                    score=item.score,
                    score_breakdown=item.score_breakdown,
                    vector_score=item.vector_score,
                    graph_score=item.graph_score,
                    graph_metadata=item.graph_metadata,
                    sources=item.sources,
                    source_event_ids=item.source_event_ids,
                    source_agent_ids=item.source_agent_ids,
                    is_disputed=item.is_disputed,
                    created_at=item.created_at,
                    metadata=item.metadata,
                )
                cleaned_items.append((cleaned_item, priority, breakdown))
            else:
                cleaned_items.append((item, priority, breakdown))

        if not self.enable_claim_merging:
            return cleaned_items

        # Step 2: Merge compatible/identical claims while preserving contradictions
        # Disputed claims must never be merged
        result: list[tuple[RetrievalItem, float, PriorityBreakdown]] = []
        # Map: (memory_type, normalized_key) -> index in result
        seen_claims: dict[tuple[str, str], int] = {}

        for item, priority, breakdown in cleaned_items:
            # If item is disputed, do NOT merge it with anything
            if item.is_disputed or item.status == "DISPUTED":
                result.append((item, priority, breakdown))
                continue

            claim_key = (item.memory_type, item.content.lower().strip().rstrip("."))

            if claim_key in seen_claims:
                idx = seen_claims[claim_key]
                existing_item, existing_priority, existing_breakdown = result[idx]

                # Consolidate provenance and graph metadata
                consolidated_agents = tuple(
                    sorted(set(existing_item.source_agent_ids + item.source_agent_ids))
                )
                consolidated_events = tuple(
                    sorted(set(existing_item.source_event_ids + item.source_event_ids))
                )
                consolidated_sources = tuple(
                    sorted(set(existing_item.sources + item.sources))
                )

                # Merge graph metadata relationships if present
                merged_meta = dict(existing_item.metadata)
                for k, v in item.metadata.items():
                    if k not in merged_meta:
                        merged_meta[k] = v

                merged_item = RetrievalItem(
                    memory_id=existing_item.memory_id,
                    root_memory_id=existing_item.root_memory_id or item.root_memory_id,
                    version=max(existing_item.version, item.version),
                    project_id=existing_item.project_id,
                    content=existing_item.content,
                    memory_type=existing_item.memory_type,
                    status=existing_item.status,
                    score=max(existing_item.score, item.score),
                    score_breakdown=existing_item.score_breakdown,
                    vector_score=existing_item.vector_score or item.vector_score,
                    graph_score=existing_item.graph_score or item.graph_score,
                    graph_metadata=existing_item.graph_metadata or item.graph_metadata,
                    sources=consolidated_sources,
                    source_event_ids=consolidated_events,
                    source_agent_ids=consolidated_agents,
                    is_disputed=False,
                    created_at=existing_item.created_at,
                    metadata=merged_meta,
                )
                # Keep highest priority
                best_priority = max(existing_priority, priority)
                result[idx] = (merged_item, best_priority, existing_breakdown)
            else:
                seen_claims[claim_key] = len(result)
                result.append((item, priority, breakdown))

        return result

