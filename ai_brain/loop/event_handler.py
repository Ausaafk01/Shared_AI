"""
Event handling and persistence layer for the Brain Loop subsystem.
"""

from datetime import datetime
import logging
from typing import Sequence
import uuid

from ai_brain.events import (
    DuplicateEventError,
    Event,
    EventStore,
    create_event,
    register_event_type,
)
from ai_brain.loop.exceptions import EventPersistenceError
from ai_brain.loop.models import AgentInteractionInput, AgentResponseInput

logger = logging.getLogger(__name__)

# Register domain-specific event types for Brain Loop orchestration
try:
    register_event_type("INTERACTION_INPUT")
    register_event_type("CONTEXT_DELIVERED")
    register_event_type("INTERACTION_RESPONSE")
    register_event_type("INTERACTION_COMPLETED")
    register_event_type("PROCESSING_ERROR")
except Exception:
    pass


def _deterministic_event_id(interaction_id: str, stage: str) -> str:
    """Generate a stable UUID v5 based on interaction ID and processing stage."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"brain-loop:{interaction_id}:{stage}"))


class LoopEventHandler:
    """
    Manages event lifecycle, linkage, and idempotency for Brain Loop interactions.
    
    Guarantees:
    - Creates immutable events adhering to existing EventSystem models and rules.
    - Preserves project, agent, and session provenance.
    - Generates deterministic IDs for retry idempotency (prevents duplicate event injection).
    - Explicitly surfaces persistence failures.
    """

    def __init__(self, event_store: EventStore) -> None:
        if event_store is None:
            raise ValueError("An EventStore instance is required for LoopEventHandler")
        self.event_store = event_store

    def store_idempotent(self, event: Event) -> Event:
        """
        Store an event, gracefully returning the existing event if already stored.
        """
        try:
            self.event_store.store_event(event)
            return event
        except DuplicateEventError:
            existing = self.event_store.get_event_by_id(event.event_id)
            if existing is not None:
                return existing
            return event
        except Exception as e:
            raise EventPersistenceError(f"Failed to persist event '{event.event_id}': {e}") from e

    def record_input_event(self, interaction_input: AgentInteractionInput) -> Event:
        """
        Record the arrival of an agent interaction input.
        """
        event_id = _deterministic_event_id(interaction_input.interaction_id, "input")
        event = create_event(
            event_id=event_id,
            event_type="INTERACTION_INPUT",
            project_id=interaction_input.project_id,
            agent_id=interaction_input.agent_id,
            session_id=interaction_input.session_id,
            actor=f"agent:{interaction_input.agent_id}",
            input={"content": interaction_input.content, "query": interaction_input.query},
            metadata={
                "interaction_id": interaction_input.interaction_id,
                "stage": "input",
                "custom_metadata": dict(interaction_input.metadata),
            },
            timestamp=interaction_input.created_at,
        )
        return self.store_idempotent(event)

    def record_context_event(
        self,
        *,
        interaction_input: AgentInteractionInput,
        parent_event_id: str,
        tokens_used: int,
        token_budget: int,
        items_count: int,
        has_disputes: bool,
        has_omitted_disputes: bool,
    ) -> Event:
        """
        Record the successful assembly and delivery of context.
        """
        event_id = _deterministic_event_id(interaction_input.interaction_id, "context")
        event = create_event(
            event_id=event_id,
            event_type="CONTEXT_DELIVERED",
            project_id=interaction_input.project_id,
            agent_id=interaction_input.agent_id,
            session_id=interaction_input.session_id,
            actor="system:brain-loop",
            result={
                "tokens_used": tokens_used,
                "token_budget": token_budget,
                "items_count": items_count,
                "has_disputes": has_disputes,
                "has_omitted_disputes": has_omitted_disputes,
            },
            metadata={
                "interaction_id": interaction_input.interaction_id,
                "parent_event_id": parent_event_id,
                "stage": "context_prepared",
            },
        )
        return self.store_idempotent(event)

    def record_response_event(
        self,
        *,
        response_input: AgentResponseInput,
        parent_event_id: str,
    ) -> Event:
        """
        Record the receipt of an agent response.
        """
        event_id = _deterministic_event_id(response_input.interaction_id, "response")
        event = create_event(
            event_id=event_id,
            event_type="INTERACTION_RESPONSE",
            project_id=response_input.project_id,
            agent_id=response_input.agent_id,
            session_id=response_input.session_id,
            actor=f"agent:{response_input.agent_id}",
            output={"content": response_input.content},
            metadata={
                "interaction_id": response_input.interaction_id,
                "parent_event_id": parent_event_id,
                "stage": "response_received",
                "custom_metadata": dict(response_input.metadata),
            },
            timestamp=response_input.created_at,
        )
        return self.store_idempotent(event)

    def record_completion_event(
        self,
        *,
        interaction_id: str,
        project_id: str,
        agent_id: str,
        session_id: str,
        parent_event_id: str,
        memories_count: int,
        conflicts_count: int,
        is_partial: bool = False,
        warnings: Sequence[str] = (),
    ) -> Event:
        """
        Record the final completion of the interaction processing cycle.
        """
        event_id = _deterministic_event_id(interaction_id, "completed")
        event = create_event(
            event_id=event_id,
            event_type="INTERACTION_COMPLETED",
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
            actor="system:brain-loop",
            result={
                "memories_count": memories_count,
                "conflicts_count": conflicts_count,
                "is_partial": is_partial,
                "warnings": list(warnings),
            },
            metadata={
                "interaction_id": interaction_id,
                "parent_event_id": parent_event_id,
                "stage": "completed",
            },
        )
        return self.store_idempotent(event)

    def record_error_event(
        self,
        *,
        interaction_id: str,
        project_id: str,
        agent_id: str,
        session_id: str,
        stage: str,
        error_message: str,
    ) -> Event:
        """
        Record a stage failure event for auditability.
        """
        event_id = _deterministic_event_id(interaction_id, f"error:{stage}")
        event = create_event(
            event_id=event_id,
            event_type="PROCESSING_ERROR",
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
            actor="system:brain-loop",
            result={"stage": stage, "error": error_message},
            metadata={"interaction_id": interaction_id, "stage": stage},
        )
        return self.store_idempotent(event)
