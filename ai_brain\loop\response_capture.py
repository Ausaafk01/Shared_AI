"""
Response capture and validation component for Brain Loop.
"""

from typing import Optional

from ai_brain.loop.exceptions import (
    InteractionNotFoundError,
    InteractionStateError,
    LoopProjectIsolationError,
    ResponseValidationError,
)
from ai_brain.loop.models import (
    AgentResponseInput,
    InteractionRecord,
    InteractionStatus,
)


class ResponseCapture:
    """
    Captures and validates agent responses against their originating interaction.
    
    Guarantees:
    - Verifies originating interaction exists and was prepared.
    - Prevents cross-project, cross-agent, or cross-session response spoofing.
    - Preserves response content and metadata faithfully.
    """

    def validate_response(
        self,
        response_input: AgentResponseInput,
        record: Optional[InteractionRecord],
    ) -> AgentResponseInput:
        """
        Validate that the response matches the originating interaction record.
        """
        if not isinstance(response_input, AgentResponseInput):
            raise ResponseValidationError(
                f"Expected AgentResponseInput instance, got {type(response_input).__name__}"
            )

        if record is None:
            raise InteractionNotFoundError(
                f"No interaction record found matching interaction_id '{response_input.interaction_id}'"
            )

        # Verify state
        if record.status not in (
            InteractionStatus.CONTEXT_PREPARED,
            InteractionStatus.RESPONSE_RECEIVED,
            InteractionStatus.PARTIAL_SUCCESS,
            InteractionStatus.FAILED,
        ):
            if record.status == InteractionStatus.COMPLETED:
                raise InteractionStateError(
                    f"Interaction '{record.interaction_id}' has already been completed"
                )
            raise InteractionStateError(
                f"Cannot process response for interaction in status '{record.status.value}'"
            )

        # Enforce project match
        if response_input.project_id != record.project_id:
            raise LoopProjectIsolationError(
                f"Response project_id '{response_input.project_id}' does not match "
                f"originating interaction project_id '{record.project_id}'"
            )

        # Enforce agent match
        if response_input.agent_id != record.agent_id:
            raise ResponseValidationError(
                f"Response agent_id '{response_input.agent_id}' does not match "
                f"originating interaction agent_id '{record.agent_id}'"
            )

        # Enforce session match
        if response_input.session_id != record.session_id:
            raise ResponseValidationError(
                f"Response session_id '{response_input.session_id}' does not match "
                f"originating interaction session_id '{record.session_id}'"
            )

        # Content validation
        if not response_input.content.strip():
            raise ResponseValidationError("Response content cannot be empty or purely whitespace")

        return response_input
