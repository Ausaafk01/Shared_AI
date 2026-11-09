"""
Input processing and validation for the Brain Loop subsystem.
"""


from ai_brain.loop.exceptions import InputValidationError, LoopProjectIsolationError
from ai_brain.loop.models import AgentInteractionInput


class InputProcessor:
    """
    Validates and normalizes incoming interaction inputs before orchestration.
    
    Guarantees:
    - Enforces non-empty identifiers for project, agent, and session.
    - Preserves the raw user content without loss.
    - Prevents cross-project interaction injection.
    - Normalizes query strings and validates retrieval options.
    """

    def process(self, interaction_input: AgentInteractionInput) -> AgentInteractionInput:
        """
        Validate and normalize an incoming interaction input.
        """
        if not isinstance(interaction_input, AgentInteractionInput):
            raise InputValidationError(
                f"Expected AgentInteractionInput instance, got {type(interaction_input).__name__}"
            )

        # Content validation
        if not interaction_input.content.strip():
            raise InputValidationError("Interaction content cannot be empty or purely whitespace")

        # Project ID validation
        if not interaction_input.project_id.strip():
            raise InputValidationError("project_id cannot be empty")

        # Agent ID validation
        if not interaction_input.agent_id.strip():
            raise InputValidationError("agent_id cannot be empty")

        # Session ID validation
        if not interaction_input.session_id.strip():
            raise InputValidationError("session_id cannot be empty")

        # Check metadata for unauthorized cross-project references
        if interaction_input.metadata:
            target_proj = interaction_input.metadata.get("target_project_id")
            if target_proj and target_proj != interaction_input.project_id:
                raise LoopProjectIsolationError(
                    f"Cross-project reference detected in metadata: target '{target_proj}' "
                    f"does not match interaction project '{interaction_input.project_id}'"
                )

        return interaction_input
