"""
Simulated Review Agent specialized in reviewing outputs, identifying errors, and detecting contradictions.
"""

from typing import Any, Callable, Mapping, Optional

from ai_brain.agents.base import Agent
from ai_brain.agents.models import (
    AgentCapability,
    AgentExecutionRequest,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.loop.models import InteractionContextResult


class ReviewAgent(Agent):
    """
    Simulated agent specializing in code review, architectural critique,
    and contradiction detection.
    
    Inspects Brain context to evaluate whether proposed implementations align
    with established project facts, decisions, and tasks. Flags contradictions
    when deviations are detected.
    """

    def __init__(
        self,
        agent_id: str = "agent-reviewer",
        name: str = "Review Agent",
        *,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        response_generator: Optional[Callable[[InteractionContextResult, AgentExecutionRequest], str]] = None,
    ) -> None:
        caps = capabilities or (
            AgentCapability.CODE_REVIEW.value,
            AgentCapability.CONTRADICTION_DETECTION.value,
            AgentCapability.REASONING.value,
        )
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="simulated",
            agent_type=AgentType.REVIEW,
            capabilities=caps,
            configuration=configuration or {},
            status=AgentStatus.ACTIVE,
        )
        super().__init__(record)
        self._custom_generator = response_generator

    def generate_response(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> str:
        """Review implementation proposals against established Brain context."""
        if self._custom_generator is not None:
            return self._custom_generator(context_result, request)

        content_lower = request.content.lower()
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""
        context_lower = context_md.lower()

        # Check for contradictions against established facts
        if "mysql" in content_lower and "postgresql" in context_lower:
            return (
                "Review critique: Contradiction detected! The proposal specifies MySQL, but established project "
                "architecture specifies PostgreSQL 16. We decided to reject MySQL and enforce PostgreSQL consistency."
            )
        elif "cookie" in content_lower and "jwt" in context_lower:
            return (
                "Review critique: Contradiction detected! The proposal specifies session cookies, but established "
                "architecture standardizes on OAuth 2.0 with JWT tokens."
            )
        elif "memcached" in content_lower and "redis" in context_lower:
            return (
                "Review critique: Contradiction detected! The proposal specifies Memcached, but established "
                "context specifies Redis for distributed caching."
            )
        elif "contradict" in content_lower:
            # Explicit contradiction requested for conflict testing
            return (
                "Review critique: Contradiction detected! The proposed changes conflict with established decisions. "
                "The project uses MySQL as its database."
            )
        else:
            # Positive review verification
            has_facts = bool(context_result.context_package and len(context_result.context_package.items) > 0)
            grounding = "validated against retrieved Brain knowledge" if has_facts else "preliminary approval"
            return (
                f"Review verdict ({grounding}): The implementation proposal for '{request.content}' adheres to "
                f"established project standards. No conflicting architectural patterns detected."
            )

