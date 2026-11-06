"""
Simulated Coding Agent specialized in software development and technical decisions.
"""

from typing import Any, Callable, Iterator, Mapping, Optional

from ai_brain.agents.base import Agent
from ai_brain.agents.models import (
    AgentCapability,
    AgentExecutionRequest,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.loop.models import InteractionContextResult


class CodingAgent(Agent):
    """
    Simulated agent specializing in technical architecture, software development,
    and implementation decisions.
    
    Inspects Brain context retrieved from research or prior interactions and formulates
    concrete implementation choices that integrate into AI Brain as DECISION or CODE_KNOWLEDGE.
    """

    def __init__(
        self,
        agent_id: str = "agent-coder",
        name: str = "Coding Agent",
        *,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        response_generator: Optional[Callable[[InteractionContextResult, AgentExecutionRequest], str]] = None,
    ) -> None:
        caps = capabilities or (
            AgentCapability.CODE_DEVELOPMENT.value,
            AgentCapability.ARCHITECTURE_DESIGN.value,
            AgentCapability.TASK_MANAGEMENT.value,
        )
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="simulated",
            agent_type=AgentType.CODING,
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
        """Generate architectural decisions and implementation plans using Brain context."""
        if self._custom_generator is not None:
            return self._custom_generator(context_result, request)

        content_lower = request.content.lower()
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""

        # Check if context contains research findings about database
        if "postgresql" in context_md.lower() or "database" in content_lower:
            return (
                "Implementation proposal: We decided to implement the database layer using PostgreSQL with Prisma ORM. "
                "Task created: Setup Prisma schema migrations and connection pool configurations."
            )
        elif "oauth" in context_md.lower() or "auth" in content_lower:
            return (
                "Implementation proposal: We decided to implement authentication using OAuth 2.0 with PKCE and JWT tokens. "
                "Task created: Implement token verification middleware with RS256 public key verification."
            )
        elif "redis" in context_md.lower() or "cache" in content_lower:
            return (
                "Implementation proposal: We decided to deploy Redis on port 6379 for distributed caching. "
                "Task created: Configure cache invalidation strategies on mutation endpoints."
            )
        else:
            return (
                f"Implementation plan for '{request.content}': We decided to adopt modular architecture "
                f"with strict typing and automated test suites for project '{request.project_id}'."
            )

    def stream_response(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Iterator[str]:
        """Stream implementation plan progressively in multi-word chunks."""
        full_text = self.generate_response(context_result, request)
        words = full_text.split(" ")
        chunk_size = 4
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk

