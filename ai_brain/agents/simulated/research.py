"""
Simulated Research Agent specialized in information gathering and research findings.
"""

from datetime import datetime
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


class ResearchAgent(Agent):
    """
    Simulated agent specializing in research, information gathering, and evidence compilation.
    
    Generates structured factual findings that integrate into AI Brain as MemoryType.FACT.
    Usable 100% offline without external API credentials.
    """

    def __init__(
        self,
        agent_id: str = "agent-researcher",
        name: str = "Research Agent",
        *,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        response_generator: Optional[Callable[[InteractionContextResult, AgentExecutionRequest], str]] = None,
    ) -> None:
        caps = capabilities or (
            AgentCapability.INFORMATION_GATHERING.value,
            AgentCapability.REASONING.value,
        )
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="simulated",
            agent_type=AgentType.RESEARCH,
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
        """Generate structured research findings given context and prompt."""
        if self._custom_generator is not None:
            return self._custom_generator(context_result, request)

        content_lower = request.content.lower()

        # Deterministic domain responses designed to integrate cleanly with MemoryExtractor
        if "database" in content_lower or "storage" in content_lower:
            return (
                "Research findings: The project uses PostgreSQL 16 as its database with connection pooling. "
                "Supporting evidence indicates PostgreSQL provides robust ACID transactions and JSONB support."
            )
        elif "auth" in content_lower or "security" in content_lower:
            return (
                "Research findings: Authentication is handled through OAuth 2.0 with JWT tokens. "
                "Evidence shows asymmetric RS256 signing is standard for distributed microservice verification."
            )
        elif "cache" in content_lower or "caching" in content_lower:
            return (
                "Research findings: The project uses Redis for distributed caching and session management. "
                "Benchmarks confirm sub-millisecond retrieval latency for hot session data."
            )
        else:
            return (
                f"Research findings on '{request.content}': The investigation confirms that standard specifications "
                f"and verified architectural patterns are applicable for project '{request.project_id}'."
            )

    def stream_response(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Iterator[str]:
        """Stream research findings progressively in multi-word chunks."""
        full_text = self.generate_response(context_result, request)
        words = full_text.split(" ")
        chunk_size = 4
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk

