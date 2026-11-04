"""
Claude (Anthropic) provider adapter for AI Brain.
"""

from typing import Any, Callable, Dict, Mapping, Optional

from ai_brain.agents.adapters.base import ProviderAdapter, execute_http_request
from ai_brain.agents.exceptions import ProviderAuthenticationError
from ai_brain.agents.models import (
    AgentCapability,
    AgentExecutionRequest,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.loop.models import InteractionContextResult


class ClaudeAdapter(ProviderAdapter):
    """
    Adapter integrating Anthropic Claude models via Messages API format.
    
    API Spec:
    - Endpoint: https://api.anthropic.com/v1/messages
    - Auth: x-api-key header (via ANTHROPIC_API_KEY environment variable)
    - Payload: model, max_tokens, system, messages: [{"role": "user", "content": ...}]
    """

    def __init__(
        self,
        agent_id: str = "agent-claude",
        name: str = "Claude Assistant",
        *,
        model: str = "claude-3-5-sonnet-20241022",
        agent_type: AgentType = AgentType.GENERAL,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        api_key: Optional[str] = None,
        mock_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        caps = capabilities or (
            AgentCapability.REASONING.value,
            AgentCapability.CODE_DEVELOPMENT.value,
            AgentCapability.INFORMATION_GATHERING.value,
        )
        conf = dict(configuration or {})
        conf["model"] = model
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="anthropic",
            agent_type=agent_type,
            capabilities=caps,
            configuration=conf,
            status=AgentStatus.ACTIVE,
        )
        super().__init__(
            record,
            env_key_name="ANTHROPIC_API_KEY",
            api_key=api_key,
            mock_transport=mock_transport,
        )
        self.model = model

    def build_provider_payload(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Dict[str, Any]:
        """Format request into Anthropic Messages API payload."""
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""
        system_prompt = (
            f"You are {self.name}, an AI collaborator for project '{request.project_id}'.\n"
            "Use the following established project context to ground your knowledge and avoid contradictions:\n\n"
            f"{context_md}"
        )
        return {
            "model": self.model,
            "max_tokens": request.token_budget or 1024,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": request.content,
                }
            ],
            "metadata": {"project_id": request.project_id, "session_id": request.session_id},
        }

    def _execute_live_call(self, payload: Dict[str, Any]) -> str:
        """Execute live HTTP call to Anthropic Messages API."""
        api_key = self.get_api_key()
        if not api_key:
            raise ProviderAuthenticationError("Missing ANTHROPIC_API_KEY for live call.")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        res = execute_http_request(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            data=payload,
        )
        content_blocks = res.get("content", [])
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text" and "text" in b]
        return "\n".join(text_parts) if text_parts else ""

