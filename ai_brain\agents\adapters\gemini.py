"""
Google Gemini provider adapter for AI Brain.
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


class GeminiAdapter(ProviderAdapter):
    """
    Adapter integrating Google Gemini models via Google GenAI REST format.
    
    API Spec:
    - Endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    - Auth: Query param or x-goog-api-key header (via GEMINI_API_KEY environment variable)
    - Payload: contents: [{"role": "user", "parts": [{"text": ...}]}], systemInstruction: {"parts": [{"text": ...}]}
    """

    def __init__(
        self,
        agent_id: str = "agent-gemini",
        name: str = "Gemini Assistant",
        *,
        model: str = "gemini-1.5-pro",
        agent_type: AgentType = AgentType.GENERAL,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        api_key: Optional[str] = None,
        mock_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        caps = capabilities or (
            AgentCapability.REASONING.value,
            AgentCapability.INFORMATION_GATHERING.value,
            AgentCapability.CONTRADICTION_DETECTION.value,
        )
        conf = dict(configuration or {})
        conf["model"] = model
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="google",
            agent_type=agent_type,
            capabilities=caps,
            configuration=conf,
            status=AgentStatus.ACTIVE,
        )
        super().__init__(
            record,
            env_key_name="GEMINI_API_KEY",
            api_key=api_key,
            mock_transport=mock_transport,
        )
        self.model = model

    def build_provider_payload(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Dict[str, Any]:
        """Format request into Google Gemini generateContent payload."""
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""
        system_instruction = (
            f"You are {self.name}, an AI collaborator working on project '{request.project_id}'.\n"
            "Align your reasoning with established shared knowledge:\n\n"
            f"{context_md}"
        )
        return {
            "model": self.model,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": request.content}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": request.token_budget or 1024,
                "temperature": 0.2,
            },
        }

    def _execute_live_call(self, payload: Dict[str, Any]) -> str:
        """Execute live HTTP call to Google Gemini REST API."""
        api_key = self.get_api_key()
        if not api_key:
            raise ProviderAuthenticationError("Missing GEMINI_API_KEY for live call.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
        headers = {"content-type": "application/json"}
        # Exclude model from body if needed by API
        body_payload = {
            "contents": payload["contents"],
            "systemInstruction": payload["systemInstruction"],
            "generationConfig": payload["generationConfig"],
        }
        res = execute_http_request(url, headers=headers, data=body_payload)
        candidates = res.get("candidates", [])
        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
            parts = candidates[0]["content"]["parts"]
            return "\n".join(p["text"] for p in parts if "text" in p)
        return ""

