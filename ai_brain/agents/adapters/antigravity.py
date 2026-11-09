"""
Google Antigravity Agent provider adapter for AI Brain.
"""

import json
import logging
import shutil
import subprocess
from typing import Any, Callable, Dict, Mapping, Optional

from ai_brain.agents.adapters.base import ProviderAdapter, execute_http_request
from ai_brain.agents.exceptions import (
    AgentTimeoutError,
    ProviderAdapterError,
    ProviderAuthenticationError,
    ProviderConnectionError,
)
from ai_brain.agents.models import (
    AgentCapability,
    AgentExecutionRequest,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.loop.models import InteractionContextResult

logger = logging.getLogger(__name__)


class AntigravityAgentAdapter(ProviderAdapter):
    """
    Google Antigravity Agent Integration.
    
    Classification: Actual Agent API and CLI Integration.
    
    Supported Antigravity Integration Surfaces:
    1. Google Gemini Interactions API (`execution_mode="interactions_api"`, default):
       - Communicates with Google's managed Antigravity Agent environment hosted via Google GenAI / Vertex AI.
       - Spec: https://ai.google.dev/gemini-api/docs/antigravity-agent
       - Payload: agent ("antigravity-preview-09-2026"), input, context, environment ("remote").
    2. Antigravity CLI (`execution_mode="cli"`):
       - Uses the Antigravity terminal CLI (`agy`) for non-interactive execution inside local/remote workspaces.
       - Spec: https://antigravity.google/docs/cli/reference
    """

    def __init__(
        self,
        agent_id: str = "agent-antigravity",
        name: str = "Antigravity Agent",
        *,
        model: str = "antigravity-preview-09-2026",
        execution_mode: str = "interactions_api",
        environment: str = "remote",
        agent_type: AgentType = AgentType.GENERAL,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        api_key: Optional[str] = None,
        mock_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
        cli_binary: str = "agy",
    ) -> None:
        caps = capabilities or (
            AgentCapability.CODE_DEVELOPMENT.value,
            AgentCapability.ARCHITECTURE_DESIGN.value,
            AgentCapability.CODE_REVIEW.value,
            AgentCapability.REASONING.value,
        )
        conf = dict(configuration or {})
        conf["model"] = model
        conf["execution_mode"] = execution_mode
        conf["environment"] = environment
        conf["adapter_type"] = "antigravity_agent_integration"
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="google-antigravity",
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
        self.execution_mode = execution_mode.lower()
        self.environment = environment
        self.cli_binary = cli_binary

    def build_provider_payload(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Dict[str, Any]:
        """Construct Antigravity execution payload for Interactions API or CLI."""
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""

        if self.execution_mode == "cli":
            return {
                "command": self.cli_binary,
                "prompt": request.content,
                "context": context_md,
                "project_id": request.project_id,
                "session_id": request.session_id,
            }
        else:
            # Documented Google Gemini Interactions API format
            return {
                "endpoint": "/v1beta/interactions",
                "agent": self.model,
                "input": request.content,
                "context": context_md,
                "environment": self.environment,
                "metadata": {
                    "project_id": request.project_id,
                    "session_id": request.session_id,
                    "source": "ai_brain_loop",
                },
            }

    def _execute_live_call(self, payload: Dict[str, Any]) -> str:
        """Execute live call to Google Interactions API or Antigravity CLI."""
        if self.execution_mode == "cli":
            bin_path = shutil.which(self.cli_binary)
            if not bin_path:
                raise ProviderConnectionError(
                    f"Antigravity CLI '{self.cli_binary}' not found on PATH. "
                    "Install the Antigravity CLI or configure mock_transport."
                )
            full_prompt = f"Project Context:\n{payload['context']}\n\nTask:\n{payload['prompt']}"
            cmd = [self.cli_binary, "--prompt", full_prompt]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    raise ProviderAdapterError(f"Antigravity CLI failed with code {proc.returncode}: {proc.stderr}")
                return proc.stdout.strip()
            except subprocess.TimeoutExpired as e:
                raise AgentTimeoutError(f"Antigravity CLI timed out: {e}") from e
            except Exception as e:
                raise ProviderConnectionError(f"Antigravity CLI execution error: {e}") from e
        else:
            # Interactions API call
            api_key = self.get_api_key()
            if not api_key:
                raise ProviderAuthenticationError("Missing GEMINI_API_KEY for Antigravity Interactions API call.")
            url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={api_key}"
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            }
            body = {
                "agent": payload["agent"],
                "input": payload["input"],
                "context": payload["context"],
                "environment": payload["environment"],
            }
            res = execute_http_request(url, headers=headers, data=body)
            output_text = res.get("output_text")
            if output_text:
                return output_text
            candidates = res.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                return "\n".join(p["text"] for p in candidates[0]["content"]["parts"] if "text" in p)
            return str(res)


# Backward compatibility alias
class AntigravityAdapter(AntigravityAgentAdapter):
    """
    AntigravityAdapter alias maintaining backward compatibility while executing as AntigravityAgentAdapter.
    """
    pass
