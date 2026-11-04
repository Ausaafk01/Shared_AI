"""
OpenAI Codex Coding Agent and OpenAI Model adapters for AI Brain.
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional

from ai_brain.agents.adapters.base import ProviderAdapter, execute_http_request
from ai_brain.agents.exceptions import (
    AgentTimeoutError,
    ProviderAdapterError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
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


class OpenAIModelAdapter(ProviderAdapter):
    """
    Generic OpenAI Model API Adapter.
    
    Classification: Generic model API execution (inference interface, NOT an autonomous agent loop).
    
    Supported Endpoints:
    1. Chat Completions (`endpoint_mode="chat_completions"`, default):
       - Endpoint: https://api.openai.com/v1/chat/completions
       - Standard multi-message inference endpoint.
    2. Responses API (`endpoint_mode="responses"`):
       - Endpoint: https://api.openai.com/v1/responses
       - Model inference interface supporting tools (code_interpreter, file_search).
       - Note: Responses API is an OpenAI model inference interface; using it does not
         constitute the autonomous Codex agent loop.
    """

    def __init__(
        self,
        agent_id: str = "agent-openai-model",
        name: str = "OpenAI Model Assistant",
        *,
        model: str = "gpt-4o",
        endpoint_mode: str = "chat_completions",
        agent_type: AgentType = AgentType.GENERAL,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        api_key: Optional[str] = None,
        mock_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        caps = capabilities or (
            AgentCapability.CODE_DEVELOPMENT.value,
            AgentCapability.REASONING.value,
            AgentCapability.TASK_MANAGEMENT.value,
        )
        conf = dict(configuration or {})
        conf["model"] = model
        conf["endpoint_mode"] = endpoint_mode
        conf["adapter_type"] = "generic_model_api"
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="openai",
            agent_type=agent_type,
            capabilities=caps,
            configuration=conf,
            status=AgentStatus.ACTIVE,
        )
        super().__init__(
            record,
            env_key_name="OPENAI_API_KEY",
            api_key=api_key,
            mock_transport=mock_transport,
        )
        self.model = model
        self.endpoint_mode = endpoint_mode.lower()

    def build_provider_payload(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Dict[str, Any]:
        """Format request into OpenAI API payload according to endpoint_mode."""
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""
        system_content = (
            f"You are {self.name}, an expert AI assistant on project '{request.project_id}'.\n"
            "Ground your code and decisions on the following project context:\n\n"
            f"{context_md}"
        )

        if self.endpoint_mode == "responses":
            return {
                "endpoint": "/v1/responses",
                "model": self.model,
                "instructions": system_content,
                "input": request.content,
                "tools": [
                    {"type": "code_interpreter"},
                    {"type": "file_search"},
                ],
                "metadata": {
                    "project_id": request.project_id,
                    "session_id": request.session_id,
                    "agent": self.agent_id,
                },
                "max_output_tokens": request.token_budget or 2048,
            }
        else:
            return {
                "endpoint": "/v1/chat/completions",
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": request.content},
                ],
                "max_tokens": request.token_budget or 1024,
                "temperature": 0.2,
            }

    def _execute_live_call(self, payload: Dict[str, Any]) -> str:
        """Execute live HTTP call to OpenAI model endpoint."""
        api_key = self.get_api_key()
        if not api_key:
            raise ProviderAuthenticationError("Missing OPENAI_API_KEY for live call.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if self.endpoint_mode == "responses":
            body = {
                "model": payload["model"],
                "instructions": payload["instructions"],
                "input": payload["input"],
                "tools": payload["tools"],
            }
            res = execute_http_request("https://api.openai.com/v1/responses", headers=headers, data=body)
            output_text = res.get("output_text")
            if output_text:
                return output_text
            items = res.get("output", [])
            for item in items:
                if item.get("type") == "message" and "content" in item:
                    return str(item["content"])
            return str(res)
        else:
            body = {
                "model": payload["model"],
                "messages": payload["messages"],
                "max_tokens": payload["max_tokens"],
                "temperature": payload["temperature"],
            }
            res = execute_http_request(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=body,
            )
            choices = res.get("choices", [])
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                return choices[0]["message"]["content"]
            return ""


class CodexAgentAdapter(ProviderAdapter):
    """
    OpenAI Codex Autonomous Coding Agent Integration.
    
    Classification: Authentic Codex Agent Execution through Codex CLI or App-Server.
    
    Supported Codex Integration Surfaces:
    1. Codex CLI (`execution_mode="cli"`, default):
       - Uses official non-interactive `codex exec` mode with granular OS-level sandbox permissions.
       - Supported permission models:
         * `--sandbox read-only` (default): Safely inspects files and reasons without filesystem writes.
         * `--sandbox workspace-write`: Allows workspace file edits and local dev commands within project root.
       - Approval model:
         * `--ask-for-approval never`: Headless automated execution within the granted sandbox.
         * `--ask-for-approval on-request`: Pauses when operations exceed boundary limits.
       - Unrestricted execution (`danger-full-access` / YOLO mode) is prohibited by default.
    2. Codex App-Server (`execution_mode="app_server"`):
       - Connects via WebSocket or JSON-RPC to the persistent Codex agent daemon (`codex app-server`).
    """

    ALLOWED_SANDBOX_MODES = ("read-only", "workspace-write")
    ALLOWED_APPROVAL_MODES = ("never", "on-request")

    def __init__(
        self,
        agent_id: str = "agent-codex",
        name: str = "Codex Coding Agent",
        *,
        model: str = "codex-1",
        execution_mode: str = "cli",
        sandbox_mode: str = "read-only",
        approval_mode: str = "never",
        agent_type: AgentType = AgentType.CODING,
        capabilities: Optional[tuple[str, ...]] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        api_key: Optional[str] = None,
        mock_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
        cli_binary: str = "codex",
        app_server_url: str = "ws://127.0.0.1:4500",
    ) -> None:
        if sandbox_mode not in self.ALLOWED_SANDBOX_MODES:
            raise ValueError(
                f"Invalid or prohibited sandbox mode '{sandbox_mode}'. "
                f"Supported safe sandbox modes are: {self.ALLOWED_SANDBOX_MODES}. "
                "Unrestricted execution (e.g. 'danger-full-access') is prohibited."
            )
        if approval_mode not in self.ALLOWED_APPROVAL_MODES:
            raise ValueError(
                f"Invalid approval mode '{approval_mode}'. "
                f"Supported approval modes are: {self.ALLOWED_APPROVAL_MODES}."
            )

        caps = capabilities or (
            AgentCapability.CODE_DEVELOPMENT.value,
            AgentCapability.ARCHITECTURE_DESIGN.value,
            AgentCapability.TASK_MANAGEMENT.value,
            AgentCapability.CODE_REVIEW.value,
        )
        conf = dict(configuration or {})
        conf["model"] = model
        conf["execution_mode"] = execution_mode
        conf["sandbox_mode"] = sandbox_mode
        conf["approval_mode"] = approval_mode
        conf["adapter_type"] = "codex_agent_integration"
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            provider="openai-codex",
            agent_type=agent_type,
            capabilities=caps,
            configuration=conf,
            status=AgentStatus.ACTIVE,
        )
        super().__init__(
            record,
            env_key_name="OPENAI_API_KEY",
            api_key=api_key,
            mock_transport=mock_transport,
        )
        self.model = model
        self.execution_mode = execution_mode.lower()
        self.sandbox_mode = sandbox_mode
        self.approval_mode = approval_mode
        self.cli_binary = cli_binary
        self.app_server_url = app_server_url

    def build_cli_args(self, prompt: str) -> List[str]:
        """
        Construct exact command-line arguments for non-interactive `codex exec`.
        Enforces:
        - `--sandbox` with restrictive safe mode (`read-only` or `workspace-write`).
        - `--ask-for-approval` with appropriate policy (`never` or `on-request`).
        - Zero usage of deprecated `--full-auto`.
        - Zero silent elevation to unrestricted execution.
        """
        return [
            self.cli_binary,
            "exec",
            "--sandbox",
            self.sandbox_mode,
            "--ask-for-approval",
            self.approval_mode,
            "--model",
            self.model,
            prompt,
        ]

    def build_provider_payload(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Dict[str, Any]:
        """Construct Codex Agent execution payload for CLI or App-Server."""
        context_md = context_result.context_package.to_markdown() if context_result.context_package else ""
        system_instructions = (
            f"You are {self.name}, an authentic Codex coding agent for project '{request.project_id}'.\n"
            "Align implementation with established project architecture and constraints:\n\n"
            f"{context_md}"
        )
        full_prompt = f"{system_instructions}\n\nTask: {request.content}"

        if self.execution_mode == "app_server":
            return {
                "server_url": self.app_server_url,
                "agent_id": self.agent_id,
                "project_id": request.project_id,
                "session_id": request.session_id,
                "model": self.model,
                "prompt": full_prompt,
                "sandbox_mode": self.sandbox_mode,
                "approval_mode": self.approval_mode,
            }
        else:
            # CLI non-interactive execution mode (codex exec --sandbox ... --ask-for-approval ...)
            cli_args = self.build_cli_args(full_prompt)
            return {
                "command": self.cli_binary,
                "subcommand": "exec",
                "args": cli_args,
                "flags": [
                    "--sandbox",
                    self.sandbox_mode,
                    "--ask-for-approval",
                    self.approval_mode,
                    "--model",
                    self.model,
                ],
                "sandbox_mode": self.sandbox_mode,
                "approval_mode": self.approval_mode,
                "instructions": request.content,
                "context": context_md,
                "full_prompt": full_prompt,
                "project_id": request.project_id,
                "session_id": request.session_id,
            }

    def _execute_live_call(self, payload: Dict[str, Any]) -> str:
        """Execute live Codex agent interaction via CLI or App-Server."""
        if self.execution_mode == "app_server":
            raise NotImplementedError(
                f"Live WebSocket connection to Codex app-server at '{self.app_server_url}' "
                "requires active daemon instance. Supply mock_transport for testing."
            )
        else:
            # CLI non-interactive execution
            bin_path = shutil.which(self.cli_binary)
            if not bin_path:
                raise ProviderConnectionError(
                    f"Codex CLI binary '{self.cli_binary}' not found on PATH. "
                    "Install the Codex CLI or configure mock_transport."
                )
            cmd = payload["args"]
            env = os.environ.copy()
            if self.get_api_key():
                env["OPENAI_API_KEY"] = self.get_api_key()

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
                if proc.returncode != 0:
                    raise ProviderAdapterError(f"Codex CLI exited with code {proc.returncode}: {proc.stderr}")
                return proc.stdout.strip()
            except subprocess.TimeoutExpired as e:
                raise AgentTimeoutError(f"Codex CLI execution timed out: {e}") from e
            except Exception as e:
                raise ProviderConnectionError(f"Codex CLI execution failed: {e}") from e


# Backward compatibility alias
class CodexAdapter(CodexAgentAdapter):
    """
    CodexAdapter alias maintaining backward compatibility while executing as CodexAgentAdapter.
    """
    pass
