"""
Base class and error handling for external provider adapters in AI Brain.
"""

from abc import abstractmethod
import json
import logging
import os
from typing import Any, Callable, Dict, Optional
import urllib.error
import urllib.request

from ai_brain.agents.base import Agent
from ai_brain.agents.exceptions import (
    AgentTimeoutError,
    ProviderAdapterError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from ai_brain.agents.models import (
    AgentExecutionRequest,
    AgentRecord,
)
from ai_brain.loop.models import InteractionContextResult

logger = logging.getLogger(__name__)


class ProviderAdapter(Agent):
    """
    Base class for external LLM provider adapters (Claude, Codex, Gemini, Antigravity).
    
    Guarantees:
    - Implements the common Agent interface.
    - Credentials and secrets are resolved strictly from environment variables or secure
      injectors, never stored in Brain memory, logs, or agent records.
    - Translates provider-specific rate limits, timeouts, and HTTP errors into domain exceptions.
    - Supports simulated offline transport for deterministic testing without external network access.
    - Preserves provider and agent identities across all operations.
    """

    def __init__(
        self,
        record: AgentRecord,
        *,
        env_key_name: Optional[str] = None,
        api_key: Optional[str] = None,
        mock_transport: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        super().__init__(record)
        self.env_key_name = env_key_name
        self._explicit_key = api_key
        self.mock_transport = mock_transport

    def get_api_key(self) -> Optional[str]:
        """
        Dynamically resolve API credential from environment or explicit parameter.
        Never written to Brain storage or logs.
        """
        if self._explicit_key:
            return self._explicit_key
        if self.env_key_name:
            return os.environ.get(self.env_key_name)
        return None

    def validate_authentication(self) -> None:
        """Verify that required credentials exist before executing live calls."""
        if self.mock_transport is None and not self.get_api_key():
            env_hint = f" ({self.env_key_name})" if self.env_key_name else ""
            raise ProviderAuthenticationError(
                f"Missing API credential for provider '{self.provider}'{env_hint}. "
                "Set the required environment variable or supply a mock transport."
            )

    @abstractmethod
    def build_provider_payload(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> Dict[str, Any]:
        """Construct the provider-specific payload format."""
        pass

    def call_provider(self, payload: Dict[str, Any]) -> str:
        """
        Execute call to the provider or mock transport with error translation.
        """
        if self.mock_transport is not None:
            try:
                return self.mock_transport(payload)
            except (ProviderAdapterError, AgentTimeoutError):
                raise
            except TimeoutError as e:
                raise AgentTimeoutError(f"Provider '{self.provider}' call timed out: {e}") from e
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    raise ProviderRateLimitError(f"Provider '{self.provider}' rate limit exceeded: {e}") from e
                if "auth" in err_str or "unauthorized" in err_str or "401" in err_str or "403" in err_str:
                    raise ProviderAuthenticationError(f"Provider '{self.provider}' authentication failed: {e}") from e
                if "connection" in err_str or "network" in err_str:
                    raise ProviderConnectionError(f"Provider '{self.provider}' connection failed: {e}") from e
                raise ProviderAdapterError(f"Provider '{self.provider}' error: {e}") from e

        self.validate_authentication()
        # Subclasses implement their live HTTP / CLI call
        return self._execute_live_call(payload)

    def _execute_live_call(self, payload: Dict[str, Any]) -> str:
        """Default hook for live provider execution."""
        raise NotImplementedError("Live network calls require provider-specific HTTP integration.")

    def generate_response(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> str:
        """Generate response by translating Brain context into provider payload."""
        payload = self.build_provider_payload(context_result, request)
        return self.call_provider(payload)


def execute_http_request(
    url: str,
    *,
    headers: Dict[str, str],
    data: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Standard library HTTP client executing JSON requests against AI provider endpoints
    with transparent mapping of status codes to domain exceptions.
    """
    post_data = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url,
        data=post_data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        if e.code == 429:
            raise ProviderRateLimitError(f"HTTP 429 Rate limit exceeded: {body}") from e
        elif e.code in (401, 403):
            raise ProviderAuthenticationError(f"HTTP {e.code} Authentication failed: {body}") from e
        else:
            raise ProviderAdapterError(f"HTTP {e.code} error: {body}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        if "timed out" in str(e).lower() or isinstance(e, TimeoutError):
            raise AgentTimeoutError(f"Request timed out: {e}") from e
        raise ProviderConnectionError(f"Connection failed: {e}") from e

