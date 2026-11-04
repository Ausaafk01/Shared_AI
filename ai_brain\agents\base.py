"""
Base Agent interface and abstraction for AI Brain.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import time
from typing import Any, List, Mapping, Optional

from ai_brain.agents.exceptions import (
    AgentExecutionError,
    AgentStateError,
    AgentTimeoutError,
    ProviderAdapterError,
)
from ai_brain.agents.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.loop.models import (
    AgentInteractionInput,
    AgentResponseInput,
    InteractionCompletionResult,
    InteractionContextResult,
    InteractionRecord,
)
from ai_brain.loop.orchestrator import BrainLoop

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Abstract base class for all AI Brain agents (simulated and real provider adapters).
    
    Guarantees:
    - Provider-agnostic interface decoupling the Brain from specific LLM models.
    - Standardized execution lifecycle: prepare_context -> generate_response -> complete_interaction.
    - Preserves agent identity, project boundaries, and session continuity.
    - Handles timeouts and execution errors gracefully.
    - Agent-specific interaction history retrieval.
    """

    def __init__(self, record: AgentRecord) -> None:
        self._record = record

    @property
    def record(self) -> AgentRecord:
        """Immutable record of the agent's identity and metadata."""
        return self._record

    @property
    def agent_id(self) -> str:
        return self._record.agent_id

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def provider(self) -> str:
        return self._record.provider

    @property
    def agent_type(self) -> AgentType:
        return self._record.agent_type

    @property
    def status(self) -> AgentStatus:
        return self._record.status

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._record.capabilities

    @property
    def configuration(self) -> Mapping[str, Any]:
        return self._record.configuration

    def initialize(self) -> None:
        """Initialize the agent resources or connections."""
        pass

    def prepare_context(
        self,
        request: AgentExecutionRequest,
        brain_loop: BrainLoop,
    ) -> InteractionContextResult:
        """
        Stage 1: Assemble and deliver agent-ready context from Brain Loop.
        """
        if self.status in (AgentStatus.INACTIVE, AgentStatus.DECOMMISSIONED):
            raise AgentStateError(f"Agent '{self.agent_id}' is {self.status.value} and cannot process inputs.")

        interaction_input = AgentInteractionInput(
            project_id=request.project_id,
            agent_id=self.agent_id,
            session_id=request.session_id,
            content=request.content,
            query=request.query,
            token_budget=request.token_budget,
            retrieval_method=request.retrieval_method,
            idempotency_key=request.idempotency_key,
            metadata=dict(request.metadata),
        )
        return brain_loop.prepare_context(interaction_input)

    @abstractmethod
    def generate_response(
        self,
        context_result: InteractionContextResult,
        request: AgentExecutionRequest,
    ) -> str:
        """
        Generate or obtain the agent's response given assembled context.
        Must be implemented by simulated agents or provider adapters.
        """
        pass

    def complete_interaction(
        self,
        interaction_id: str,
        response_content: str,
        request: AgentExecutionRequest,
        brain_loop: BrainLoop,
    ) -> InteractionCompletionResult:
        """
        Stage 2: Submit agent response to Brain Loop to extract memories and sync knowledge.
        """
        response_input = AgentResponseInput(
            interaction_id=interaction_id,
            project_id=request.project_id,
            agent_id=self.agent_id,
            session_id=request.session_id,
            content=response_content,
            metadata=dict(request.metadata),
        )
        return brain_loop.complete_interaction(response_input)

    def execute(
        self,
        request: AgentExecutionRequest,
        brain_loop: BrainLoop,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> AgentExecutionResponse:
        """
        Execute full agent interaction lifecycle against AI Brain.
        """
        if self.status in (AgentStatus.INACTIVE, AgentStatus.DECOMMISSIONED):
            raise AgentStateError(f"Agent '{self.agent_id}' is {self.status.value} and cannot execute requests.")

        start_time = time.perf_counter()

        # Step 1: Prepare Brain Context
        context_result = self.prepare_context(request, brain_loop)

        # Step 2: Generate Response with timeout/error protection
        try:
            if timeout_seconds is not None and timeout_seconds <= 0:
                raise AgentTimeoutError(f"Agent '{self.agent_id}' timed out before response generation.")

            response_content = self.generate_response(context_result, request)
        except (AgentTimeoutError, ProviderAdapterError) as e:
            self.handle_error(e, request)
            raise
        except Exception as e:
            self.handle_error(e, request)
            raise AgentExecutionError(f"Agent '{self.agent_id}' execution failed: {e}") from e

        # Step 3: Complete Brain Interaction
        completion_result = self.complete_interaction(
            context_result.interaction_id,
            response_content,
            request,
            brain_loop,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentExecutionResponse(
            agent_id=self.agent_id,
            project_id=request.project_id,
            session_id=request.session_id,
            interaction_id=context_result.interaction_id,
            content=response_content,
            context_package=context_result.context_package,
            completion_result=completion_result,
            execution_time_ms=elapsed_ms,
            metadata={"provider": self.provider, "agent_type": self.agent_type.value},
        )

    def get_history(
        self,
        brain_loop: BrainLoop,
        *,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[InteractionRecord]:
        """
        Access agent-specific interaction history from Brain Loop.
        """
        records = brain_loop.list_interactions(project_id=project_id, session_id=session_id)
        return [r for r in records if r.agent_id == self.agent_id]

    def handle_error(self, error: Exception, request: AgentExecutionRequest) -> None:
        """Log and track error occurrences during execution."""
        logger.error(
            "Error executing agent '%s' on project '%s', session '%s': %s",
            self.agent_id,
            request.project_id,
            request.session_id,
            error,
        )
