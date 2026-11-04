"""
Multi-Agent System for AI Brain.

Provides:
- Common agent abstraction and lifecycle interface.
- Persistent Agent Registry with duplicate protection.
- Simulated agents (Research, Coding, Review).
- Session continuity and persistence across restarts.
- Agent switching within and across sessions.
- Multi-agent collaboration workflows.
- Provider adapters (Claude, Codex, Gemini, Antigravity).
"""

from ai_brain.agents.adapters import (
    AntigravityAdapter,
    AntigravityAgentAdapter,
    ClaudeAdapter,
    CodexAdapter,
    CodexAgentAdapter,
    GeminiAdapter,
    OpenAIModelAdapter,
    ProviderAdapter,
)
from ai_brain.agents.base import Agent
from ai_brain.agents.collaboration import (
    CollaborationEngine,
    CollaborationStep,
    CollaborationStepResult,
    CollaborationWorkflowRecord,
    WorkflowStatus,
)
from ai_brain.agents.exceptions import (
    AgentExecutionError,
    AgentNotFoundError,
    AgentRegistrationError,
    AgentStateError,
    AgentSystemError,
    AgentTimeoutError,
    CollaborationWorkflowError,
    DuplicateAgentError,
    ProviderAdapterError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    SessionError,
    SessionNotFoundError,
    SessionStateError,
)
from ai_brain.agents.models import (
    AgentCapability,
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentRecord,
    AgentStatus,
    AgentType,
)
from ai_brain.agents.registry import (
    AgentRegistry,
    AgentRegistryStore,
    SQLiteAgentRegistryStore,
)
from ai_brain.agents.sessions import (
    SessionManager,
    SessionRecord,
    SessionStatus,
    SessionStore,
    SQLiteSessionStore,
)
from ai_brain.agents.simulated import (
    CodingAgent,
    ResearchAgent,
    ReviewAgent,
)
from ai_brain.agents.switching import AgentSwitcher

__all__ = [
    # Base & Models
    "Agent",
    "AgentRecord",
    "AgentType",
    "AgentStatus",
    "AgentCapability",
    "AgentExecutionRequest",
    "AgentExecutionResponse",
    # Registry
    "AgentRegistry",
    "AgentRegistryStore",
    "SQLiteAgentRegistryStore",
    # Simulated Agents
    "ResearchAgent",
    "CodingAgent",
    "ReviewAgent",
    # Sessions
    "SessionRecord",
    "SessionStatus",
    "SessionStore",
    "SQLiteSessionStore",
    "SessionManager",
    # Switching & Collaboration
    "AgentSwitcher",
    "CollaborationEngine",
    "CollaborationStep",
    "CollaborationStepResult",
    "CollaborationWorkflowRecord",
    "WorkflowStatus",
    # Provider Adapters
    "ProviderAdapter",
    "ClaudeAdapter",
    "CodexAdapter",
    "CodexAgentAdapter",
    "OpenAIModelAdapter",
    "GeminiAdapter",
    "AntigravityAdapter",
    "AntigravityAgentAdapter",
    # Exceptions
    "AgentSystemError",
    "AgentRegistrationError",
    "DuplicateAgentError",
    "AgentNotFoundError",
    "AgentStateError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "ProviderAdapterError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderConnectionError",
    "SessionError",
    "SessionNotFoundError",
    "SessionStateError",
    "CollaborationWorkflowError",
]

