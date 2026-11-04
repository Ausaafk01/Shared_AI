"""
Exception hierarchy for the Multi-Agent System subsystem in AI Brain.
"""


class AgentSystemError(Exception):
    """Base exception for all agent subsystem errors."""
    pass


class AgentRegistrationError(AgentSystemError):
    """Raised when registering an agent fails."""
    pass


class DuplicateAgentError(AgentRegistrationError):
    """Raised when an agent with the same ID is already registered."""
    pass


class AgentNotFoundError(AgentSystemError):
    """Raised when an agent is not found in the registry."""
    pass


class AgentStateError(AgentSystemError):
    """Raised when an operation is invalid for the agent's current state."""
    pass


class AgentExecutionError(AgentSystemError):
    """Raised when an agent execution fails."""
    pass


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent execution times out."""
    pass


class ProviderAdapterError(AgentSystemError):
    """Raised when a provider adapter encounters an error."""
    pass


class ProviderAuthenticationError(ProviderAdapterError):
    """Raised when provider authentication fails or credentials are missing."""
    pass


class ProviderRateLimitError(ProviderAdapterError):
    """Raised when provider rate limits are exceeded."""
    pass


class ProviderConnectionError(ProviderAdapterError):
    """Raised when connection to provider fails."""
    pass


class SessionError(AgentSystemError):
    """Base exception for session management errors."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when a requested session is not found."""
    pass


class SessionStateError(SessionError):
    """Raised when an operation is invalid for the session's current status."""
    pass


class CollaborationWorkflowError(AgentSystemError):
    """Raised when a multi-agent collaboration workflow encounters an error."""
    pass

