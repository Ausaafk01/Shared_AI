"""Domain exceptions for Authentication and Role-Based Access Control."""

from __future__ import annotations


class AuthError(Exception):
    """Base exception for all authentication and authorization errors."""
    pass


class AuthenticationError(AuthError):
    """Raised when authentication fails.

    Deliberately uniform to prevent account or key enumeration.
    """
    def __init__(self, message: str = "Invalid or missing authentication credentials.") -> None:
        super().__init__(message)


class InvalidTokenError(AuthenticationError):
    """Raised when an access token is malformed, has invalid claims, or invalid signature."""
    def __init__(self, message: str = "Invalid access token.") -> None:
        super().__init__(message)


class TokenExpiredError(InvalidTokenError):
    """Raised when an access token has expired."""
    def __init__(self, message: str = "Access token has expired.") -> None:
        super().__init__(message)


class AlgorithmMismatchError(InvalidTokenError):
    """Raised when a token specifies an unexpected or prohibited signing algorithm."""
    def __init__(self, message: str = "Unsupported or prohibited token signing algorithm.") -> None:
        super().__init__(message)


class CredentialRevokedError(AuthenticationError):
    """Raised when an API key or credential has been revoked."""
    def __init__(self, message: str = "Credential has been revoked.") -> None:
        super().__init__(message)


class CredentialExpiredError(AuthenticationError):
    """Raised when an API key or credential has expired."""
    def __init__(self, message: str = "Credential has expired.") -> None:
        super().__init__(message)


class PermissionDeniedError(AuthError):
    """Raised when an authenticated principal attempts an unauthorized operation."""
    def __init__(self, message: str = "Access denied: insufficient permissions.") -> None:
        super().__init__(message)


class PrivilegeEscalationError(PermissionDeniedError):
    """Raised when an identity attempts to grant roles or permissions exceeding their own."""
    def __init__(self, message: str = "Privilege escalation denied.") -> None:
        super().__init__(message)


class UserNotFoundError(AuthError):
    """Raised when a requested user identity does not exist."""
    pass


class ProjectNotFoundError(AuthError):
    """Raised when a requested project does not exist."""
    pass


class MembershipNotFoundError(AuthError):
    """Raised when a user is not a member of the requested project."""
    pass


class CredentialNotFoundError(AuthError):
    """Raised when a requested credential record does not exist."""
    pass

