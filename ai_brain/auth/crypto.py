"""Cryptographic primitives, API key hashing, and JWT token management for AI Brain."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidAlgorithmError, PyJWTError

from ai_brain.auth.exceptions import (
    AlgorithmMismatchError,
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError,
)
from ai_brain.auth.models import IdentityType

logger = logging.getLogger(__name__)

# Standard permitted algorithms for signed tokens
SUPPORTED_ALGORITHMS = frozenset({"RS256", "EdDSA", "HS256"})


def generate_api_key(prefix: str = "agy") -> Tuple[str, str, str]:
    """Generate a high-entropy cryptographically secure API key.

    Returns:
        (raw_key, key_prefix, key_hash)
        - raw_key: The plaintext key string, shown once to caller.
        - key_prefix: The public key identifier prefix for safe logging and indexing.
        - key_hash: Hexadecimal SHA-256 digest of raw_key for secure persistent storage.
    """
    clean_prefix = (prefix or "agy").strip().lower().replace("_", "")
    key_prefix = secrets.token_hex(4)  # 8 hex characters
    secret_part = secrets.token_urlsafe(32)  # 43+ chars URL-safe base64 (256 bits entropy)
    raw_key = f"{clean_prefix}_ak_{key_prefix}_{secret_part}"

    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, f"{clean_prefix}_{key_prefix}", key_hash


def hash_api_key(raw_key: str) -> str:
    """Compute deterministic SHA-256 hash of a raw API key."""
    if not raw_key or not isinstance(raw_key, str):
        raise ValueError("raw_key must be a non-empty string")
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, expected_hash: str) -> bool:
    """Verify raw API key against stored hash in constant time."""
    if not raw_key or not expected_hash:
        return False
    computed_hash = hash_api_key(raw_key)
    return hmac.compare_digest(computed_hash, expected_hash)


class KeyManager:
    """Manages active signing keys and historical verification keys for zero-downtime rotation."""

    def __init__(
        self,
        active_key_id: str,
        active_signing_key: Union[str, bytes],
        active_verification_key: Optional[Union[str, bytes]] = None,
    ) -> None:
        if not active_key_id:
            raise ValueError("active_key_id cannot be empty")
        self._active_key_id = active_key_id
        self._active_signing_key = active_signing_key
        # For symmetric keys (e.g. HS256), verification key equals signing key
        self._active_verification_key = active_verification_key or active_signing_key

        self._verification_keys: Dict[str, Union[str, bytes]] = {
            active_key_id: self._active_verification_key
        }

    @property
    def active_key_id(self) -> str:
        """Key ID currently used to sign newly issued access tokens."""
        return self._active_key_id

    @property
    def active_signing_key(self) -> Union[str, bytes]:
        """Key currently used for signing tokens."""
        return self._active_signing_key

    def get_verification_key(self, key_id: str) -> Union[str, bytes]:
        """Retrieve verification key by key ID, raising InvalidTokenError if unknown."""
        if key_id not in self._verification_keys:
            raise InvalidTokenError(f"Unknown signing key ID: {key_id!r}")
        return self._verification_keys[key_id]

    def register_verification_key(self, key_id: str, verification_key: Union[str, bytes]) -> None:
        """Register a historical verification key to validate tokens during rotation."""
        self._verification_keys[key_id] = verification_key

    def rotate_signing_key(
        self,
        new_key_id: str,
        new_signing_key: Union[str, bytes],
        new_verification_key: Optional[Union[str, bytes]] = None,
    ) -> None:
        """Rotate active signing key, maintaining old verification keys in pool."""
        if not new_key_id:
            raise ValueError("new_key_id cannot be empty")
        self._active_key_id = new_key_id
        self._active_signing_key = new_signing_key
        self._active_verification_key = new_verification_key or new_signing_key
        self._verification_keys[new_key_id] = self._active_verification_key
        logger.info("Rotated token signing key to key ID '%s'", new_key_id)


class TokenService:
    """Issues and validates short-lived signed JWT access tokens with strict algorithm verification."""

    def __init__(
        self,
        key_manager: KeyManager,
        algorithm: str = "HS256",
        issuer: str = "ai-brain",
        audience: str = "ai-brain-api",
        default_expiry_seconds: int = 900,  # 15 minutes
    ) -> None:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported JWT algorithm '{algorithm}'. Must be one of {sorted(SUPPORTED_ALGORITHMS)}"
            )
        self.key_manager = key_manager
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.default_expiry_seconds = default_expiry_seconds

    def create_access_token(
        self,
        subject: str,
        identity_type: IdentityType = IdentityType.USER,
        name: str = "",
        scopes: Sequence[str] = (),
        extra_claims: Optional[Dict[str, Any]] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> str:
        """Create and sign a short-lived access token.

        Never puts API keys, passwords, or secrets into claims.
        """
        now = int(time.time())
        exp = now + (expires_in_seconds if expires_in_seconds is not None else self.default_expiry_seconds)

        payload: Dict[str, Any] = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": str(subject),
            "iat": now,
            "nbf": now,
            "exp": exp,
            "jti": str(uuid.uuid4()),
            "typ": identity_type.value,
            "name": name,
            "scopes": list(scopes),
        }

        # Validate extra_claims does not contain secrets
        if extra_claims:
            for k, v in extra_claims.items():
                lower_k = k.lower()
                if any(sec in lower_k for sec in ("secret", "password", "api_key", "token", "private")):
                    raise ValueError(f"Prohibited sensitive field '{k}' in token claims")
                if k not in payload:
                    payload[k] = v

        headers = {
            "kid": self.key_manager.active_key_id,
            "alg": self.algorithm,
            "typ": "JWT",
        }

        token = jwt.encode(
            payload,
            self.key_manager.active_signing_key,
            algorithm=self.algorithm,
            headers=headers,
        )
        return token

    def decode_and_validate_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token.

        Strict checks:
        - Rejects algorithm confusion and 'none' algorithm.
        - Resolves verification key via 'kid' in header.
        - Validates signature, expiration, issued-at, not-before, issuer, and audience.
        """
        if not token or not isinstance(token, str):
            raise InvalidTokenError("Missing or empty token string.")

        # 1. Inspect unverified header to obtain algorithm and key ID
        try:
            unverified_header = jwt.get_unverified_header(token)
        except PyJWTError as e:
            raise InvalidTokenError(f"Malformed token header: {e}") from e

        token_alg = unverified_header.get("alg")
        if not token_alg or token_alg.lower() == "none" or token_alg != self.algorithm:
            raise AlgorithmMismatchError(
                f"Token algorithm '{token_alg}' rejected; expected '{self.algorithm}'."
            )

        kid = unverified_header.get("kid")
        if not kid:
            # Fall back to active key if kid is omitted, or enforce kid
            kid = self.key_manager.active_key_id

        # 2. Retrieve verification key
        verification_key = self.key_manager.get_verification_key(kid)

        # 3. Decode and validate claims
        try:
            payload = jwt.decode(
                token,
                verification_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "require": ["exp", "iat", "iss", "aud", "sub"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
            return payload
        except ExpiredSignatureError as e:
            raise TokenExpiredError("Access token has expired.") from e
        except InvalidAlgorithmError as e:
            raise AlgorithmMismatchError(f"Invalid signing algorithm: {e}") from e
        except PyJWTError as e:
            raise InvalidTokenError(f"Invalid token signature or claims: {e}") from e

