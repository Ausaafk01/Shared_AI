"""Cryptographic primitives, API key hashing, and JWT token management for AI Brain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import uuid

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidAlgorithmError, PyJWTError

from ai_brain.auth.exceptions import (
    AlgorithmMismatchError,
    InvalidTokenError,
    TokenExpiredError,
)
from ai_brain.auth.models import IdentityType

logger = logging.getLogger(__name__)

# Standard permitted algorithms for signed tokens
SUPPORTED_ALGORITHMS = frozenset({"RS256", "EdDSA", "HS256"})


class KeyType(StrEnum):
    """Cryptographic key type classification."""
    SYMMETRIC = "SYMMETRIC"
    RSA = "RSA"
    EDDSA = "EDDSA"


ALGORITHM_KEY_TYPES: Mapping[str, KeyType] = {
    "HS256": KeyType.SYMMETRIC,
    "RS256": KeyType.RSA,
    "EdDSA": KeyType.EDDSA,
}

# Markers indicating asymmetric key/certificate material
_ASYMMETRIC_MARKERS = (
    "BEGIN PUBLIC KEY",
    "BEGIN RSA PUBLIC KEY",
    "BEGIN CERTIFICATE",
    "BEGIN OPENSSH PRIVATE KEY",
    "ssh-rsa",
    "ssh-ed25519",
)


def _validate_key_compatibility(
    algorithm: str,
    key_type: KeyType,
    key_material: Union[str, bytes],
    is_signing: bool = False,
) -> None:
    """Validate that key material is cryptographically compatible with algorithm and key type.
    
    Strictly prevents algorithm substitution, key-type confusion (e.g. RSA public keys used as HMAC secret),
    and inadequate key length. Never formats secret key bytes into exception messages.
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")

    expected_type = ALGORITHM_KEY_TYPES.get(algorithm)
    if key_type != expected_type:
        raise AlgorithmMismatchError(
            f"Key type '{key_type.value}' is incompatible with algorithm '{algorithm}' (expected '{expected_type.value}')"
        )

    # Check for key-type confusion: Asymmetric key passed as HMAC symmetric secret
    if key_type == KeyType.SYMMETRIC:
        raw_str = key_material if isinstance(key_material, str) else key_material.decode("utf-8", errors="ignore")
        for marker in _ASYMMETRIC_MARKERS:
            if marker in raw_str:
                raise AlgorithmMismatchError(
                    "Asymmetric key / certificate material cannot be used as a symmetric HMAC key (key-type confusion rejected)"
                )
        raw_bytes = key_material.encode("utf-8") if isinstance(key_material, str) else key_material
        if len(raw_bytes) < 32:
            raise ValueError(f"Symmetric key for {algorithm} must be at least 32 bytes (got {len(raw_bytes)})")
    elif key_type == KeyType.RSA:
        raw_str = key_material if isinstance(key_material, str) else key_material.decode("utf-8", errors="ignore")
        if is_signing and "PRIVATE KEY" not in raw_str:
            raise ValueError("RSA signing key must be a private key PEM")
    elif key_type == KeyType.EDDSA:
        raw_str = key_material if isinstance(key_material, str) else key_material.decode("utf-8", errors="ignore")
        if is_signing and "PRIVATE KEY" not in raw_str:
            raise ValueError("EdDSA signing key must be a private key PEM")


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


@dataclass
class KeyEntry:
    """Cryptographic key metadata and material bound to algorithm and key type."""
    key_id: str
    algorithm: str
    key_type: KeyType
    verification_key: Union[str, bytes]
    signing_key: Optional[Union[str, bytes]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    revoked: bool = False

    def is_valid_at(self, dt: Optional[datetime] = None) -> bool:
        """Check whether key is unrevoked and unexpired at given timestamp."""
        if self.revoked:
            return False
        if self.expires_at is None:
            return True
        now = dt or datetime.now(timezone.utc)
        exp = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        return now < exp

    def __repr__(self) -> str:
        # Never leak raw key secrets in repr
        return (
            f"KeyEntry(key_id={self.key_id!r}, algorithm={self.algorithm!r}, "
            f"key_type={self.key_type.value!r}, revoked={self.revoked}, "
            f"expires_at={self.expires_at!r})"
        )


class KeyManager:
    """Manages active signing keys and historical verification keys for zero-downtime rotation.

    Binds every key to an explicit algorithm, key type, and expiration window.
    Strictly prevents algorithm substitution, key-type confusion, and leaks.
    """

    def __init__(
        self,
        active_key_id: str,
        active_signing_key: Union[str, bytes],
        active_verification_key: Optional[Union[str, bytes]] = None,
        algorithm: str = "HS256",
        key_type: Optional[KeyType] = None,
        default_retention_seconds: float = 86400.0,
    ) -> None:
        if not active_key_id:
            raise ValueError("active_key_id cannot be empty")
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm '{algorithm}'")

        inferred_type = key_type or ALGORITHM_KEY_TYPES[algorithm]
        _validate_key_compatibility(algorithm, inferred_type, active_signing_key, is_signing=True)
        v_key = active_verification_key or active_signing_key
        _validate_key_compatibility(algorithm, inferred_type, v_key, is_signing=False)

        self._default_retention_seconds = default_retention_seconds
        self._active_key_id = active_key_id
        self._keys: Dict[str, KeyEntry] = {
            active_key_id: KeyEntry(
                key_id=active_key_id,
                algorithm=algorithm,
                key_type=inferred_type,
                signing_key=active_signing_key,
                verification_key=v_key,
            )
        }

    @property
    def active_key_id(self) -> str:
        """Key ID currently used to sign newly issued access tokens."""
        return self._active_key_id

    @property
    def active_signing_key(self) -> Union[str, bytes]:
        """Key currently used for signing tokens."""
        entry = self._keys.get(self._active_key_id)
        if not entry or not entry.signing_key or entry.revoked:
            raise InvalidTokenError(f"Active signing key '{self._active_key_id}' is unavailable or revoked.")
        return entry.signing_key

    def register_verification_key(
        self,
        key_id: str,
        verification_key: Union[str, bytes],
        algorithm: str = "HS256",
        key_type: Optional[KeyType] = None,
        expires_at: Optional[datetime] = None,
    ) -> None:
        """Register a historical verification key to validate tokens during rotation."""
        if not key_id:
            raise ValueError("key_id cannot be empty")
        inferred_type = key_type or ALGORITHM_KEY_TYPES.get(algorithm, KeyType.SYMMETRIC)
        _validate_key_compatibility(algorithm, inferred_type, verification_key, is_signing=False)
        self._keys[key_id] = KeyEntry(
            key_id=key_id,
            algorithm=algorithm,
            key_type=inferred_type,
            verification_key=verification_key,
            expires_at=expires_at,
        )

    def rotate_signing_key(
        self,
        new_key_id: str,
        new_signing_key: Union[str, bytes],
        new_verification_key: Optional[Union[str, bytes]] = None,
        algorithm: Optional[str] = None,
        key_type: Optional[KeyType] = None,
        retention_seconds: Optional[float] = None,
    ) -> None:
        """Rotate active signing key, maintaining old verification keys in pool during retention window."""
        if not new_key_id:
            raise ValueError("new_key_id cannot be empty")

        # Retire old active key with retention window
        old_entry = self._keys.get(self._active_key_id)
        if old_entry:
            retention = retention_seconds if retention_seconds is not None else self._default_retention_seconds
            old_entry.expires_at = datetime.now(timezone.utc) + timedelta(seconds=retention)
            old_entry.signing_key = None  # Retired key can no longer sign new tokens

        alg = algorithm or (old_entry.algorithm if old_entry else "HS256")
        ktype = key_type or ALGORITHM_KEY_TYPES[alg]
        _validate_key_compatibility(alg, ktype, new_signing_key, is_signing=True)
        v_key = new_verification_key or new_signing_key
        _validate_key_compatibility(alg, ktype, v_key, is_signing=False)

        self._active_key_id = new_key_id
        self._keys[new_key_id] = KeyEntry(
            key_id=new_key_id,
            algorithm=alg,
            key_type=ktype,
            signing_key=new_signing_key,
            verification_key=v_key,
        )
        logger.info("Rotated token signing key to key ID '%s'", new_key_id)

    def revoke_key(self, key_id: str) -> None:
        """Explicitly revoke a key ID immediately, rejecting all tokens validated against it."""
        entry = self._keys.get(key_id)
        if entry:
            entry.revoked = True
            logger.warning("Revoked cryptographic key '%s'", key_id)

    def retire_key(self, key_id: str, retention_seconds: float = 0.0) -> None:
        """Retire a key ID, optionally setting a retention window after which it is rejected."""
        entry = self._keys.get(key_id)
        if entry:
            entry.expires_at = datetime.now(timezone.utc) + timedelta(seconds=retention_seconds)
            entry.signing_key = None

    def prune_retired_keys(self) -> int:
        """Remove keys whose retention window has passed."""
        now = datetime.now(timezone.utc)
        to_remove = [
            k for k, entry in self._keys.items()
            if k != self._active_key_id and entry.expires_at and (
                entry.expires_at if entry.expires_at.tzinfo else entry.expires_at.replace(tzinfo=timezone.utc)
            ) <= now
        ]
        for k in to_remove:
            del self._keys[k]
        return len(to_remove)

    def get_verification_key(self, key_id: str, expected_algorithm: Optional[str] = None) -> Union[str, bytes]:
        """Retrieve verification key by key ID with strict algorithm and validity checks."""
        entry = self._keys.get(key_id)
        if not entry:
            raise InvalidTokenError(f"Unknown signing key ID: {key_id!r}")
        if entry.revoked:
            raise InvalidTokenError(f"Signing key '{key_id}' has been explicitly revoked.")
        if entry.expires_at and datetime.now(timezone.utc) >= (
            entry.expires_at if entry.expires_at.tzinfo else entry.expires_at.replace(tzinfo=timezone.utc)
        ):
            raise InvalidTokenError(f"Signing key '{key_id}' has expired beyond its retention window.")
        if expected_algorithm and entry.algorithm != expected_algorithm:
            raise AlgorithmMismatchError(
                f"Key '{key_id}' configured for algorithm '{entry.algorithm}', but requested '{expected_algorithm}'."
            )
        return entry.verification_key

    def __repr__(self) -> str:
        return f"KeyManager(active_key_id={self._active_key_id!r}, key_count={len(self._keys)})"


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
        - Enforces header typ == 'JWT'.
        - Resolves verification key via 'kid' in header, validating against expected algorithm.
        - Validates signature, expiration, issued-at, not-before, issuer, audience, and subject.
        - Validates token payload identity type.
        """
        if not token or not isinstance(token, str):
            raise InvalidTokenError("Missing or empty token string.")

        # 1. Inspect unverified header to obtain algorithm, key ID, and token type
        try:
            unverified_header = jwt.get_unverified_header(token)
        except PyJWTError as e:
            raise InvalidTokenError(f"Malformed token header: {e}") from e

        header_typ = unverified_header.get("typ")
        if not header_typ or str(header_typ).upper() != "JWT":
            raise InvalidTokenError(f"Invalid token type '{header_typ}' in header; expected 'JWT'.")

        token_alg = unverified_header.get("alg")
        if not token_alg or str(token_alg).lower() == "none" or token_alg != self.algorithm:
            raise AlgorithmMismatchError(
                f"Token algorithm '{token_alg}' rejected; expected '{self.algorithm}'."
            )

        kid = unverified_header.get("kid")
        if not kid:
            raise InvalidTokenError("Token header missing required 'kid' parameter.")

        # 2. Retrieve verification key with algorithm validation
        verification_key = self.key_manager.get_verification_key(kid, expected_algorithm=self.algorithm)

        # 3. Decode and validate claims
        try:
            payload = jwt.decode(
                token,
                verification_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "require": ["exp", "iat", "nbf", "iss", "aud", "sub"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )

            # Validate identity type in payload
            typ_claim = payload.get("typ")
            if not typ_claim or not isinstance(typ_claim, str):
                raise InvalidTokenError("Token payload missing required 'typ' identity claim.")

            return payload
        except ExpiredSignatureError as e:
            raise TokenExpiredError("Access token has expired.") from e
        except InvalidAlgorithmError as e:
            raise AlgorithmMismatchError(f"Invalid signing algorithm: {e}") from e
        except PyJWTError as e:
            raise InvalidTokenError(f"Invalid token signature or claims: {e}") from e
