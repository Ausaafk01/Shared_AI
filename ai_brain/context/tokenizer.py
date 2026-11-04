"""
Tokenizer abstraction and deterministic offline fallback for Context Builder.
"""

from abc import ABC, abstractmethod
import re
from typing import Sequence

from ai_brain.context.exceptions import TokenizerError


class BaseTokenizer(ABC):
    """
    Abstract base class for context tokenizers.
    
    Allows plugging in production tokenizers (e.g., tiktoken, huggingface)
    while providing a stable interface for budget management.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the estimated or exact number of tokens in the given text."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Convert text into a list of token strings."""
        pass


class DeterministicTokenizer(BaseTokenizer):
    """
    Deterministic offline tokenizer with zero external dependencies.
    
    Uses standard subword tokenization heuristics:
    - Words, punctuation marks, and significant whitespace are recognized.
    - Words longer than chars_per_token produce additional tokens for subword chunks.
    - Punctuation and standalone symbols are counted individually.
    - Deterministic, consistent, non-negative, and handles edge cases safely.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise TokenizerError("chars_per_token must be greater than zero")
        self.chars_per_token = float(chars_per_token)
        # Regex captures words (alphanumeric+underscore/hyphen), individual punctuation/symbols, and newlines
        self._token_pattern = re.compile(r"[a-zA-Z0-9_\-]+|[^\w\s]|\n+")

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into subword/symbol units deterministically.
        """
        if not text:
            return []

        raw_matches = self._token_pattern.findall(text)
        tokens: list[str] = []

        chunk_size = max(1, int(round(self.chars_per_token)))
        for match in raw_matches:
            if match.startswith("\n"):
                tokens.append(match)
            elif match.isalnum() or ("_" in match) or ("-" in match):
                # Word match: if long, break into subword pieces
                if len(match) <= chunk_size:
                    tokens.append(match)
                else:
                    for i in range(0, len(match), chunk_size):
                        tokens.append(match[i:i + chunk_size])
            else:
                # Punctuation or symbol
                tokens.append(match)

        return tokens

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text deterministically.
        """
        if not text:
            return 0
        if not text.strip():
            # If purely whitespace with newlines
            return len(self.tokenize(text))
        return len(self.tokenize(text))

