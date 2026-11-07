"""Recursive container freezing and unfreezing utilities for immutability."""

from types import MappingProxyType
from typing import Any


def _freeze_container(data: Any) -> Any:
    """Recursively freeze dicts, lists, sets, and sequences for immutability.

    Guarantees:
    - Dicts become read-only MappingProxyType.
    - Lists and tuples become immutable tuples.
    - Sets and frozensets become frozensets.
    - Nested combinations are deeply frozen.
    - Primitive and already-immutable types are preserved.
    """
    if isinstance(data, (dict, MappingProxyType)):
        return MappingProxyType({k: _freeze_container(v) for k, v in data.items()})
    elif isinstance(data, (list, tuple)):
        return tuple(_freeze_container(v) for v in data)
    elif isinstance(data, (set, frozenset)):
        return frozenset(_freeze_container(v) for v in data)
    return data


def _unfreeze_container(data: Any) -> Any:
    """Recursively unfreeze containers for serialization and JSON encoding.

    Guarantees:
    - MappingProxyType and dicts become standard dicts.
    - Tuples and lists become standard lists.
    - Frozensets and sets become standard lists.
    - Nested structures are deeply unfrozen into mutable primitives.
    """
    if isinstance(data, (MappingProxyType, dict)):
        return {k: _unfreeze_container(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list, frozenset, set)):
        return [_unfreeze_container(v) for v in data]
    return data
