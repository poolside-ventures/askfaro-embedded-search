"""Per-object-type indexer registry.

Heterogeneity is handled in code, not schema: each application object type
registers a function that turns its objects into one or more IndexDocs
(typically a leaf plus an optional summary node).
"""

from __future__ import annotations

from typing import Any, Callable

from .types import IndexDoc

IndexerFn = Callable[[Any], IndexDoc | list[IndexDoc]]

_registry: dict[str, IndexerFn] = {}


def register(object_type: str) -> Callable[[IndexerFn], IndexerFn]:
    def decorator(fn: IndexerFn) -> IndexerFn:
        _registry[object_type] = fn
        return fn

    return decorator


def docs_for(object_type: str, obj: Any) -> list[IndexDoc]:
    fn = _registry.get(object_type)
    if fn is None:
        raise KeyError(f"no indexer registered for object_type={object_type!r}")
    result = fn(obj)
    return result if isinstance(result, list) else [result]


def registered_types() -> list[str]:
    return sorted(_registry)
