"""faro-embedded-search: incremental hybrid retrieval, server-side and on-device."""

from .embedder import CallableEmbedder, Embedder, OpenAICompatibleEmbedder
from .errors import ConfigurationError, FaroSearchError, MissingDependencyError
from .fusion import RRF_K, rrf_fuse
from .index import DEFAULT_MIN_SEMANTIC_SCORE, SearchIndex
from .registry import docs_for, register, registered_types
from .sync import export_shard, replicate
from .types import (
    NODE_KIND_CLUSTER,
    NODE_KIND_LEAF,
    NODE_KIND_SUMMARY,
    Filters,
    IndexDoc,
    RawHit,
    SearchResult,
)

__version__ = "0.4.0"

__all__ = [
    "CallableEmbedder",
    "ConfigurationError",
    "DEFAULT_MIN_SEMANTIC_SCORE",
    "Embedder",
    "FaroSearchError",
    "Filters",
    "MissingDependencyError",
    "IndexDoc",
    "NODE_KIND_CLUSTER",
    "NODE_KIND_LEAF",
    "NODE_KIND_SUMMARY",
    "OpenAICompatibleEmbedder",
    "RRF_K",
    "RawHit",
    "SearchIndex",
    "SearchResult",
    "docs_for",
    "export_shard",
    "register",
    "registered_types",
    "replicate",
    "rrf_fuse",
]
