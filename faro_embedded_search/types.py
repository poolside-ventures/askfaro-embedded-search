"""Core data types shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Node kinds form the tiering vocabulary. Leaves are raw objects; summary
# and cluster nodes are enrichment rows that live in the same flat pool and
# are retrieved by the same top-k (no tree traversal at query time).
NODE_KIND_LEAF = "leaf"
NODE_KIND_SUMMARY = "summary"
NODE_KIND_CLUSTER = "cluster"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IndexDoc:
    """One indexable unit: a (object_type, object_id, node_kind) triple.

    `payload` is opaque display metadata stored alongside the index row so
    consumers (e.g. an on-device shard) can render results without joining
    back to the application database.
    """

    object_type: str
    object_id: str
    title: str | None = None
    body: str | None = None
    node_kind: str = NODE_KIND_LEAF
    partition: str | None = None
    payload: dict[str, Any] | None = None
    # Structured fields to filter on at query time (e.g. {"category": "finance",
    # "status": "active"}). Stored as JSON; matched by equality / containment.
    attrs: dict[str, Any] | None = None
    source_updated_at: str = field(default_factory=utcnow_iso)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.object_type, self.object_id, self.node_kind)

    def index_text(self) -> str:
        parts = [p for p in (self.title, self.body) if p]
        return "\n".join(parts)


@dataclass
class RawHit:
    """A candidate row as returned by a backend retriever (pre-fusion)."""

    object_type: str
    object_id: str
    node_kind: str
    partition: str | None
    title: str | None
    payload: dict[str, Any] | None
    sim: float | None = None  # populated by the semantic retriever only

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.object_type, self.object_id, self.node_kind)


@dataclass
class SearchResult:
    object_type: str
    object_id: str
    node_kind: str
    partition: str | None
    title: str | None
    payload: dict[str, Any] | None
    score: float
    match_type: str  # "keyword" | "semantic" | "hybrid"
    lexical_rank: int | None
    semantic_rank: int | None
    semantic_score: float | None
    matched_node_kinds: list[str] = field(default_factory=list)


@dataclass
class Filters:
    """Backend-agnostic query filters."""

    partition: str | None = None
    object_types: list[str] | None = None
    node_kinds: list[str] | None = None
    attrs: dict[str, Any] | None = None  # equality match on stored IndexDoc.attrs
