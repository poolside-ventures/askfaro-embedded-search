"""SearchIndex — the public facade.

Wire a backend (where rows live) to an embedder (how text becomes vectors)
and get incremental upsert/delete plus hybrid search. All ranking logic
lives here and in `fusion`, so any two backends return identically-ranked
results for the same corpus and query.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Sequence

from .backends.base import Backend, ChangeRow
from .embedder import Embedder
from .fusion import collapse_objects as _collapse
from .fusion import diversify as _diversify
from .fusion import rrf_fuse
from .types import Filters, IndexDoc, SearchResult, utcnow_iso

# Floor below which a cosine match is noise, not signal (carried over from
# Faro's production tuning).
DEFAULT_MIN_SEMANTIC_SCORE = 0.20


class SearchIndex:
    def __init__(
        self,
        backend: Backend,
        embedder: Embedder | None = None,
        *,
        query_cache_size: int = 256,
    ):
        self.backend = backend
        self.embedder = embedder
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_size = query_cache_size

    # -- indexing ---------------------------------------------------------

    async def upsert(self, doc: IndexDoc) -> None:
        await self.upsert_many([doc])

    async def upsert_many(self, docs: Sequence[IndexDoc]) -> None:
        """Index documents incrementally — one upsert per doc, no rebuilds.

        Embedding failure is non-fatal: the row is written without a vector
        and serves lexical queries until a backfill re-embeds it.
        """
        vectors: list[list[float] | None] = [None] * len(docs)
        if self.embedder is not None:
            texts = [doc.index_text() for doc in docs]
            try:
                vectors = await self.embedder.embed(texts)
            except Exception:
                vectors = [None] * len(docs)
        for doc, vec in zip(docs, vectors):
            indexed_at = utcnow_iso() if vec is not None else None
            await self.backend.upsert_row(doc, vec, indexed_at)

    async def delete(
        self, object_type: str, object_id: str, node_kind: str | None = None
    ) -> None:
        await self.backend.delete_row(object_type, object_id, node_kind)

    # -- retrieval --------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        k: int = 10,
        partition: str | None = None,
        object_types: list[str] | None = None,
        node_kinds: list[str] | None = None,
        min_semantic_score: float = DEFAULT_MIN_SEMANTIC_SCORE,
        collapse: bool = True,
        diversity_key: Callable[[SearchResult], str] | None = None,
        diversity_cap: int = 2,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        filters = Filters(
            partition=partition, object_types=object_types, node_kinds=node_kinds
        )
        # Pull more candidates per retriever than we return — RRF benefits
        # from seeing each retriever's fuller ranked list.
        candidates = max(k * 3, 30)

        lexical = await self.backend.lexical_search(query, filters, candidates)
        semantic = []
        query_vec = await self._embed_query(query)
        if query_vec is not None:
            semantic = await self.backend.semantic_search(
                query_vec, filters, candidates, min_semantic_score
            )

        results = rrf_fuse(lexical, semantic)
        if collapse:
            results = _collapse(results)
        if diversity_key is not None:
            return _diversify(results, key=diversity_key, cap=diversity_cap, limit=k)
        return results[:k]

    async def _embed_query(self, query: str) -> list[float] | None:
        if self.embedder is None:
            return None
        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached
        try:
            vectors = await self.embedder.embed([query])
        except Exception:
            return None
        vec = vectors[0] if vectors else None
        if vec is not None:
            self._query_cache[query] = vec
            while len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return vec

    # -- sync (pass-throughs for shard replication) ------------------------

    async def changes_since(
        self, cursor: int | None, *, partition: str | None = None, limit: int = 500
    ) -> tuple[list[ChangeRow], int | None]:
        return await self.backend.changes_since(cursor, partition=partition, limit=limit)

    async def apply_changes(self, rows: Sequence[ChangeRow]) -> None:
        await self.backend.apply_changes(rows)

    async def close(self) -> None:
        await self.backend.close()
