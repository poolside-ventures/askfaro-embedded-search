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
from .errors import ConfigurationError
from .fusion import collapse_objects as _collapse
from .fusion import diversify as _diversify
from .fusion import rrf_fuse
from .types import Filters, IndexDoc, SearchResult, utcnow_iso

# Floor below which a cosine match is noise, not signal (carried over from
# Faro's production tuning).
DEFAULT_MIN_SEMANTIC_SCORE = 0.20

DEFAULT_SPACE = "default"


class SearchIndex:
    """Wire a backend to one or more embedders and get incremental
    upsert/delete plus hybrid search.

    Single-model (the common case):

        SearchIndex(backend, embedder)            # one "default" space

    Multi-model (e.g. a hosted server model + an on-device model):

        SearchIndex(backend, embedders={"server": e1, "local": e2},
                    default_space="server")

    The backend must be configured with matching space names. All ranking
    logic lives here and in `fusion`, so any two backends return
    identically-ranked results for the same corpus, query, and space.
    """

    def __init__(
        self,
        backend: Backend,
        embedder: Embedder | None = None,
        *,
        embedders: dict[str, Embedder] | None = None,
        default_space: str = DEFAULT_SPACE,
        query_cache_size: int = 256,
    ):
        if embedders is None:
            embedders = {default_space: embedder} if embedder is not None else {}
        backend_spaces = set(getattr(backend, "spaces", []) or [])
        unknown = set(embedders) - backend_spaces
        if backend_spaces and unknown:
            raise ConfigurationError(
                f"SearchIndex was given embedders for space(s) {sorted(unknown)} that "
                f"the backend doesn't define (backend spaces: {sorted(backend_spaces)}).\n"
                f"  Configure the backend with matching spaces, e.g.\n"
                f"    PostgresBackend(..., spaces={{'{next(iter(unknown))}': 1536}})\n"
                f"    SQLiteBackend(..., spaces={tuple(sorted(unknown))!r})"
            )
        self.backend = backend
        self.embedders = embedders
        self.default_space = default_space
        # One query-embedding cache per space.
        self._query_cache: dict[str, OrderedDict[str, list[float]]] = {
            space: OrderedDict() for space in embedders
        }
        self._query_cache_size = query_cache_size

    # -- indexing ---------------------------------------------------------

    async def upsert(self, doc: IndexDoc) -> None:
        await self.upsert_many([doc])

    async def upsert_many(self, docs: Sequence[IndexDoc]) -> None:
        """Index documents incrementally — one upsert per doc, no rebuilds.

        Each configured embedding space is embedded for the docs that opt into
        it (`IndexDoc.embed_spaces`, default all). Embedding failure is
        non-fatal: the row is written without that vector and still serves
        lexical queries until a backfill re-embeds it.
        """
        texts = [doc.index_text() for doc in docs]
        # space -> per-doc vector (None where the doc opted out or embed failed)
        space_vectors: dict[str, list[list[float] | None]] = {}
        for space, embedder in self.embedders.items():
            idxs = [
                i
                for i, doc in enumerate(docs)
                if doc.embed_spaces is None or space in doc.embed_spaces
            ]
            vectors: list[list[float] | None] = [None] * len(docs)
            if idxs:
                try:
                    embedded = await embedder.embed([texts[i] for i in idxs])
                except Exception:
                    embedded = [None] * len(idxs)
                for j, i in enumerate(idxs):
                    vectors[i] = embedded[j] if j < len(embedded) else None
            space_vectors[space] = vectors

        for i, doc in enumerate(docs):
            embeddings = {
                space: space_vectors[space][i]
                for space in self.embedders
                if space_vectors[space][i] is not None
            }
            indexed_at = utcnow_iso() if embeddings else None
            await self.backend.upsert_row(doc, embeddings, indexed_at)

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
        space: str | None = None,
        partition: str | None = None,
        object_types: list[str] | None = None,
        node_kinds: list[str] | None = None,
        attrs: dict | None = None,
        min_semantic_score: float = DEFAULT_MIN_SEMANTIC_SCORE,
        collapse: bool = True,
        diversity_key: Callable[[SearchResult], str] | None = None,
        diversity_cap: int = 2,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        space = space or self.default_space
        filters = Filters(
            partition=partition,
            object_types=object_types,
            node_kinds=node_kinds,
            attrs=attrs,
        )
        # Pull more candidates per retriever than we return — RRF benefits
        # from seeing each retriever's fuller ranked list.
        candidates = max(k * 3, 30)

        lexical = await self.backend.lexical_search(query, filters, candidates)
        semantic = []
        query_vec = await self._embed_query(space, query)
        if query_vec is not None:
            semantic = await self.backend.semantic_search(
                query_vec, space, filters, candidates, min_semantic_score
            )

        results = rrf_fuse(lexical, semantic)
        if collapse:
            results = _collapse(results)
        if diversity_key is not None:
            return _diversify(results, key=diversity_key, cap=diversity_cap, limit=k)
        return results[:k]

    async def _embed_query(self, space: str, query: str) -> list[float] | None:
        embedder = self.embedders.get(space)
        if embedder is None:
            return None
        cache = self._query_cache.setdefault(space, OrderedDict())
        cached = cache.get(query)
        if cached is not None:
            cache.move_to_end(query)
            return cached
        try:
            vectors = await embedder.embed([query])
        except Exception:
            return None
        vec = vectors[0] if vectors else None
        if vec is not None:
            cache[query] = vec
            while len(cache) > self._query_cache_size:
                cache.popitem(last=False)
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
