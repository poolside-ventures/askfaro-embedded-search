"""Reciprocal Rank Fusion shared by all backends.

Fusion operates on ranks, never on raw scores. Postgres `ts_rank_cd` and
SQLite FTS5 `bm25()` are not on comparable scales (nor are different vector
stores' similarity outputs), but their *orderings* are — which is what makes
retrieval semantics identical across server and on-device backends.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .types import RawHit, SearchResult

RRF_K = 60  # canonical constant from Cormack et al. (2009)


def rrf_fuse(
    lexical: Sequence[RawHit],
    semantic: Sequence[RawHit],
    *,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse two ranked candidate lists into one, ordered by RRF score."""
    lexical_ranks = {hit.key: i + 1 for i, hit in enumerate(lexical)}
    semantic_ranks = {hit.key: i + 1 for i, hit in enumerate(semantic)}
    semantic_scores = {hit.key: hit.sim for hit in semantic if hit.sim is not None}

    by_key: dict[tuple[str, str, str], RawHit] = {}
    for hit in lexical:
        by_key.setdefault(hit.key, hit)
    for hit in semantic:
        by_key.setdefault(hit.key, hit)

    fused: list[SearchResult] = []
    for key, hit in by_key.items():
        lex_rank = lexical_ranks.get(key)
        sem_rank = semantic_ranks.get(key)
        score = 0.0
        if lex_rank is not None:
            score += 1.0 / (rrf_k + lex_rank)
        if sem_rank is not None:
            score += 1.0 / (rrf_k + sem_rank)
        if lex_rank is not None and sem_rank is not None:
            match_type = "hybrid"
        elif lex_rank is not None:
            match_type = "keyword"
        else:
            match_type = "semantic"
        fused.append(
            SearchResult(
                object_type=hit.object_type,
                object_id=hit.object_id,
                node_kind=hit.node_kind,
                partition=hit.partition,
                title=hit.title,
                payload=hit.payload,
                score=score,
                match_type=match_type,
                lexical_rank=lex_rank,
                semantic_rank=sem_rank,
                semantic_score=semantic_scores.get(key),
            )
        )

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def collapse_objects(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Collapse multiple node kinds of the same object into its best row.

    Summary/cluster hits act as additional retrieval handles for an object;
    after fusion the caller usually wants one entry per object with a record
    of which kinds matched.
    """
    best: dict[tuple[str, str], SearchResult] = {}
    order: list[tuple[str, str]] = []
    for r in results:
        obj_key = (r.object_type, r.object_id)
        existing = best.get(obj_key)
        if existing is None:
            best[obj_key] = r
            r.matched_node_kinds = [r.node_kind]
            order.append(obj_key)
        else:
            existing.matched_node_kinds.append(r.node_kind)
            if r.score > existing.score:
                r.matched_node_kinds = existing.matched_node_kinds
                best[obj_key] = r
    return [best[k] for k in order]


def diversify(
    results: Sequence[SearchResult],
    *,
    key: Callable[[SearchResult], str],
    cap: int,
    limit: int,
) -> list[SearchResult]:
    """Cap results per group (e.g. per object_type or namespace).

    Overflow is dropped, not deferred — a slightly shorter list beats one
    padded with near-duplicate siblings.
    """
    out: list[SearchResult] = []
    counts: dict[str, int] = {}
    for r in results:
        group = key(r)
        if counts.get(group, 0) >= cap:
            continue
        counts[group] = counts.get(group, 0) + 1
        out.append(r)
        if len(out) >= limit:
            break
    return out
