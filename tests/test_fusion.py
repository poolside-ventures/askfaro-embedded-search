from faro_embedded_search.fusion import RRF_K, collapse_objects, diversify, rrf_fuse
from faro_embedded_search.types import RawHit


def hit(oid: str, kind: str = "leaf", sim: float | None = None) -> RawHit:
    return RawHit(
        object_type="note",
        object_id=oid,
        node_kind=kind,
        partition=None,
        title=oid,
        payload=None,
        sim=sim,
    )


def test_rrf_scores_and_match_types():
    lexical = [hit("a"), hit("b")]
    semantic = [hit("b", sim=0.9), hit("c", sim=0.5)]
    results = rrf_fuse(lexical, semantic)

    by_id = {r.object_id: r for r in results}
    assert by_id["a"].match_type == "keyword"
    assert by_id["b"].match_type == "hybrid"
    assert by_id["c"].match_type == "semantic"
    # b appears in both lists, so it must outrank single-retriever hits.
    assert results[0].object_id == "b"
    assert abs(by_id["b"].score - (1 / (RRF_K + 2) + 1 / (RRF_K + 1))) < 1e-9
    assert by_id["c"].semantic_score == 0.5


def test_collapse_keeps_best_row_and_records_kinds():
    lexical = [hit("a", kind="summary")]
    semantic = [hit("a", kind="leaf", sim=0.9), hit("a", kind="summary", sim=0.8)]
    collapsed = collapse_objects(rrf_fuse(lexical, semantic))

    assert len(collapsed) == 1
    assert collapsed[0].node_kind == "summary"  # hybrid summary beats semantic-only leaf
    assert set(collapsed[0].matched_node_kinds) == {"leaf", "summary"}


def test_diversify_caps_per_group():
    results = rrf_fuse([hit("a1"), hit("a2"), hit("a3"), hit("b1")], [])
    capped = diversify(
        results, key=lambda r: r.object_id[0], cap=2, limit=10
    )
    assert [r.object_id for r in capped] == ["a1", "a2", "b1"]
