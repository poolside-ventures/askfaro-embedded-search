"""Keyword channel (index_text vs lexical_body split) + precision/explore modes."""

import pytest

from askfaro_embedded_search.fusion import rrf_fuse
from askfaro_embedded_search.index import SEARCH_MODES
from askfaro_embedded_search.types import IndexDoc, RawHit


def test_keywords_excluded_from_embedded_text_included_in_lexical():
    doc = IndexDoc(object_type="doc", object_id="1", title="Invoices",
                   body="How to send a bill.", keywords=["dunning", "receivable", "AR"])
    # embedded field stays the human-reading prose (no keyword dilution)
    assert "dunning" not in doc.index_text()
    assert "How to send a bill." in doc.index_text()
    # lexical/FTS body carries the keyword channel
    assert "dunning" in doc.lexical_body() and "receivable" in doc.lexical_body()


def test_lexical_body_none_when_empty():
    assert IndexDoc(object_type="d", object_id="1").lexical_body() is None


def _hit(oid, sim=None):
    return RawHit(object_type="d", object_id=oid, node_kind="leaf", partition=None,
                  title=oid, payload=None, sim=sim)


def test_precision_weight_favors_lexical_only_hit():
    lexical = [_hit("only_lexical")]
    semantic = [_hit("only_semantic", sim=0.9)]
    # precision weights lexical 2x -> the lexical-only hit outranks the semantic-only one
    fused = rrf_fuse(lexical, semantic, lexical_weight=2.0, semantic_weight=1.0)
    assert fused[0].object_id == "only_lexical"
    # explore flips it
    fused = rrf_fuse(lexical, semantic, lexical_weight=1.0, semantic_weight=2.0)
    assert fused[0].object_id == "only_semantic"


def test_modes_table_shape():
    assert SEARCH_MODES["precision"][0] > SEARCH_MODES["precision"][1]   # lexical-leaning
    assert SEARCH_MODES["explore"][1] > SEARCH_MODES["explore"][0]        # semantic-leaning
    assert SEARCH_MODES["precision"][2] > SEARCH_MODES["explore"][2]      # tighter floor
