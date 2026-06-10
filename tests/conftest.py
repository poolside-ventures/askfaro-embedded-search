"""Shared test fixtures: a deterministic embedder so the hybrid pipeline is
exercised end-to-end without any model provider."""

from __future__ import annotations

import hashlib
import math
import re

import pytest

from faro_search import CallableEmbedder

# Wide enough that md5-bucket collisions between unrelated tokens are
# vanishingly rare; collisions at small dims caused spurious semantic hits.
DIM = 512


def bow_vector(text: str) -> list[float]:
    """Deterministic bag-of-words hash embedding: token overlap → cosine
    similarity, which is all the pipeline mechanics need."""
    vec = [0.0] * DIM
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.md5(token.encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


@pytest.fixture
def embedder() -> CallableEmbedder:
    return CallableEmbedder(lambda texts: [bow_vector(t) for t in texts])
