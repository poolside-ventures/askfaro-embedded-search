"""Setup/configuration errors should be clear and actionable."""

import pytest

from faro_embedded_search import (
    CallableEmbedder,
    ConfigurationError,
    IndexDoc,
    MissingDependencyError,
    SearchIndex,
)
from faro_embedded_search.backends.sqlite import SQLiteBackend


def _embedder(dim=8):
    return CallableEmbedder(lambda texts: [[0.1] * dim for _ in texts])


async def test_embedder_for_unknown_space_is_rejected():
    # Backend only knows "local"; configuring an embedder for "server" is a
    # silent footgun (its vectors would have nowhere to go) — reject loudly.
    backend = SQLiteBackend(spaces=("local",))
    with pytest.raises(ConfigurationError) as ei:
        SearchIndex(backend, embedders={"server": _embedder()})
    msg = str(ei.value)
    assert "server" in msg and "spaces" in msg
    await backend.close()


async def test_missing_dependency_error_message():
    # The message must name the package and the exact install command.
    err = MissingDependencyError("PostgresBackend", "postgres", "sqlalchemy / asyncpg")
    msg = str(err)
    assert 'pip install "faro-embedded-search[postgres]"' in msg
    assert "sqlalchemy" in msg
    # It's both a FaroSearchError and an ImportError so existing `except
    # ImportError` handlers still catch it.
    assert isinstance(err, ImportError)


async def test_single_embedder_default_space_ok():
    # The common single-model path must not trip the space-mismatch check.
    backend = SQLiteBackend()  # default space
    idx = SearchIndex(backend, _embedder())
    await idx.upsert(IndexDoc(object_type="note", object_id="1", title="hello world"))
    assert await idx.search("hello")
    await idx.close()
