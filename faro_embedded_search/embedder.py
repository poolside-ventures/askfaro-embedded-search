"""Embedder protocol and built-in implementations.

The library never talks to a model provider directly from the core — apps
inject an embedder. Embedding failures are non-fatal by contract: a row
without a vector still participates in lexical retrieval and gains semantic
retrieval after a later backfill pass.
"""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Return one vector per input text; None entries mark failures."""
        ...


class CallableEmbedder:
    """Wrap a (sync or async) function `texts -> vectors` as an Embedder."""

    def __init__(
        self,
        fn: Callable[[Sequence[str]], list[list[float] | None] | Awaitable[list[list[float] | None]]],
    ):
        self._fn = fn

    async def embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        result = self._fn(texts)
        if inspect.isawaitable(result):
            result = await result
        return result


class OpenAICompatibleEmbedder:
    """Embedder for any OpenAI-compatible `/embeddings` endpoint.

    Covers OpenAI itself, LiteLLM proxies, and most self-hosted servers.
    Requires the `http` extra (httpx).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        *,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        try:
            import httpx
        except ImportError as e:
            from .errors import MissingDependencyError

            raise MissingDependencyError(
                "OpenAICompatibleEmbedder", "http", "httpx"
            ) from e

        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(texts)},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        # The API may reorder; index field is authoritative.
        out: list[list[float] | None] = [None] * len(texts)
        for item in data:
            out[item["index"]] = item["embedding"]
        return out
