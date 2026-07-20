"""Embedding strategy is bring-your-own by default.

Pass any callable ``list[str] -> list[list[float]]`` (or an object with an
``embed`` method) wherever an embedder is accepted.  Optional thin wrappers for
common providers are lazy-imported so the base install stays light.

``HashEmbedder`` is a dependency-free, deterministic embedder used by the
in-memory adapter, tests, and examples so the whole harness runs with no API
key.  It is *not* semantically meaningful — swap in a real provider for quality.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def as_embedder(obj: "Embedder | Callable[[Sequence[str]], list[list[float]]]") -> Embedder:
    """Coerce a plain callable into the :class:`Embedder` protocol."""
    if isinstance(obj, Embedder):
        return obj
    if callable(obj):
        return _CallableEmbedder(obj)
    raise TypeError("embedder must be an Embedder or a callable(list[str]) -> vectors")


class _CallableEmbedder:
    def __init__(self, fn: Callable[[Sequence[str]], list[list[float]]]):
        self._fn = fn

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._fn(texts)


class HashEmbedder:
    """Deterministic hashing embedder — zero dependencies, no network.

    Uses hashed token bucketing (a poor-man's bag-of-words) so that texts
    sharing tokens land near each other.  Good enough to exercise the full
    retrieval path in tests and demos.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def get_embedder(provider: str, **kwargs) -> Embedder:
    """Factory for optional built-in providers (lazy-imported)."""
    provider = provider.lower()
    if provider in ("hash", "test"):
        return HashEmbedder(**kwargs)
    if provider == "openai":
        return _OpenAIEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


class _OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", **client_kwargs):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install 'search-as-code[providers]' to use OpenAI embeddings") from e
        self._client = OpenAI(**client_kwargs)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - network
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]
