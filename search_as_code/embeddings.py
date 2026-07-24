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

from .errors import ConfigurationError, InvalidEmbedderError, MissingDependencyError


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def as_embedder(obj: "Embedder | Callable[[Sequence[str]], list[list[float]]]") -> Embedder:
    """Coerce a plain callable into the :class:`Embedder` protocol."""
    if isinstance(obj, Embedder):
        return obj
    if callable(obj):
        return _CallableEmbedder(obj)
    raise InvalidEmbedderError(
        "embedder must be an Embedder or a callable(list[str]) -> vectors",
        got=type(obj).__name__,
    )


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
    if provider in ("transformers", "hf", "sentence-transformers"):
        return _TransformersEmbedder(**kwargs)
    raise ConfigurationError("unknown embedding provider", provider=provider)


class _TransformersEmbedder:
    """Robust loader for HF ``transformers`` embedding models — including custom GTE-v1.5 /
    ``model_type="new"`` models whose meta-device init corrupts non-persistent buffers
    (``position_ids``, rotary ``cos_cached``/``sin_cached``) and causes a GPU device-assert
    or NaN outputs. Fixes: load with ``low_cpu_mem_usage=False``, disable the memory-efficient
    ``unpad`` attention path, and re-materialize those buffers ON THE MODEL DEVICE.

        emb = get_embedder("transformers", model="my/custom-gte", pooling="cls")
        vecs = emb.embed(["a query"])
    """

    def __init__(self, model: str, device: str | None = None, pooling: str = "cls",
                 max_length: int = 512, token: str | None = None, trust_remote_code: bool = True):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.max_length = max_length
        conf = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, token=token)
        for attr in ("unpad_inputs", "use_memory_efficient_attention"):
            if hasattr(conf, attr):
                setattr(conf, attr, False)
        self.tok = AutoTokenizer.from_pretrained(model, token=token)
        self.model = AutoModel.from_pretrained(
            model, config=conf, trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False, token=token).to(self.device).eval()
        self._fix_meta_buffers(conf)

    def _fix_meta_buffers(self, conf) -> None:
        torch, dev = self._torch, self.device
        maxp = getattr(conf, "max_position_embeddings", 512)
        for m in self.model.modules():
            if hasattr(m, "position_ids"):
                m.register_buffer("position_ids", torch.arange(maxp, device=dev), persistent=False)
            if hasattr(m, "cos_cached") and hasattr(m, "inv_freq"):
                L = int(m.cos_cached.shape[0])
                t = torch.arange(L, dtype=torch.float32, device=dev)
                freqs = torch.outer(t, m.inv_freq.float().to(dev))
                emb = torch.cat((freqs, freqs), dim=-1)
                m.register_buffer("cos_cached", emb.cos(), persistent=False)
                m.register_buffer("sin_cached", emb.sin(), persistent=False)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        torch = self._torch
        out = []
        for text in texts:
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            v = h[:, 0] if self.pooling == "cls" else h.mean(dim=1)
            v = torch.nn.functional.normalize(v, p=2, dim=-1)
            out.append(v[0].detach().cpu().tolist())
        return out


class _OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", **client_kwargs):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise MissingDependencyError("openai", extra="search-as-code[providers]") from e
        self._client = OpenAI(**client_kwargs)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - network
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]
