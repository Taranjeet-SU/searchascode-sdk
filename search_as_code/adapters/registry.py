"""Backend registry — ``connect(backend, **opts)`` returns a uniform store.

Real adapters are imported lazily so their heavy client libraries are only
required when actually used. Registering a custom backend is one call:

    from search_as_code.adapters import register
    register("mystore", MyStore)
"""

from __future__ import annotations

from typing import Any, Callable

from ..errors import BackendNotFoundError
from .base import VectorStore
from .memory import MemoryStore

# Lazy factories keep optional deps out of the base install.
_LAZY: dict[str, Callable[..., VectorStore]] = {}


def _load_qdrant(**opts: Any) -> VectorStore:
    from .qdrant import QdrantStore

    return QdrantStore(**opts)


def _load_chroma(**opts: Any) -> VectorStore:
    from .chroma import ChromaStore

    return ChromaStore(**opts)


def _load_pgvector(**opts: Any) -> VectorStore:
    from .pgvector import PgVectorStore

    return PgVectorStore(**opts)


def _load_opensearch(**opts: Any) -> VectorStore:
    from .opensearch import OpenSearchStore

    return OpenSearchStore(**opts)


def _load_faiss(**opts: Any) -> VectorStore:
    from .faiss_store import FaissStore

    return FaissStore(**opts)


def _load_sqlite(**opts: Any) -> VectorStore:
    from .sqlite_store import SqliteStore

    return SqliteStore(**opts)


def _load_nmslib(**opts: Any) -> VectorStore:
    from .nmslib_store import NmslibStore

    return NmslibStore(**opts)


def _load_milvus(**opts: Any) -> VectorStore:
    from .milvus_store import MilvusStore

    return MilvusStore(**opts)


_REGISTRY: dict[str, Callable[..., VectorStore]] = {
    "memory": MemoryStore,
    "qdrant": _load_qdrant,
    "chroma": _load_chroma,
    "pgvector": _load_pgvector,
    "opensearch": _load_opensearch,
    "faiss": _load_faiss,
    "sqlite": _load_sqlite,
    "nmslib": _load_nmslib,
    "milvus": _load_milvus,
}


def register(name: str, factory: Callable[..., VectorStore]) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def connect(backend: str = "memory", **opts: Any) -> VectorStore:
    try:
        factory = _REGISTRY[backend]
    except KeyError:
        raise BackendNotFoundError(
            "unknown backend", backend=backend, available=available()
        ) from None
    return factory(**opts)
