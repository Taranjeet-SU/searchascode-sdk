"""search-as-code — one API, any vector database.

A unified "search as code" agentic harness: agents write portable Python against
a single primitive API, executed in a sandbox with intermediate state kept out
of the model context, regardless of which vector DB is underneath.

Quickstart:

    import search_as_code as sac

    s = sac.Session("memory")                       # any backend, same call
    s.add([{"id": "1", "text": "hello world"}])
    hits = s.search("hello", top_k=3)
    print(hits.to_evidence())
"""

from .adapters import MemoryStore, VectorStore, available, connect, register
from .embeddings import Embedder, HashEmbedder, as_embedder, get_embedder
from .primitives import (
    dedup, decompose, expand, extract, fan_out, freshness, fuse, mmr, rerank,
)
from .sandbox import ExecResult, LocalExecutor, Sandbox
from .session import Session, route
from .types import Capabilities, Document, Hit, ResultSet

__version__ = "0.0.1"

__all__ = [
    # data model
    "Document", "Hit", "ResultSet", "Capabilities",
    # backends
    "connect", "register", "available", "VectorStore", "MemoryStore",
    # embeddings
    "Embedder", "HashEmbedder", "as_embedder", "get_embedder",
    # harness
    "Session", "route", "Sandbox", "LocalExecutor", "ExecResult",
    # primitives
    "fan_out", "fuse", "dedup", "rerank", "freshness", "extract",
    "mmr", "expand", "decompose",
]
