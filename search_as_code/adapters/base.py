"""The one interface every vector database must satisfy.

This ABC is the contract that makes "no separate SDK per DB" possible: agent
code and the primitive layer only ever touch these methods.  Adapters translate
them to their backend's native client.

Design rules for adapters:
* Return :class:`Hit` with **larger-is-better** scores (convert distances).
* Accept the portable filter dialect from :mod:`search_as_code.filters`.
* Declare honestly via :meth:`capabilities`; the primitive layer emulates any
  capability you report as ``False`` (e.g. client-side rerank or keyword search)
  so agent code behaves the same everywhere.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Sequence

from ..types import Capabilities, Document, ResultSet


class VectorStore(abc.ABC):
    #: registry key, e.g. "memory", "qdrant"
    backend: str = "base"

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def upsert(self, docs: Sequence[Document]) -> None: ...

    @abc.abstractmethod
    def query_vector(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        flt: Optional[dict[str, Any]] = None,
    ) -> ResultSet:
        """Dense nearest-neighbour search."""

    # ---- optional capabilities: default to NotImplemented so the primitive
    # layer knows to emulate. Adapters override when the backend supports them.

    def query_keyword(
        self,
        text: str,
        top_k: int = 10,
        flt: Optional[dict[str, Any]] = None,
    ) -> ResultSet:
        raise NotImplementedError

    def query_hybrid(
        self,
        vector: Sequence[float],
        text: str,
        top_k: int = 10,
        flt: Optional[dict[str, Any]] = None,
        alpha: float = 0.5,
    ) -> ResultSet:
        raise NotImplementedError

    def query_regex(
        self,
        pattern: str,
        top_k: int = 10,
        flt: Optional[dict[str, Any]] = None,
    ) -> ResultSet:
        """Exact / regex / operator search over document text.

        The "search as code for code" primitive (Hornet, Chroma): agents need
        exact, case-sensitive matching that semantic search blurs away.
        """
        raise NotImplementedError

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Server-side embedding, if the backend does it. Returns None otherwise."""
        return None

    @abc.abstractmethod
    def get(self, ids: Sequence[str]) -> list[Document]: ...

    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    # ---- introspection: let the agent discover the DATA SHAPE before querying.
    # "The first call tells the agent what data exists and what queries are safe to
    # construct" (schema-first agentic retrieval). Adapters override with real detail.

    def describe_schema(self) -> dict[str, Any]:
        """Field names -> types (+ sample values where cheap), doc count, backend.
        Feed this to the LLM so it knows the fields/types before writing retrieval code."""
        try:
            n = self.count()
        except Exception:
            n = None
        return {"backend": self.backend, "count": n, "fields": {}, "note": "override for full schema"}

    def sample(self, n: int = 5) -> list[Document]:
        """A few representative documents so the agent sees the data shape (prose vs
        table vs fact-card, typical length, metadata)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
