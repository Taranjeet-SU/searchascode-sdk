"""FAISS adapter — in-process ANN, no server.

FAISS handles dense (exact inner-product on L2-normalized vectors == cosine); the
keyword/hybrid/regex capabilities are served client-side by a composed
:class:`MemoryStore` so agent code behaves identically to a full backend. This is
the "one primitive API, any vector DB" thesis for an embedded index.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore
from .memory import MemoryStore


def _norm(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


class FaissStore(VectorStore):
    backend = "faiss"

    def __init__(self, dim: int, **_: Any):
        import faiss

        self.dim = int(dim)
        self.index = faiss.IndexFlatIP(self.dim)  # exact cosine (vectors normalized)
        self._ids: list[str] = []
        self._mem = MemoryStore()  # holds docs + keyword/regex/text

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=True, hybrid=True, regex=True,
                            server_side_embedding=False, native_rerank=False,
                            metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        """Upsert by id. IndexFlatIP has no in-place update, so re-inserting a known id
        rebuilds the index.

        This used to unconditionally ``index.add``, while the composed MemoryStore replaced
        the document — so re-upserting an id appended a SECOND vector, ``count()`` drifted
        away from the real document count, and the stale vector stayed searchable forever.
        Caught by the conformance suite (issues.md ADP-1).
        """
        vecs, keep = [], []
        for d in docs:
            if d.vector is None:
                continue
            vecs.append(d.vector)
            keep.append(d)
        self._mem.upsert(docs)
        known = set(self._ids)
        if any(d.id in known for d in keep):
            self._rebuild()
            return
        if vecs:
            self.index.add(_norm(np.asarray(vecs, dtype=np.float32)))
            self._ids.extend(d.id for d in keep)

    def _rebuild(self) -> None:
        """Rebuild the flat index from the documents the MemoryStore currently holds."""
        import faiss

        self.index = faiss.IndexFlatIP(self.dim)
        self._ids = []
        rows, ids = [], []
        for doc_id, d in self._mem._docs.items():
            if d.vector is None:
                continue
            rows.append(d.vector)
            ids.append(doc_id)
        if rows:
            self.index.add(_norm(np.asarray(rows, dtype=np.float32)))
        self._ids = ids

    def delete(self, ids: Sequence[str]) -> None:
        """Delete by id (was unimplemented, so it raised NotImplementedError — ADP-2)."""
        self._mem.delete(ids)
        self._rebuild()

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        if self.index.ntotal == 0:
            return ResultSet()
        q = _norm(np.asarray([vector], dtype=np.float32))
        # over-fetch when filtering, then apply the portable filter client-side
        k = min(self.index.ntotal, top_k * (8 if flt else 1))
        scores, idx = self.index.search(q, k)
        from ..filters import matches
        hits = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            doc = self._mem._docs.get(self._ids[i])
            if doc is None or (flt and not matches(doc.metadata, flt)):
                continue
            hits.append(Hit(id=self._ids[i], score=float(s), document=doc, store=self.backend))
            if len(hits) >= top_k:
                break
        return ResultSet(hits)

    def query_keyword(self, text, top_k=10, flt=None) -> ResultSet:
        return self._mem.query_keyword(text, top_k=top_k, flt=flt)

    def query_regex(self, pattern, top_k=10, flt=None) -> ResultSet:
        return self._mem.query_regex(pattern, top_k=top_k, flt=flt)

    def query_hybrid(self, vector, text, top_k=10, flt=None, alpha=0.5) -> ResultSet:
        dense = self.query_vector(vector, top_k=top_k * 4, flt=flt)
        kw = self.query_keyword(text, top_k=top_k * 4, flt=flt)
        from ..primitives import fuse
        return fuse([dense, kw], weights=[alpha, 1 - alpha]).top(top_k)

    def get(self, ids: Sequence[str]) -> list[Document]:
        return self._mem.get(ids)

    def count(self) -> int:
        return int(self.index.ntotal)
