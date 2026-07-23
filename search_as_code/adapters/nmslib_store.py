"""nmslib adapter — in-process HNSW ANN, no server.

nmslib builds its graph in one shot (not incremental), so vectors are accumulated
on upsert and the index is built lazily on first query. Dense is native HNSW;
keyword/regex/hybrid are served by a composed MemoryStore. Integer point ids map
to the original doc ids (like the FAISS adapter).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore
from .memory import MemoryStore


def _norm(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


class NmslibStore(VectorStore):
    backend = "nmslib"

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200,
                 ef_search: int = 200, **_: Any):
        import nmslib

        self._nmslib = nmslib
        self.dim = int(dim)
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []
        self._mem = MemoryStore()
        self._index = None
        self._built = False
        self.params = {"M": M, "efConstruction": ef_construction}
        self.ef_search = ef_search

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=True, hybrid=True, regex=True,
                            server_side_embedding=False, native_rerank=False,
                            metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        for d in docs:
            if d.vector is None:
                continue
            self._ids.append(d.id); self._vecs.append(d.vector)
        self._mem.upsert(docs)
        self._built = False

    def _build(self) -> None:
        idx = self._nmslib.init(method="hnsw", space="cosinesimil")
        data = _norm(np.asarray(self._vecs, dtype=np.float32))
        idx.addDataPointBatch(data, list(range(len(self._ids))))
        idx.createIndex(self.params, print_progress=False)
        idx.setQueryTimeParams({"efSearch": self.ef_search})
        self._index = idx
        self._built = True

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        if not self._ids:
            return ResultSet()
        if not self._built:
            self._build()
        from ..filters import matches
        q = _norm(np.asarray([vector], dtype=np.float32))[0]
        k = min(len(self._ids), top_k * (8 if flt else 1))
        pos, dist = self._index.knnQuery(q, k=k)
        hits = []
        for p, dd in zip(pos, dist):
            doc = self._mem._docs.get(self._ids[p])
            if doc is None or (flt and not matches(doc.metadata, flt)):
                continue
            # cosinesimil distance = 1 - cosine -> larger-is-better score
            hits.append(Hit(id=self._ids[p], score=float(1.0 - dd), document=doc, store=self.backend))
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
        return len(self._ids)
