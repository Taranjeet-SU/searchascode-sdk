"""SQLite adapter — the "you don't even need a vector DB" reference.

Vectors are stored as float32 BLOBs in an ordinary SQLite table; dense search is
brute-force cosine in numpy. Persistent, zero-server, stdlib-only. Keyword/regex
run over the stored text. Proves the primitive API works over a plain SQL store.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional, Sequence

import numpy as np

from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore
from .memory import MemoryStore


class SqliteStore(VectorStore):
    backend = "sqlite"

    def __init__(self, dim: int, path: str = ":memory:", **_: Any):
        self.dim = int(dim)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS docs "
            "(id TEXT PRIMARY KEY, text TEXT, meta TEXT, vec BLOB)"
        )
        self.db.commit()
        self._mem = MemoryStore()  # keyword/regex + fast doc cache
        self._mat: Optional[np.ndarray] = None
        self._ids: list[str] = []
        self._dirty = True

    def capabilities(self) -> Capabilities:
        return Capabilities(dense=True, keyword=True, hybrid=True, regex=True,
                            server_side_embedding=False, native_rerank=False,
                            metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        import json
        rows = []
        for d in docs:
            vec = None if d.vector is None else np.asarray(d.vector, dtype=np.float32).tobytes()
            rows.append((d.id, d.text or "", json.dumps(d.metadata or {}), vec))
        self.db.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)", rows)
        self.db.commit()
        self._mem.upsert(docs)
        self._dirty = True

    def _rebuild(self) -> None:
        self._ids, vecs = [], []
        for did, blob in self.db.execute("SELECT id, vec FROM docs WHERE vec IS NOT NULL"):
            self._ids.append(did)
            vecs.append(np.frombuffer(blob, dtype=np.float32))
        if vecs:
            mat = np.vstack(vecs)
            n = np.linalg.norm(mat, axis=1, keepdims=True)
            n[n == 0] = 1.0
            self._mat = mat / n
        else:
            self._mat = None
        self._dirty = False

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        if self._dirty:
            self._rebuild()
        if self._mat is None:
            return ResultSet()
        from ..filters import matches
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._mat @ q
        order = np.argsort(-sims)
        hits = []
        for i in order:
            doc = self._mem._docs.get(self._ids[i])
            if doc is None or (flt and not matches(doc.metadata, flt)):
                continue
            hits.append(Hit(id=self._ids[i], score=float(sims[i]), document=doc, store=self.backend))
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
        return self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def close(self) -> None:
        self.db.close()
