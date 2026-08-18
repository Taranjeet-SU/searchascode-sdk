"""Zero-dependency reference adapter.

Brute-force cosine over an in-process dict plus a naive BM25-ish keyword score.
It exists so the entire harness — primitives, session, sandbox, tests, demos —
runs with nothing installed but numpy.  Treat it as the executable spec every
real adapter must match.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np

from ..embeddings import _tokenize
from ..filters import matches
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore


class MemoryStore(VectorStore):
    backend = "memory"

    def __init__(self, dim: Optional[int] = None, **_: Any):
        # `dim` is accepted (not swallowed by **_) so Session._check_dims can enforce the
        # declared width on the DEFAULT backend used by tests, demos and the SU experiments.
        # It previously found neither `dim` nor `_dim`, so only intra-batch consistency was
        # checked — which is what forced the SAC_DIM env hack in qwen8b_sac/issues.md #2
        # (SDK-C10).
        self.dim = dim
        self._docs: dict[str, Document] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []
        self._dirty = True
        # keyword index (SDK-C11) — built lazily, invalidated on write
        self._kw_lock = threading.Lock()
        self._kw_df: Optional[Counter] = None
        self._kw_toks: dict[str, Counter] = {}
        self._kw_dl: dict[str, int] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities(
            dense=True,
            keyword=True,
            hybrid=True,
            regex=True,
            server_side_embedding=False,
            native_rerank=False,
            metadata_filter=True,
        )

    def upsert(self, docs: Sequence[Document]) -> None:
        for d in docs:
            self._docs[d.id] = d
        self._dirty = True
        self._invalidate_kw_index()

    def _rebuild(self) -> None:
        self._ids = [d.id for d in self._docs.values() if d.vector is not None]
        if self._ids:
            mat = np.array([self._docs[i].vector for i in self._ids], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = mat / norms
        else:
            self._matrix = None
        self._dirty = False

    def _candidates(self, flt: Optional[dict[str, Any]]) -> list[int]:
        if not flt:
            return list(range(len(self._ids)))
        return [i for i, did in enumerate(self._ids) if matches(self._docs[did].metadata, flt)]

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        if self._dirty:
            self._rebuild()
        if self._matrix is None:
            return ResultSet()
        q = np.asarray(vector, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        sims = self._matrix @ (q / qn)
        cand = self._candidates(flt)
        cand.sort(key=lambda i: sims[i], reverse=True)
        hits = [
            Hit(id=self._ids[i], score=float(sims[i]), document=self._docs[self._ids[i]], store=self.backend)
            for i in cand[:top_k]
        ]
        return ResultSet(hits)

    def _ensure_kw_index(self) -> None:
        """Build (once) the document-frequency table + per-doc term counts.

        Previously ``query_keyword`` re-tokenized **every document on every call** — O(N·tokens)
        per keyword search with no index and no caching, which is unusable at ~100K docs and
        made ``explore/multihop`` (n_docs-1 keyword searches per candidate chain) do thousands
        of full-corpus scans (SDK-C11). The working version of this fix already existed in
        ``experiments/browsecomp/bc_common.py::FastMemoryStore``; this promotes it into the SDK
        (BC-3). The index is invalidated on upsert/delete.
        """
        with self._kw_lock:
            if self._kw_df is not None:
                return
            df: Counter = Counter()
            toks_by_id: dict[str, Counter] = {}
            dl_by_id: dict[str, int] = {}
            for d in self._docs.values():
                if not d.text:
                    continue
                toks = Counter(_tokenize(d.text))
                toks_by_id[d.id] = toks
                dl_by_id[d.id] = sum(toks.values()) or 1
                for t in toks:
                    df[t] += 1
            self._kw_df, self._kw_toks, self._kw_dl = df, toks_by_id, dl_by_id

    def _invalidate_kw_index(self) -> None:
        with self._kw_lock:
            self._kw_df = None

    def query_keyword(self, text, top_k=10, flt=None) -> ResultSet:
        terms = Counter(_tokenize(text))
        if not terms:
            return ResultSet()
        self._ensure_kw_index()
        df, toks_by_id, dl_by_id = self._kw_df, self._kw_toks, self._kw_dl
        assert df is not None
        n = len(toks_by_id) or 1
        scored = []
        for doc_id, toks in toks_by_id.items():
            d = self._docs.get(doc_id)
            if d is None or not matches(d.metadata, flt):
                continue
            dl = dl_by_id[doc_id]
            score = 0.0
            for term, qf in terms.items():
                tf = toks.get(term)
                if tf:
                    idf = math.log(1 + n / (1 + df[term]))
                    score += qf * idf * (tf / dl)
            if score > 0:
                scored.append(Hit(id=doc_id, score=score, document=d, store=self.backend))
        scored.sort(key=lambda h: h.score, reverse=True)
        return ResultSet(scored[:top_k])

    def query_hybrid(self, vector, text, top_k=10, flt=None, alpha=0.5) -> ResultSet:
        dense = self.query_vector(vector, top_k=top_k * 4, flt=flt)
        kw = self.query_keyword(text, top_k=top_k * 4, flt=flt)
        from ..primitives import fuse  # local import to avoid cycle

        return fuse([dense, kw], weights=[alpha, 1 - alpha]).top(top_k)

    def query_regex(self, pattern, top_k=10, flt=None) -> ResultSet:
        import re

        rx = re.compile(pattern)
        hits = []
        for d in self._docs.values():
            if not d.text or not matches(d.metadata, flt):
                continue
            found = rx.findall(d.text)
            if found:
                hits.append(Hit(id=d.id, score=float(len(found)), document=d, store=self.backend))
        hits.sort(key=lambda h: h.score, reverse=True)
        return ResultSet(hits[:top_k])

    def get(self, ids: Sequence[str]) -> list[Document]:
        return [self._docs[i] for i in ids if i in self._docs]

    def delete(self, ids: Sequence[str]) -> None:
        for i in ids:
            self._docs.pop(i, None)
        self._dirty = True
        self._invalidate_kw_index()

    def count(self) -> int:
        return len(self._docs)

    def sample(self, n: int = 5) -> list[Document]:
        # deterministic (first-n) so corpus_fingerprint / resume stay stable; the OpenSearch
        # adapter samples randomly server-side, and the qrels path uses real pairs directly.
        return list(self._docs.values())[:n]

    def describe_schema(self) -> dict:
        docs = self.sample(3)
        keys: set = set()
        for d in docs:
            keys |= set((d.metadata or {}).keys())
        return {"backend": self.backend, "count": len(self._docs),
                "metadata_keys": sorted(keys), "sample_text": [(d.text or "")[:200] for d in docs]}
