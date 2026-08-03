"""Shared helpers for the BrowseComp-Plus 3-arm benchmark (INTERNAL / local only).

Provides:
  - paths / qrels + query loading
  - FastMemoryStore: a MemoryStore subclass that builds the BM25-ish keyword
    index ONCE (the stock MemoryStore re-tokenizes every doc on every keyword
    query, which is unusable at ~100K docs).
"""
from __future__ import annotations

import json
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Optional

# BrowseComp-Plus docs are long (~33KB avg). A pure-Python BM25 over the full
# 3.3GB of text is intractable, so the keyword index uses each doc's first
# KW_CHARS characters. Dense is unaffected (full precomputed gte-base vectors).
KW_CHARS = 2000

import numpy as np

from search_as_code.adapters.memory import MemoryStore
from search_as_code.embeddings import _tokenize
from search_as_code.filters import matches
from search_as_code.types import Document, Hit, ResultSet

HERE = Path(__file__).parent
BC_REPO = Path("/tmp/claude-1001/-home-taranjeet-bakshi-code-search-harness/"
               "e0637587-783d-4571-898c-6a1bbc3b84c4/scratchpad/BrowseComp-Plus")
QREL_GOLDS = BC_REPO / "topics-qrels" / "qrel_golds.txt"
QUERIES_JSONL = HERE / "queries_decrypted.jsonl"

VECS_NPY = HERE / "corpus_vecs.npy"
IDS_JSON = HERE / "corpus_ids.json"
TEXTS_JSONL = HERE / "corpus_texts.jsonl"   # {"id":..., "text":...} per line


def load_golds() -> dict[str, list[str]]:
    golds: dict[str, list[str]] = {}
    for line in QREL_GOLDS.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
        if rel > 0:
            golds.setdefault(qid, []).append(docid)
    return golds


def load_queries() -> dict[str, str]:
    q: dict[str, str] = {}
    for line in QUERIES_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        q[str(row["query_id"])] = row.get("query", "")
    return q


class FastMemoryStore(MemoryStore):
    """MemoryStore with a precomputed keyword index (df + per-doc token counts)."""

    backend = "memory"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._kw_ready = False
        self._df: Counter = Counter()
        self._toks: dict[str, Counter] = {}
        self._dl: dict[str, int] = {}
        self._n = 0
        self._kw_lock = threading.Lock()

    def build_kw(self) -> None:
        with self._kw_lock:
            if self._kw_ready:
                return
            docs = [d for d in self._docs.values() if d.text]
            self._n = len(docs) or 1
            df: Counter = Counter()
            toks_by_id: dict[str, Counter] = {}
            dl: dict[str, int] = {}
            for d in docs:
                toks = Counter(_tokenize((d.text or "")[:KW_CHARS]))
                toks_by_id[d.id] = toks
                dl[d.id] = sum(toks.values()) or 1
                for t in toks:
                    df[t] += 1
            self._df, self._toks, self._dl = df, toks_by_id, dl
            self._kw_ready = True

    def query_keyword(self, text, top_k=10, flt=None) -> ResultSet:
        if not self._kw_ready:
            self.build_kw()
        terms = Counter(_tokenize(text))
        if not terms:
            return ResultSet()
        scored = []
        for did, toks in self._toks.items():
            d = self._docs[did]
            if not matches(d.metadata, flt):
                continue
            dl = self._dl[did]
            score = 0.0
            for term, qf in terms.items():
                tf = toks.get(term)
                if tf:
                    idf = math.log(1 + self._n / (1 + self._df[term]))
                    score += qf * idf * (tf / dl)
            if score > 0:
                scored.append(Hit(id=did, score=score, document=d, store=self.backend))
        scored.sort(key=lambda h: h.score, reverse=True)
        return ResultSet(scored[:top_k])


def gte_embedder():
    """gte-base query embedder — MUST match the corpus vectors' model."""
    import torch
    from sentence_transformers import SentenceTransformer

    from phase1 import common

    em = SentenceTransformer(common.EMB_MODEL,
                             device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    return embed


def load_session(generator=None):
    """Load precomputed corpus vectors + texts into a FastMemoryStore Session.

    Docs are added WITH their precomputed gte-base vectors so nothing is re-embedded;
    the session's query-side embedder is gte-base so query vectors match the corpus.
    """
    import search_as_code as sac

    vecs = np.load(VECS_NPY)
    ids = json.loads(IDS_JSON.read_text())
    texts = {}
    for line in TEXTS_JSONL.open():
        row = json.loads(line)
        texts[row["id"]] = row["text"]
    store = FastMemoryStore()
    docs = [
        Document(id=str(i), text=texts.get(str(i), ""), vector=vecs[k].tolist())
        for k, i in enumerate(ids)
    ]
    store.upsert(docs)
    sess = sac.Session(store, embedder=gte_embedder(), generator=generator)
    return sess
