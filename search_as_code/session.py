"""The Session is the single handle agent-generated code writes against.

It binds a backend + embedder + optional reranker/extractor, exposes the
primitives as methods, and — crucially — owns a *state store* that lives outside
the model context.  Agent code fans out, filters, and stashes bulky results in
the session; only ``.to_evidence()`` output ever needs to return to the model.

    s = Session(connect("qdrant", ...), embedder=my_embedder)
    seeds = s.search_many(["q1", "q2", "q3"], top_k=8).dedup()
    s.remember("seeds", seeds)
    evidence = seeds.top(5).to_evidence(fields=["title", "url"])
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from . import primitives as P
from .adapters.base import VectorStore
from .adapters.registry import connect
from .embeddings import Embedder, HashEmbedder, as_embedder
from .filters import normalize
from .types import Document, Hit, ResultSet


class Session:
    def __init__(
        self,
        store: VectorStore | str,
        embedder: Optional[Embedder | Callable] = None,
        reranker: Optional[Callable[[str, list[str]], list[float]]] = None,
        extractor: Optional[Callable[[list[str], dict, str], list[dict]]] = None,
        generator: Optional[Callable[[str], list[str]]] = None,
        **connect_opts: Any,
    ):
        self.store = connect(store, **connect_opts) if isinstance(store, str) else store
        self._caps = self.store.capabilities()
        self.embedder: Embedder = as_embedder(embedder) if embedder else HashEmbedder()
        self.reranker = reranker
        self.extractor = extractor
        self.generator = generator  # LLM callable(prompt) -> list[str], for query-side primitives
        self._state: dict[str, Any] = {}

    # ---- ingestion -------------------------------------------------------
    def add(self, docs: Sequence[Document | dict]) -> None:
        """Upsert documents, embedding any that lack a vector."""
        norm = [d if isinstance(d, Document) else Document(**d) for d in docs]
        missing = [d for d in norm if d.vector is None and d.text]
        if missing and not self._caps.server_side_embedding:
            vecs = self.embedder.embed([d.text for d in missing])  # type: ignore[arg-type]
            for d, v in zip(missing, vecs):
                d.vector = v
        self.store.upsert(norm)

    # ---- core search primitives (capability-aware) -----------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[dict] = None,
        mode: str = "dense",
    ) -> ResultSet:
        """Single search. ``mode`` is dense | keyword | hybrid; unsupported modes
        are emulated in-SDK so agent code is identical on every backend."""
        flt = normalize(filter)
        if mode == "keyword":
            return self._keyword(query, top_k, flt)
        if mode == "hybrid":
            return self._hybrid(query, top_k, flt)
        if mode == "regex":
            return self._regex(query, top_k, flt)
        vec = self._embed_query(query)
        return self.store.query_vector(vec, top_k=top_k, flt=flt)

    def search_many(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        filter: Optional[dict] = None,
        mode: str = "dense",
        concurrency: int = 8,
        fuse: bool = True,
    ) -> ResultSet:
        """Fan out over many queries concurrently, then RRF-fuse (or return the
        flat union). The primitive that replaces N serial model turns."""
        sets = P.fan_out(lambda q: self.search(q, top_k, filter, mode), queries, concurrency)
        for q, rs in zip(queries, sets):
            for h in rs:
                h.query = q
        if fuse:
            return P.fuse(sets).top(top_k * 2)
        flat = ResultSet()
        for rs in sets:
            flat.extend(rs)
        return flat

    # ---- portable helpers over ResultSets --------------------------------
    def rerank(self, query: str, results: ResultSet, top_k: Optional[int] = None) -> ResultSet:
        return P.rerank(query, results, reranker=self.reranker, top_k=top_k)

    def fuse(self, sets: Sequence[ResultSet], weights: Optional[Sequence[float]] = None) -> ResultSet:
        return P.fuse(sets, weights)

    def dedup(self, results: ResultSet, key: Optional[Callable[[Hit], Any]] = None) -> ResultSet:
        return results.dedup(key)

    def extract(self, results: ResultSet, schema: dict, instruction: str) -> list[dict]:
        return P.extract(results, schema, instruction, extractor=self.extractor)

    def mmr(self, query: str, results: ResultSet, lambda_: float = 0.5, top_k: int = 10) -> ResultSet:
        """Diversify results with Maximal Marginal Relevance (embeds ``query``)."""
        results = self.hydrate(results)
        return P.mmr(self._embed_query(query), results, lambda_=lambda_, top_k=top_k)

    diversify = mmr  # alias

    def compress(self, query: str, results: ResultSet, keep: int = 2, per_hit: bool = True) -> ResultSet:
        """Contextual compression — keep only the sentences in each hit most
        relevant to ``query`` (LangChain contextual compression, but model-free:
        it scores sentences with the embedder). Shrinks what returns to the model.
        """
        qv = self._embed_query(query)
        import numpy as np

        q = np.asarray(qv, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        out = ResultSet()
        for h in results:
            if not h.text:
                out.append(h)
                continue
            sents = [s.strip() for s in _split_sentences(h.text) if s.strip()]
            if len(sents) <= keep:
                out.append(h)
                continue
            svecs = self.embedder.embed(sents)
            scores = []
            for s, v in zip(sents, svecs):
                vv = np.asarray(v, dtype=np.float32)
                scores.append((float(q @ (vv / (np.linalg.norm(vv) or 1.0))), s))
            top = [s for _, s in sorted(scores, reverse=True)[:keep]]
            kept = " ".join(s for s in sents if s in top)  # preserve reading order
            doc = h.document.with_metadata() if h.document else Document(id=h.id)
            doc.text = kept
            out.append(Hit(id=h.id, score=h.score, document=doc, query=h.query, store=h.store))
        return out

    # ---- pluggable query-side primitives (need a generator/LLM) -----------
    def expand_search(self, query: str, top_k: int = 10, n: int = 4, mode: str = "dense") -> ResultSet:
        """Multi-query expansion → fan-out → RRF fuse (RAG-Fusion)."""
        variants = P.expand(query, self._require_generator(), n=n)
        return self.search_many(variants, top_k=top_k, mode=mode)

    def decompose_search(self, query: str, top_k: int = 10, mode: str = "dense") -> ResultSet:
        """Decompose into sub-questions → fan-out → RRF fuse."""
        subs = P.decompose(query, self._require_generator())
        return self.search_many(subs or [query], top_k=top_k, mode=mode)

    def rephrase_search(self, query: str, top_k: int = 10, mode: str = "dense") -> ResultSet:
        """Rewrite the query with the LLM, then search with the improved form."""
        better = P.rephrase(query, self._require_generator())
        return self.search(better, top_k=top_k, mode=mode)

    def hyde_search(self, query: str, top_k: int = 10) -> ResultSet:
        """HyDE — generate a hypothetical answer document, embed it, and search
        with that vector instead of the bare query (Gao et al. 2023). Reaches the
        answer region of embedding space, a DIFFERENT neighborhood than the query."""
        gen = self._require_generator()
        prompt = f"Write a short passage that directly answers this query.\nQuery: {query}"
        doc = (gen(prompt) or [query])[0]
        vec = self.embedder.embed([doc])[0]
        return self.store.query_vector(vec, top_k=top_k)

    def prf_search(self, query: str, top_k: int = 10, feedback_k: int = 5,
                   alpha: float = 1.0, beta: float = 0.7, filter: Optional[dict] = None) -> ResultSet:
        """Pseudo-relevance feedback (Rocchio): retrieve, then MOVE the query vector
        toward the centroid of the top ``feedback_k`` retrieved docs and re-search.
        This literally shifts the query into a NEW embedding neighborhood (the region
        where the pseudo-relevant docs live) — reaching neighbors a paraphrase can't.
        """
        import numpy as np

        flt = normalize(filter)
        qv = np.asarray(self._embed_query(query), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        seed = self.hydrate(self.store.query_vector(qv, top_k=feedback_k, flt=flt))
        texts = [h.text for h in seed if h.text]
        if not texts:
            return self.store.query_vector(qv, top_k=top_k, flt=flt)
        dvecs = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        new_q = alpha * qv + beta * dvecs.mean(axis=0)
        new_q = new_q / (np.linalg.norm(new_q) or 1.0)
        return self.store.query_vector(new_q, top_k=top_k, flt=flt)

    def hydrate(self, results: ResultSet) -> ResultSet:
        """Fetch full documents for hits that only carry ids (e.g. after fusion
        across a store that returned partial payloads)."""
        need = [h.id for h in results if h.document is None]
        if need:
            byid = {d.id: d for d in self.store.get(need)}
            for h in results:
                if h.document is None:
                    h.document = byid.get(h.id)
        return results

    # ---- out-of-context state store --------------------------------------
    def remember(self, key: str, value: Any) -> None:
        self._state[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def forget(self, key: str) -> None:
        self._state.pop(key, None)

    def state_keys(self) -> list[str]:
        return list(self._state)

    # ---- internals -------------------------------------------------------
    def _embed_query(self, query: str) -> list[float]:
        if self._caps.server_side_embedding:
            v = self.store.embed_query(query)
            if v is not None:
                return v
        return self.embedder.embed([query])[0]

    def _keyword(self, query: str, top_k: int, flt: dict) -> ResultSet:
        if self._caps.keyword:
            return self.store.query_keyword(query, top_k=top_k, flt=flt)
        # Emulate: dense recall then lexical rerank.
        pool = self.store.query_vector(self._embed_query(query), top_k=top_k * 5, flt=flt)
        return P.rerank(query, pool, top_k=top_k)

    def _hybrid(self, query: str, top_k: int, flt: dict) -> ResultSet:
        vec = self._embed_query(query)
        if self._caps.hybrid:
            return self.store.query_hybrid(vec, query, top_k=top_k, flt=flt)
        dense = self.store.query_vector(vec, top_k=top_k * 3, flt=flt)
        kw = self._keyword(query, top_k * 3, flt)
        return P.fuse([dense, kw]).top(top_k)

    def _regex(self, pattern: str, top_k: int, flt: dict) -> ResultSet:
        if self._caps.regex:
            return self.store.query_regex(pattern, top_k=top_k, flt=flt)
        # Emulate client-side: pull a metadata-filtered pool and scan text.
        import re

        rx = re.compile(pattern)
        pool = self.store.query_vector(self._embed_query(pattern), top_k=max(top_k * 20, 200), flt=flt)
        pool = self.hydrate(pool)
        hits = [h for h in pool if h.text and rx.search(h.text)]
        return ResultSet(hits[:top_k])

    def _require_generator(self) -> Callable[[str], list[str]]:
        if self.generator is None:
            raise RuntimeError(
                "This primitive needs an LLM. Pass Session(..., generator=fn) where "
                "fn(prompt: str) -> list[str]."
            )
        return self.generator


def route(
    sessions: Sequence[Session],
    query: str,
    top_k: int = 10,
    mode: str = "dense",
    concurrency: int = 8,
) -> ResultSet:
    """Multi-store retrieval: fan the same query across several backends and
    RRF-fuse the results (query routing / federated search).  Each hit keeps its
    ``store`` tag so you can see which backend it came from."""
    sets = P.fan_out(lambda s: s.search(query, top_k, mode=mode), list(sessions), concurrency)
    return P.fuse(sets).top(top_k)


def _split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[.!?])\s+", text)
