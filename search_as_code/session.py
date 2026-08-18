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

from . import filters as F
from . import primitives as P
from ._genutil import gen_text
from .adapters.base import VectorStore
from .adapters.registry import connect
from .embeddings import Embedder, HashEmbedder, as_embedder
from .errors import (
    DimensionMismatchError,
    GeneratorRequiredError,
    InvalidArgumentError,
    InvalidModeError,
)
from .filters import normalize
from .types import Capabilities, Document, Hit, ResultSet

_MODES = ("dense", "keyword", "hybrid", "regex")


def _check_query(query: Any) -> None:
    if not isinstance(query, str) or not query.strip():
        raise InvalidArgumentError("query must be a non-empty string", query=query)


def _check_top_k(top_k: Any, caps: Capabilities) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise InvalidArgumentError("top_k must be a positive integer", top_k=top_k)
    if caps.max_top_k is not None and top_k > caps.max_top_k:
        raise InvalidArgumentError(
            "top_k exceeds backend maximum", top_k=top_k, max_top_k=caps.max_top_k
        )


def _check_mode(mode: Any) -> None:
    if mode not in _MODES:
        raise InvalidModeError("unknown search mode", mode=mode, allowed=list(_MODES))


class Session:
    def __init__(
        self,
        store: VectorStore | str,
        embedder: Optional[Embedder | Callable] = None,
        reranker: Optional[Callable[[str, list[str]], list[float]]] = None,
        extractor: Optional[Callable[[list[str], dict, str], list[dict]]] = None,
        generator: Optional[Callable[[str], list[str]]] = None,
        query_embedder: Optional[Embedder | Callable] = None,
        **connect_opts: Any,
    ):
        # ``query_embedder`` — asymmetric encoding (P2-5): bge/e5/Qwen-style models need a
        # DIFFERENT prefix/instruction for queries than for passages; getting this wrong costs
        # more than every augmentation measured in this repo (Qwen3-8B BrowseComp R@10 0.149
        # plain vs 0.277 instructed). ``embedder`` embeds passages (add / hyde docs / dedup);
        # ``query_embedder`` embeds search queries; defaults to ``embedder`` (symmetric).
        self.store = connect(store, **connect_opts) if isinstance(store, str) else store
        self._caps = self.store.capabilities()
        self.embedder: Embedder = as_embedder(embedder) if embedder else HashEmbedder()
        self.query_embedder: Embedder = (as_embedder(query_embedder) if query_embedder
                                         else self.embedder)
        self.reranker = reranker
        self.extractor = extractor
        self.generator = generator  # LLM callable(prompt) -> list[str], for query-side primitives
        self._state: dict[str, Any] = {}

    # ---- ingestion -------------------------------------------------------
    def add(self, docs: Sequence[Document | dict]) -> None:
        """Upsert documents, embedding any that lack a vector."""
        norm = [d if isinstance(d, Document) else Document(**d) for d in docs]
        for d in norm:
            if not d.id:
                raise InvalidArgumentError("every document needs a non-empty id")
        missing = [d for d in norm if d.vector is None and d.text]
        if missing and not self._caps.server_side_embedding:
            vecs = self.embedder.embed([d.text or "" for d in missing])
            for d, v in zip(missing, vecs):
                d.vector = v
        self._check_dims(norm)
        self.store.upsert(norm)

    def describe(self, n_samples: int = 4, llm: bool = False) -> dict:
        """Corpus profile for the LLM (introspect BEFORE writing retrieval code):
        the store's schema (fields/types/count) + the content-type mix of a sample
        (prose vs table vs fact-card vs list) + short sample snippets.

        Feed this into the agent prompt so it knows *what kind of data* it is querying —
        the schema-first agentic-retrieval pattern.

        The base profile is **sampled + heuristic** (real random sample from the store,
        rule-based ``content_type`` tagging). Pass ``llm=True`` to add a **model-generated**
        characterization: the generator reads the sample and returns a free-text summary of
        the data (types, key entities/fields) plus recommended primitives — under
        ``profile["llm"]``. Needs a generator on the Session.
        """
        from .primitives import content_type
        try:
            schema = self.store.describe_schema()
        except Exception:
            schema = {"backend": getattr(self.store, "backend", "?")}
        try:
            samples = self.store.sample(n_samples)
        except NotImplementedError:
            samples = []
        types: dict[str, int] = {}
        snippets = []
        for d in samples:
            ct = content_type(d.text or "")
            types[ct] = types.get(ct, 0) + 1
            snippets.append({"type": ct, "text": (d.text or "")[:160]})
        profile = {**schema, "content_types": types, "samples": snippets}
        if llm and self.generator is not None:
            profile["llm"] = self._llm_profile(schema, samples)
        return profile

    def _llm_profile(self, schema: dict, samples: Sequence[Document]) -> str:
        """Ask the generator to characterize the corpus from the sample — genuinely
        LLM-driven data exploration (vs the heuristic content_type tagging)."""
        fields = schema.get("fields") or schema.get("metadata_keys") or {}
        rows = "\n".join(f"- {(d.text or '')[:300]}" for d in samples) or "(no text samples)"
        prompt = (
            "You are profiling a search corpus before writing retrieval code.\n"
            f"Backend: {schema.get('backend') or schema.get('index')}  "
            f"Approx docs: {schema.get('count')}\n"
            f"Fields: {fields}\n\n"
            f"Random sample of documents:\n{rows}\n\n"
            "In 4-6 short lines describe: (1) what kind of data this is (prose docs, tables, "
            "curated fact-cards, code, mixed?), (2) the key entities/fields a query would target, "
            "(3) which retrieval primitives fit best (e.g. keyword/exact & regex for part-numbers "
            "and fact-cards, dense/hyde for prose, fielded/phrase for structured fields)."
        )
        try:
            out = self.generator(prompt)  # type: ignore[misc]  # guarded by caller (generator is not None)
            # gen_text, not out[0]: a line-splitting generator adapter would otherwise return
            # line 1 of a 4-6 line profile — dropping item (3), the recommended primitives (GEN-1).
            return gen_text(out)
        except Exception as e:  # pragma: no cover - profiling is best-effort
            return f"(llm profile unavailable: {e})"

    def _check_dims(self, docs: Sequence[Document]) -> None:
        """Guardrail: all supplied/embedded vectors must share one dimension, and
        match the backend's declared ``dim`` when it has one."""
        vecs = [d for d in docs if d.vector is not None]
        if not vecs:
            return
        want = getattr(self.store, "dim", None) or getattr(self.store, "_dim", None)
        seen = len(vecs[0].vector)  # type: ignore[arg-type]
        for d in vecs:
            n = len(d.vector)  # type: ignore[arg-type]
            if n != seen or (want is not None and n != want):
                raise DimensionMismatchError(
                    "vector dimensionality mismatch", id=d.id, dim=n,
                    expected=want if want is not None else seen,
                )

    # ---- core search primitives (capability-aware) -----------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[dict] = None,
        mode: str = "dense",
        alpha: float = 0.5,
    ) -> ResultSet:
        """Single search. ``mode`` is dense | keyword | hybrid | regex; unsupported
        modes are emulated in-SDK so agent code is identical on every backend.

        ``alpha`` (hybrid only) is the dense weight in the dense↔keyword fusion:
        alpha=1.0 → pure dense, 0.0 → pure keyword, 0.8 → dense-dominant (best on FiQA)."""
        _check_query(query)
        _check_mode(mode)
        _check_top_k(top_k, self._caps)
        F.validate(filter)
        flt = normalize(filter)
        if mode == "keyword":
            return self._keyword(query, top_k, flt)
        if mode == "hybrid":
            return self._hybrid(query, top_k, flt, alpha)
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
        if not queries:
            raise InvalidArgumentError("queries must be a non-empty sequence")
        if concurrency < 1:
            raise InvalidArgumentError("concurrency must be >= 1", concurrency=concurrency)
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

    def smart_search(self, query: str, top_k: int = 10) -> ResultSet:
        """Spelling/acronym-normalize the query, and if it carries rare exact tokens
        (acronyms, quoted phrases, numbers) boost them via a keyword pass fused with
        dense — otherwise fall back to dense. Targets the exact-token miss slice."""
        q = P.normalize_query(query)
        terms = P.rare_terms(query)
        dense = self.search(q, top_k=top_k * 4, mode="dense")
        if not terms:
            return dense.top(top_k)
        kwb = self.search(" ".join(terms), top_k=top_k * 4, mode="keyword")
        return P.fuse([dense, kwb], weights=[0.6, 0.4]).dedup().top(top_k)

    def retrieve_rerank(self, query: str, pool_k: int = 500, top_k: int = 10,
                        mode: str = "dense") -> ResultSet:
        """Wide-pool retrieve → rerank → top_k. The near-miss recovery pipeline
        (gold at rank 100-500 pulled forward by the reranker). Needs a reranker set."""
        pool = self.search(query, top_k=pool_k, mode=mode)
        return self.rerank(query, self.hydrate(pool), top_k=top_k)

    def adaptive_search(self, query: str, mode: str = "dense", max_k: int = 100,
                        min_k: int = 10, rel_band: float = 0.1, method: str = "band",
                        filter: Optional[dict] = None) -> ResultSet:
        """Retrieve a wide pool then size it by the score distribution: return MORE
        results when similarity stays flat past the top (many near-equally-relevant
        docs), FEWER when it drops off. Recovers gold that a hard top-10 cut misses
        (higher recall-in-context) without a reranker. See primitives.score_cutoff."""
        pool = self.search(query, top_k=max_k, mode=mode, filter=filter)
        return P.score_cutoff(pool, method=method, rel_band=rel_band, min_k=min_k, max_k=max_k)

    def semantic_dedup(self, results: ResultSet, threshold: float = 0.85) -> ResultSet:
        """Collapse near-duplicate results (cosine >= threshold on their embeddings),
        keeping one representative per semantic cluster — distinct from exact-id dedup.
        (SemDeDup.) Uses the session embedder on hit texts."""
        import numpy as np

        hits = sorted(results, key=lambda h: h.score, reverse=True)
        texts = [h.text or "" for h in hits]
        if not texts:
            return ResultSet()
        v = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        kept: list[Hit] = []
        kv: list = []
        for h, vec in zip(hits, v):
            if kv and max(float(vec @ k) for k in kv) >= threshold:
                continue
            kept.append(h)
            kv.append(vec)
        return ResultSet(kept)

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

    def rephrase(self, query: str) -> str:
        """LLM rewrite of the query (returns the improved string, does not search)."""
        return P.rephrase(query, self._require_generator())

    def rephrase_search(self, query: str, top_k: int = 10, mode: str = "dense") -> ResultSet:
        """Rewrite the query with the LLM, then search with the improved form."""
        better = P.rephrase(query, self._require_generator())
        return self.search(better, top_k=top_k, mode=mode)

    def topics(self, query: str, n: int = 5) -> list[str]:
        """LLM-extracted key topics/entities in the query (for routing or filtering)."""
        return P.topics(query, self._require_generator(), n=n)

    def auto_filter(self, query: str, fields: Optional[Sequence[str]] = None) -> dict:
        """Self-query: LLM infers a metadata filter implied by the query. Feed the
        result to search(filter=...). Pass ``fields`` (the metadata keys in your
        corpus) to constrain what the LLM may filter on."""
        return P.auto_filter(query, self._require_generator(), fields=fields)

    def hyde_search(self, query: str, top_k: int = 10) -> ResultSet:
        """HyDE — generate a hypothetical answer document, embed it, and search
        with that vector instead of the bare query (Gao et al. 2023). Reaches the
        answer region of embedding space, a DIFFERENT neighborhood than the query."""
        gen = self._require_generator()
        prompt = f"Write a short passage that directly answers this query.\nQuery: {query}"
        # The WHOLE passage, not its first line: a multi-line or preamble-prefixed completion
        # would otherwise embed a fragment instead of the hypothetical document (GEN-3).
        doc = gen_text(gen(prompt), default=query)
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
        seed = self.hydrate(self.store.query_vector(qv.tolist(), top_k=feedback_k, flt=flt))
        texts = [h.text for h in seed if h.text]
        if not texts:
            return self.store.query_vector(qv.tolist(), top_k=top_k, flt=flt)
        dvecs = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        new_q = alpha * qv + beta * dvecs.mean(axis=0)
        new_q = new_q / (np.linalg.norm(new_q) or 1.0)
        return self.store.query_vector(new_q.tolist(), top_k=top_k, flt=flt)

    def answerability(self, query: str, top_k: int = 30) -> dict:
        """Probe whether the corpus can even answer, cheaply. Embed a hypothetical ANSWER (HyDE)
        and dense-search it — the top score is the max cosine similarity of any doc to what a real
        answer looks like. LOW ``max_sim`` ⇒ the answer is probably NOT in the corpus → abstain/stop
        instead of burning tokens going wider. Needs a generator."""
        gen = self._require_generator()
        doc = gen_text(gen(f"Write a short passage that directly answers this query.\nQuery: {query}"),
                       default=query)  # whole passage, not line 1 (GEN-3)
        vec = self.embedder.embed([doc])[0]
        hits = self.store.query_vector(vec, top_k=top_k)
        top = max((h.score for h in hits), default=0.0)
        return {"max_sim": round(float(top), 3), "likely_answerable": top >= 0.5}

    def diversity(self, results: ResultSet, top_k: int = 10) -> dict:
        """Mean pairwise cosine similarity of the top-k hits (re-embeds their text, so it works on
        backends that don't return stored vectors). HIGH (→1.0) = results collapsed to near-duplicates
        (the search is stuck / one-source-dominated); LOW = diverse coverage."""
        import numpy as np

        hits = sorted(results, key=lambda h: h.score, reverse=True)[:top_k]
        texts = [h.text or "" for h in hits]
        if len([t for t in texts if t]) < 2:
            return {"mean_similarity": 0.0, "redundant": False, "n": len(hits)}
        v = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        sim = v @ v.T
        iu = np.triu_indices(len(hits), k=1)
        mean = float(sim[iu].mean())
        return {"mean_similarity": round(mean, 3), "redundant": mean >= 0.92, "n": len(hits)}

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

    def reset_state(self) -> None:
        """Clear the whole out-of-context state store — call BETWEEN QUERIES when one Session
        serves many. Benchmarks used to clear three hand-picked keys and leak everything else
        an agent program stashed via ``remember(...)`` into later queries' ``recall(...)``
        (issues.md P1-4)."""
        self._state.clear()

    def state_keys(self) -> list[str]:
        return list(self._state)

    # ---- internals -------------------------------------------------------
    def _embed_query(self, query: str) -> list[float]:
        if self._caps.server_side_embedding:
            v = self.store.embed_query(query)
            if v is not None:
                return v
        return self.query_embedder.embed([query])[0]

    def _keyword(self, query: str, top_k: int, flt: dict) -> ResultSet:
        if self._caps.keyword:
            return self.store.query_keyword(query, top_k=top_k, flt=flt)
        # Emulate: dense recall then lexical rerank.
        pool = self.store.query_vector(self._embed_query(query), top_k=top_k * 5, flt=flt)
        return P.rerank(query, pool, top_k=top_k)

    def _hybrid(self, query: str, top_k: int, flt: dict, alpha: float = 0.5) -> ResultSet:
        vec = self._embed_query(query)
        if self._caps.hybrid:
            return self.store.query_hybrid(vec, query, top_k=top_k, flt=flt, alpha=alpha)
        dense = self.store.query_vector(vec, top_k=top_k * 3, flt=flt)
        kw = self._keyword(query, top_k * 3, flt)
        return P.fuse([dense, kw], weights=[alpha, 1 - alpha]).top(top_k)

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
            raise GeneratorRequiredError(
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
