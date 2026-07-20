"""Composable retrieval primitives — the atoms agent code writes against.

These are deliberately *lower level* than a monolithic ``search()`` endpoint
(the Perplexity "search as code" insight): fan-out, fusion, dedup, rerank, and
extraction are separate so the model can orchestrate them however the task
needs.  Everything here is portable — it runs on ``ResultSet`` objects and never
touches a specific backend.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from .types import Hit, ResultSet


def fan_out(
    fn: Callable[[Any], ResultSet],
    items: Sequence[Any],
    concurrency: int = 8,
) -> list[ResultSet]:
    """Run ``fn`` over ``items`` concurrently. The workhorse behind querying many
    variants at once without serializing through model turns."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        return list(ex.map(fn, items))


def fuse(
    result_sets: Sequence[ResultSet],
    weights: Optional[Sequence[float]] = None,
    k: int = 60,
) -> ResultSet:
    """Reciprocal Rank Fusion across result sets.

    RRF is rank-based, so it merges dense, keyword, and multi-query results
    without needing comparable score scales — the portable, model-free way to
    combine signals.  ``weights`` optionally biases each set.
    """
    weights = list(weights) if weights is not None else [1.0] * len(result_sets)
    agg: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for rs, w in zip(result_sets, weights):
        for rank, hit in enumerate(sorted(rs, key=lambda h: h.score, reverse=True)):
            agg[hit.id] = agg.get(hit.id, 0.0) + w * (1.0 / (k + rank + 1))
            if hit.id not in keep or hit.score > keep[hit.id].score:
                keep[hit.id] = hit
    fused = [
        Hit(id=i, score=s, document=keep[i].document, query=keep[i].query, store=keep[i].store)
        for i, s in agg.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return ResultSet(fused)


def dedup(results: ResultSet, key: Optional[Callable[[Hit], Any]] = None) -> ResultSet:
    return results.dedup(key)


def rerank(
    query: str,
    results: ResultSet,
    reranker: Optional[Callable[[str, list[str]], list[float]]] = None,
    top_k: Optional[int] = None,
) -> ResultSet:
    """Re-score ``results`` for ``query``.

    Pass a ``reranker(query, texts) -> scores`` (e.g. a cross-encoder). When none
    is supplied we emulate with lexical-overlap scoring so the primitive still
    works on every backend — quality tracks whatever reranker you inject.
    """
    if not results:
        return results
    texts = [h.text or "" for h in results]
    scores = reranker(query, texts) if reranker else _lexical_overlap(query, texts)
    rescored = [
        Hit(id=h.id, score=float(s), document=h.document, query=h.query, store=h.store)
        for h, s in zip(results, scores)
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    out = ResultSet(rescored)
    return out.top(top_k) if top_k else out


def freshness(
    results: ResultSet,
    timestamp: Callable[[Hit], float],
    now: float,
    half_life: float,
    weight: float = 0.5,
) -> ResultSet:
    """Blend a recency decay into scores (a Hornet-style freshness primitive).

    ``now``/``half_life`` are caller-supplied (seconds) because the sandbox
    forbids wall-clock nondeterminism; pass ``time.time()`` from the harness.
    """
    import math

    out = []
    for h in results:
        age = max(0.0, now - timestamp(h))
        decay = 0.5 ** (age / half_life) if half_life > 0 else 1.0
        blended = (1 - weight) * h.score + weight * decay
        out.append(Hit(id=h.id, score=blended, document=h.document, query=h.query, store=h.store))
    out.sort(key=lambda h: h.score, reverse=True)
    return ResultSet(out)


def mmr(
    query_vector: Sequence[float],
    results: ResultSet,
    lambda_: float = 0.5,
    top_k: int = 10,
) -> ResultSet:
    """Maximal Marginal Relevance — greedily pick results that are relevant to
    the query but diverse from what's already chosen (Carbonell & Goldstein '98).

    Portable and model-free: needs only the query vector and per-hit vectors, so
    it complements ``dedup`` by suppressing *near*-duplicates, not just exact ids.
    Hits without a vector are appended after the diversified set.
    """
    import numpy as np

    q = np.asarray(query_vector, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    withvec = [h for h in results if h.document and h.document.vector is not None]
    novec = [h for h in results if not (h.document and h.document.vector is not None)]
    vecs = {}
    for h in withvec:
        v = np.asarray(h.document.vector, dtype=np.float32)
        vecs[h.id] = v / (np.linalg.norm(v) or 1.0)

    selected: list[Hit] = []
    pool = list(withvec)
    while pool and len(selected) < top_k:
        best, best_score = None, float("-inf")
        for h in pool:
            rel = float(q @ vecs[h.id])
            div = max((float(vecs[h.id] @ vecs[s.id]) for s in selected), default=0.0)
            score = lambda_ * rel - (1 - lambda_) * div
            if score > best_score:
                best, best_score = h, score
        selected.append(best)  # type: ignore[arg-type]
        pool.remove(best)  # type: ignore[arg-type]
    return ResultSet(selected + novec[: max(0, top_k - len(selected))])


def expand(query: str, generate: Callable[[str], list[str]], n: int = 4) -> list[str]:
    """Query expansion / multi-query (RAG-Fusion, LangChain MultiQuery, Haystack
    QueryExpander).  ``generate(prompt) -> list[str]`` is your LLM; the original
    query is always included so recall never drops below the baseline."""
    prompt = (
        f"Generate {n} alternative search queries that capture different phrasings "
        f"and facets of this query. Return one per line.\nQuery: {query}"
    )
    variants = [q.strip() for q in generate(prompt) if q.strip()]
    return [query, *[v for v in variants if v != query]]


def rephrase(query: str, generate: Callable[[str], list[str]]) -> str:
    """Rewrite a query into a single retrieval-optimized, standalone formulation
    (Rewrite-Retrieve-Read). Returns the original if the model yields nothing."""
    prompt = (
        "Rewrite the following search query to be clearer and more retrieval-effective "
        "while preserving its exact information need. Return only the rewritten query.\n"
        f"Query: {query}"
    )
    out = [q.strip() for q in generate(prompt) if q.strip()]
    return out[0] if out else query


def decompose(query: str, generate: Callable[[str], list[str]]) -> list[str]:
    """Break a complex query into answerable sub-questions (least-to-most,
    LlamaIndex sub-question, Haystack decomposition)."""
    prompt = (
        "Decompose this question into the minimal set of simpler sub-questions "
        f"needed to answer it. Return one per line.\nQuestion: {query}"
    )
    return [q.strip() for q in generate(prompt) if q.strip()]


def extract(
    results: ResultSet,
    schema: dict[str, Any],
    instruction: str,
    extractor: Optional[Callable[[list[str], dict, str], list[dict]]] = None,
) -> list[dict[str, Any]]:
    """Pull structured records out of hits (the SaC "verification" stage).

    Requires an ``extractor(texts, schema, instruction) -> list[dict]`` — usually
    an LLM call.  Kept pluggable so the core stays model-agnostic.
    """
    if extractor is None:
        raise RuntimeError(
            "extract() needs an extractor callable (e.g. an LLM). "
            "Pass Session(..., extractor=fn) or extract(..., extractor=fn)."
        )
    return extractor(results.texts(), schema, instruction)


def _lexical_overlap(query: str, texts: Sequence[str]) -> list[float]:
    from .embeddings import _tokenize

    q = set(_tokenize(query))
    if not q:
        return [0.0] * len(texts)
    scores = []
    for t in texts:
        toks = set(_tokenize(t))
        scores.append(len(q & toks) / len(q) if toks else 0.0)
    return scores
