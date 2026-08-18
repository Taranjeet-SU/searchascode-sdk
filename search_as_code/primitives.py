"""Composable retrieval primitives — the atoms agent code writes against.

These are deliberately *lower level* than a monolithic ``search()`` endpoint
(the Perplexity "search as code" insight): fan-out, fusion, dedup, rerank, and
extraction are separate so the model can orchestrate them however the task
needs.  Everything here is portable — it runs on ``ResultSet`` objects and never
touches a specific backend.
"""

from __future__ import annotations

import re as _re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from .errors import ExtractorRequiredError
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


# RRF is exactly the fusion primitive above; `rrf` is an explicit, discoverable alias.
rrf = fuse


def dedup(results: ResultSet, key: Optional[Callable[[Hit], Any]] = None) -> ResultSet:
    return results.dedup(key)


def score_cutoff(results: ResultSet, method: str = "band", rel_band: float = 0.1,
                 min_k: int = 10, max_k: int = 100) -> ResultSet:
    """Adaptive result-set sizing from the score distribution — keep MORE when the
    similarity curve is flat (many near-equally-relevant hits), FEWER when it drops
    off sharply. This is the "don't hard-cut at k" idea from score-based retrieval.

    method="band": keep hits within ``rel_band`` of the top score (relative), i.e.
      score >= top*(1-rel_band). Flat curve → keeps a large pool; peaked → keeps few.
    method="knee": cut at the largest score gap between consecutive ranks (elbow).
    Always returns between ``min_k`` and ``max_k`` hits.
    """
    hits = sorted(results, key=lambda h: h.score, reverse=True)[:max_k]
    if not hits:
        return ResultSet()
    if method == "knee":
        scores = [h.score for h in hits]
        best_i, best_gap = len(hits), -1.0
        for i in range(min_k, len(scores)):
            gap = scores[i - 1] - scores[i]
            if gap > best_gap:
                best_gap, best_i = gap, i
        kept = hits[:best_i]
    else:  # band (relative to the top score)
        top = hits[0].score
        cut = top * (1 - rel_band) if top > 0 else top - rel_band
        kept = [h for h in hits if h.score >= cut]
    if len(kept) < min_k:
        kept = hits[:min_k]
    return ResultSet(kept[:max_k])


def normalize_scores(results: ResultSet, method: str = "minmax") -> ResultSet:
    """Rescale scores so incomparable BM25/cosine scales become fusable while
    KEEPING magnitude (unlike RRF, which discards it). method=minmax → [0,1];
    zscore → standardized. (Weaviate hybrid-fusion.)"""
    if not results:
        return ResultSet()
    scores = [h.score for h in results]
    if method == "zscore":
        mu = sum(scores) / len(scores)
        var = sum((s - mu) ** 2 for s in scores) / len(scores)
        sd = var ** 0.5 or 1.0
        norm = [(s - mu) / sd for s in scores]
    else:
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        norm = [(s - lo) / rng for s in scores]
    return ResultSet(Hit(id=h.id, score=float(n), document=h.document, query=h.query, store=h.store)
                     for h, n in zip(results, norm))


def relative_score_fusion(result_sets: Sequence[ResultSet], weights: Optional[Sequence[float]] = None,
                          method: str = "minmax") -> ResultSet:
    """Fuse by summing NORMALIZED scores (not ranks) — preserves how much better a
    hit is, not just its position. Weaviate's default fusion since v1.24; often
    beats RRF when score magnitudes are meaningful."""
    weights = list(weights) if weights is not None else [1.0] * len(result_sets)
    agg: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for rs, w in zip(result_sets, weights):
        for h in normalize_scores(rs, method):
            agg[h.id] = agg.get(h.id, 0.0) + w * h.score
            if h.id not in keep or h.score > keep[h.id].score:
                keep[h.id] = h
    fused = [Hit(id=i, score=s, document=keep[i].document, query=keep[i].query, store=keep[i].store)
             for i, s in agg.items()]
    fused.sort(key=lambda h: h.score, reverse=True)
    return ResultSet(fused)


def diversity_quota(results: ResultSet, key: Callable[[Hit], Any],
                    max_per_group: int = 1, top_k: Optional[int] = None) -> ResultSet:
    """Enforce source/topic/entity diversity: at most ``max_per_group`` hits per
    group (by ``key``) while walking down the ranking. (Vespa result diversity.)"""
    counts: dict[Any, int] = {}
    out: list[Hit] = []
    for h in sorted(results, key=lambda x: x.score, reverse=True):
        g = key(h)
        if counts.get(g, 0) < max_per_group:
            out.append(h)
            counts[g] = counts.get(g, 0) + 1
            if top_k and len(out) >= top_k:
                break
    return ResultSet(out)


def confidence(results: ResultSet) -> dict[str, float]:
    """Retrieval-confidence signals from the score curve: top score and the gap to
    #2 (a large gap = a confident single winner). Feeds abstain/reformulate. (R³AG.)"""
    hits = sorted(results, key=lambda h: h.score, reverse=True)
    if not hits:
        return {"top": 0.0, "gap": 0.0, "n": 0}
    top = hits[0].score
    gap = top - hits[1].score if len(hits) > 1 else top
    return {"top": float(top), "gap": float(gap), "n": len(hits)}


def abstain(results: ResultSet, min_top: float = 0.0, min_gap: float = 0.0) -> bool:
    """True when the result is too weak/uncertain to trust (top score or score-gap
    below threshold) — the agent should reformulate or say 'insufficient' rather
    than answer from noise. (R³AG confidence gating.)"""
    c = confidence(results)
    return c["n"] == 0 or c["top"] < min_top or c["gap"] < min_gap


def consensus(result_sets: Sequence[ResultSet], top_k: int = 10, per_list_k: int = 10) -> ResultSet:
    """Vote across MANY ranked lists (rephrasings × modes × HyDE/PRF × rerankers): count how
    often each id lands in the top ``per_list_k`` of EACH list, tie-broken by mean reciprocal
    rank. Surfaces the passages a MAJORITY of independent strategies agree on — the "wins for
    everyone" signal that a single ranker can't give you.

    Returns a ResultSet ranked by consensus, with three attributes attached for gating/learning:
    ``.agreement`` (top id's votes ÷ n_lists, in [0,1] — high = confident) · ``.votes``
    ({id: vote_count}) · ``.n_lists``. Low agreement ⇒ strategies disagree ⇒ enrich hop 2.
    """
    lists = [rs for rs in result_sets if rs]
    n = len(lists) or 1
    votes: dict[str, int] = {}
    rr: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for rs in lists:
        for rank, h in enumerate(sorted(rs, key=lambda x: x.score, reverse=True)[:per_list_k]):
            votes[h.id] = votes.get(h.id, 0) + 1
            rr[h.id] = rr.get(h.id, 0.0) + 1.0 / (rank + 1)
            if h.id not in keep or h.score > keep[h.id].score:
                keep[h.id] = h
    scored = [Hit(id=i, score=votes[i] + rr[i] / n, document=keep[i].document,
                  query=keep[i].query, store=keep[i].store) for i in votes]
    scored.sort(key=lambda h: h.score, reverse=True)
    # Carried on .info so the signals SURVIVE chaining. They used to be set as ad-hoc
    # attributes, and top()/dedup()/where() build a new ResultSet — so the documented gating
    # signals vanished as soon as agent code chained anything (SDK-C13).
    return ResultSet(scored[:top_k], info={
        "agreement": round(max(votes.values()) / n, 3) if votes else 0.0,
        "votes": {i: votes[i] for i in sorted(votes, key=lambda x: -votes[x])[:top_k]},
        "n_lists": n,
    })


# ---- stop signals: "is an answer even here?" without burning tokens ----------
def score_cliff(results: ResultSet, min_k: int = 3) -> dict[str, Any]:
    """Detect a sharp score DROP-OFF (a "cliff") in the ranked scores. A clear cliff after a
    few hits ⇒ a well-separated relevant set (answer likely present, keep above the cliff).
    NO cliff + flat low scores ⇒ nothing stands out (answer may be absent). Returns
    {has_cliff, cliff_at (rank), drop (relative size of the biggest gap)}."""
    scores = sorted((h.score for h in results), reverse=True)
    if len(scores) < min_k + 2:
        return {"has_cliff": False, "cliff_at": len(scores), "drop": 0.0}
    best_i, best_drop = len(scores), 0.0
    for i in range(min_k, len(scores)):
        prev = scores[i - 1]
        drop = (prev - scores[i]) / (abs(prev) + 1e-9)
        if drop > best_drop:
            best_drop, best_i = drop, i
    return {"has_cliff": best_drop >= 0.5, "cliff_at": best_i, "drop": round(best_drop, 3)}


def result_diversity(results: ResultSet, top_k: int = 10) -> dict[str, Any]:
    """Mean pairwise cosine similarity of the top-k hits' vectors. HIGH mean (→1.0) = results
    collapsed to near-duplicates (redundant / the search is stuck re-finding the same doc);
    LOW = diverse coverage. Returns {mean_similarity, redundant (>=0.9), n}. Needs hit vectors."""
    import numpy as np

    hits = [h for h in sorted(results, key=lambda h: h.score, reverse=True)[:top_k]
            if h.document is not None and h.document.vector is not None]
    if len(hits) < 2:
        return {"mean_similarity": 0.0, "redundant": False, "n": len(hits)}
    v = np.asarray([h.document.vector for h in hits], dtype=np.float32)  # type: ignore[union-attr]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    sim = v @ v.T
    iu = np.triu_indices(len(hits), k=1)
    mean = float(sim[iu].mean())
    return {"mean_similarity": round(mean, 3), "redundant": mean >= 0.9, "n": len(hits)}


def max_similarity(vector: Sequence[float], results: ResultSet) -> float:
    """Max cosine similarity between a probe ``vector`` (e.g. a HyDE hypothetical-answer
    embedding) and the retrieved hits. LOW ⇒ even the best candidate is far from what a real
    answer looks like → the answer is probably NOT in the corpus (abstain, stop searching)."""
    import numpy as np

    hits = [h for h in results if h.document is not None and h.document.vector is not None]
    if not hits:
        return 0.0
    q = np.asarray(vector, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    m = np.asarray([h.document.vector for h in hits], dtype=np.float32)  # type: ignore
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return round(float((m @ q).max()), 3)


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
        assert h.document is not None  # guaranteed by the withvec filter above
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
        f"Generate {n} alternative search queries. Crucially, include SYNONYMS, EUPHEMISMS, and "
        f"domain aliases (e.g. 'disappear'->'die', 'hobbyist'->'hobby', 'CD'->'certificate of "
        f"deposit'), not just reworded questions — paraphrases alone retrieve the same documents. "
        f"Return one per line.\nQuery: {query}"
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


def topics(query: str, generate: Callable[[str], list[str]], n: int = 5) -> list[str]:
    """LLM topic/entity extraction — the key concepts to search or filter on."""
    prompt = (
        f"List up to {n} key topics or named entities in this search query, "
        f"one per line, no numbering or extra text.\nQuery: {query}"
    )
    return [t.strip(" -*\t") for t in generate(prompt) if t.strip()][:n]


def auto_filter(query: str, generate: Callable[[str], list[str]],
                fields: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Self-query: the LLM infers a metadata filter implied by the query
    (LangChain SelfQueryRetriever / LlamaIndex auto-retrieval). Returns a filter
    dict in the portable dialect, or {} if the query implies no constraint."""
    import json
    import re

    fdesc = f"Available metadata fields: {', '.join(fields)}.\n" if fields else ""
    prompt = (
        "Infer a metadata filter that this query implies, as a JSON object in this dialect: "
        '{"field": value} or {"field": {"$gte": n}} / {"$in": [...]} etc. '
        "Return ONLY the JSON object, or {} if the query implies no filter.\n"
        f"{fdesc}Query: {query}"
    )
    raw = "\n".join(generate(prompt))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


# Spelling / acronym normalization (extend per domain). Fixes the cheque↔check class.
DEFAULT_ALIASES = {
    "cheque": "check", "cheques": "checks", "favour": "favor", "colour": "color",
    "organisation": "organization", "cancelled": "canceled", "cheque's": "check's",
}


def normalize_query(query: str, aliases: Optional[dict] = None) -> str:
    """Normalize spelling/acronym variants so query and doc share tokens
    (cheque→check, US/UK, domain aliases). Apply to query AND at index time."""
    aliases = aliases or DEFAULT_ALIASES
    return "".join(aliases.get(t.lower(), t) if t.strip() else t
                   for t in _re.findall(r"\w+|\W+", query))


def rare_terms(query: str) -> list[str]:
    """High-signal exact tokens worth boosting in a keyword pass: quoted phrases,
    ALL-CAPS acronyms (CD, EIN, SEC, LLC), numbers/$amounts/versions, Proper Bigrams."""
    terms: list[str] = []
    terms += _re.findall(r'"([^"]+)"', query)                       # quoted phrases
    terms += _re.findall(r"\b[A-Z]{2,}\b", query)                   # acronyms
    terms += _re.findall(r"\$?\d[\d,.]*[kKmM%]?", query)            # numbers / $ / versions
    terms += _re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", query)   # Proper Bigrams
    seen: set = set()
    out: list[str] = []
    for t in terms:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def quality_filter(results: ResultSet, min_chars: int = 40) -> ResultSet:
    """Drop empty / near-empty docs (label & parse artifacts, pure noise)."""
    return ResultSet(h for h in results if h.text and len(h.text.strip()) >= min_chars)


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
        raise ExtractorRequiredError(
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


def content_type(text: str) -> str:
    """Heuristic classifier so an agent knows the DATA SHAPE of a chunk — one of
    'table', 'fact-card', 'list', 'code', 'prose', 'short-fact', 'empty'.
    Pairs with store.describe_schema()/sample() for schema-first agentic retrieval."""
    if not text or not text.strip():
        return "empty"
    t = text.strip()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    kv = sum(1 for ln in lines if "=" in ln and len(ln) < 90)
    if t.count("|") >= 4 or t.count("\t") >= 3:
        return "table"
    if lines and kv >= max(2, len(lines) // 2):
        return "fact-card"
    if lines and sum(1 for ln in lines if ln.lstrip()[:2] in ("- ", "* ", "• ")
                     or ln.lstrip()[:1].isdigit()) >= max(2, len(lines) // 2):
        return "list"
    if any(k in t for k in ("def ", "import ", "function ", "{", "};", "</")):
        return "code"
    if t.count(". ") >= 2 or len(t) > 300:
        return "prose"
    return "short-fact"
