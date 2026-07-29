"""20 named SAC retrieval **templates** — each a distinct recipe composed from primitives.

A template maps a query to a ranked list of doc ids. To label many queries cheaply, we
compute a small set of *base pools* once per query (dense / keyword / hyde / decompose /
exact / regex) and an optional single rerank pass, then every template is a cheap
recombination of those pools. This is what the router learns to pick per query.

Portable: OpenSearch-only primitives (``query_phrase``/``query_regex``) degrade to
keyword/regex search on other backends; generator-only pools (hyde/decompose) are skipped
when no generator is attached.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# part numbers (AGFC019, MT40A1G16RC-062E), rails/signals (VDDQ6, ETH0_GPIO1)
CODE_RE = re.compile(
    r"\b(?=[A-Z0-9_]*\d)[A-Z0-9]{2,}(?:[-_/][A-Z0-9]+)+\b"
    r"|\b[A-Z]{2,}\d{2,}[A-Z0-9]*\b|\bVDD[A-Z0-9]*\b|\b[A-Z]{2,}\d*_[A-Z0-9_]+\b"
)


def extract_codes(q: str) -> list[str]:
    seen, out = set(), []
    for m in CODE_RE.findall(q):
        if m not in seen and len(m) >= 4:
            seen.add(m); out.append(m)
    return out[:2]


def _ids(rs) -> list[str]:
    return [h.id for h in rs] if rs is not None else []


def _rrf(lists, weights=None, k=60) -> list[str]:
    """Weighted reciprocal-rank fusion over lists of id-lists."""
    weights = weights or [1.0] * len(lists)
    score: dict[str, float] = {}
    for w, lst in zip(weights, lists):
        for rank, did in enumerate(lst):
            score[did] = score.get(did, 0.0) + w / (k + rank + 1)
    return sorted(score, key=lambda d: -score[d])


def _by_rerank(ids: list[str], rr: Optional[dict]) -> list[str]:
    """Reorder ids by cached rerank scores (stable fallback when no reranker)."""
    if not rr:
        return ids
    return sorted(ids, key=lambda d: -rr.get(d, -1e9))


# --------------------------------------------------------------------------- #
# base pools — computed ONCE per query, shared by all templates                #
# --------------------------------------------------------------------------- #
def base_pools(session, query: str, P: int = 25, use_llm: bool = True,
               use_codes: bool = True, query_vec=None):
    """Return (pools, docs): pools = {name: [ids]}, docs = {id: text} harvested from the
    retrieved hits (so a rerank pass needs no extra fetches).

    ``query_vec`` (optional): a precomputed query embedding — reused for the dense pool so
    the caller doesn't pay a second embed (e.g. when it already embedded for features)."""
    pools: dict[str, list[str]] = {}
    docs: dict[str, str] = {}

    def _run(fn):
        try:
            rs = fn()
        except Exception:
            return []
        ids = []
        for h in rs:
            ids.append(h.id)
            if h.id not in docs:
                docs[h.id] = getattr(h.document, "text", None) or ""
        return ids

    if query_vec is not None:
        pools["dense"] = _run(lambda: session.store.query_vector(query_vec, top_k=P))
    else:
        pools["dense"] = _run(lambda: session.search(query, top_k=P, mode="dense"))
    pools["keyword"] = _run(lambda: session.search(query, top_k=P, mode="keyword"))
    if use_llm and session.generator is not None:
        pools["hyde"] = _run(lambda: session.hyde_search(query, top_k=P))
        pools["decompose"] = _run(lambda: session.decompose_search(query, top_k=P))
    if use_codes:
        codes = extract_codes(query)
        if codes:
            qp = getattr(session.store, "query_phrase", None)
            exact, regex = [], []
            for c in codes:
                exact += (_run(lambda c=c: session.store.query_phrase(c, top_k=P)) if qp
                          else _run(lambda c=c: session.search(c, top_k=P, mode="keyword")))
                regex += _run(lambda c=c: session.search(re.escape(c), top_k=P, mode="regex"))
            if exact:
                pools["exact"] = list(dict.fromkeys(exact))
            if regex:
                pools["regex"] = list(dict.fromkeys(regex))
    return pools, docs


def rerank_cache(session, query: str, pools: dict, cap: int = 40,
                 docs: Optional[dict] = None) -> Optional[dict]:
    """One cross-encoder pass over the union of pool candidates -> {id: score}.
    ``docs`` maps id -> text (needed to score); returns None if no reranker/text."""
    if getattr(session, "reranker", None) is None or not docs:
        return None
    cand = list(dict.fromkeys(d for lst in pools.values() for d in lst))[:cap]
    texts = [docs.get(d, "") for d in cand]
    if not any(texts):
        return None
    try:
        scores = session.reranker(query, texts)
        return {d: float(s) for d, s in zip(cand, scores)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# the 20 templates: (pools, rr) -> ranked ids                                   #
# --------------------------------------------------------------------------- #
def _p(pools, name):
    return pools.get(name, [])


TEMPLATES: dict[str, Callable[[dict, Optional[dict]], list[str]]] = {
    "dense":              lambda P, r: _p(P, "dense"),
    "keyword":            lambda P, r: _p(P, "keyword"),
    "hybrid":             lambda P, r: _rrf([_p(P, "dense"), _p(P, "keyword")]),
    "hybrid_dense_heavy": lambda P, r: _rrf([_p(P, "dense"), _p(P, "keyword")], [0.8, 0.2]),
    "hybrid_kw_heavy":    lambda P, r: _rrf([_p(P, "dense"), _p(P, "keyword")], [0.2, 0.8]),
    "hyde":               lambda P, r: _p(P, "hyde") or _p(P, "dense"),
    "hyde_dense":         lambda P, r: _rrf([_p(P, "hyde"), _p(P, "dense")]),
    "hyde_hybrid":        lambda P, r: _rrf([_p(P, "hyde"), _p(P, "dense"), _p(P, "keyword")]),
    "decompose":          lambda P, r: _p(P, "decompose") or _p(P, "dense"),
    "decompose_dense":    lambda P, r: _rrf([_p(P, "decompose"), _p(P, "dense")]),
    "dense_rerank":       lambda P, r: _by_rerank(_p(P, "dense"), r),
    "hybrid_rerank":      lambda P, r: _by_rerank(_rrf([_p(P, "dense"), _p(P, "keyword")]), r),
    "keyword_rerank":     lambda P, r: _by_rerank(_p(P, "keyword"), r),
    "all_rrf":            lambda P, r: _rrf([_p(P, "dense"), _p(P, "keyword"),
                                            _p(P, "hyde"), _p(P, "decompose")]),
    "all_rerank":         lambda P, r: _by_rerank(_rrf([_p(P, "dense"), _p(P, "keyword"),
                                            _p(P, "hyde"), _p(P, "decompose")]), r),
    "exact":              lambda P, r: _p(P, "exact") or _p(P, "keyword"),
    "regex":              lambda P, r: _p(P, "regex") or _p(P, "keyword"),
    "exact_dense":        lambda P, r: _rrf([_p(P, "exact"), _p(P, "dense")]) or _p(P, "dense"),
    "code_fusion":        lambda P, r: _rrf([_p(P, "exact"), _p(P, "regex"),
                                            _p(P, "dense")]) or _p(P, "dense"),
    "code_rerank":        lambda P, r: _by_rerank(_rrf([_p(P, "exact"), _p(P, "regex"),
                                            _p(P, "dense"), _p(P, "keyword")]), r),
}

TEMPLATE_NAMES = list(TEMPLATES)


def apply_template(name: str, pools: dict, rr: Optional[dict], top_k: int = 10) -> list[str]:
    return TEMPLATES[name](pools, rr)[:top_k]
