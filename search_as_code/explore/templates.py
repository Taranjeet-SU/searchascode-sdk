"""SAC retrieval **templates** — named end-to-end *strategies* (not primitive configs).

Each template is a small procedure composing primitives at a given **effort tier** — some
are **adaptive** (retrieve, check whether the top score "fell off", and escalate/return
early). The router learns which strategy to spend on per query, which is what controls both
quality *and* latency (cheap one-hop for easy queries, deep only when needed).

Tiers:
  light    - one-hop, no LLM query-ops (dense / keyword / hybrid / rephrase+rerank)
  medium   - one extra signal (hyde, prf, mmr, multi-rephrase, part-number exact)
  deep     - decompose fan-out and/or hyde, fused + reranked (+compress)
  adaptive - score-guarded / escalating cascade / confidence-gated

To keep labeling affordable, a per-query :class:`StrategyContext` **memoizes** the shared
sub-results (dense/keyword/hyde/decompose/rephrase/exact/regex pools and rerank passes), so
running all templates for one query costs about one of each primitive, not N.
"""

from __future__ import annotations

import re
from typing import Callable

from .. import primitives as P
from ..types import ResultSet

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


def _margin(rs: ResultSet) -> float:
    """Relative gap between the top-1 and top-2 scores — a cheap 'is the top hit clearly
    the winner' signal. Low margin => ambiguous => a deeper strategy may help."""
    if not rs or len(rs) < 2:
        return 1.0 if rs else 0.0
    s0, s1 = float(rs[0].score), float(rs[1].score)
    return (s0 - s1) / (abs(s0) + 1e-9)


class StrategyContext:
    """Per-query memoized access to primitives, so templates compose without recomputing."""

    def __init__(self, session, query, P_pool: int = 25, emb=None,
                 use_llm: bool = True, use_rerank: bool = True, top_k: int = 10,
                 rerank_lock=None):
        self.s = session
        self.q = query
        self.P = P_pool
        self.emb = emb
        self.use_llm = use_llm and session.generator is not None
        self.use_rerank = use_rerank and getattr(session, "reranker", None) is not None
        self.top_k = top_k
        self._c: dict = {}
        self._rr_lock = rerank_lock            # serialize GPU reranker across worker threads

    def _memo(self, key, fn) -> ResultSet:
        if key not in self._c:
            try:
                self._c[key] = fn() or ResultSet()
            except Exception:
                self._c[key] = ResultSet()
        return self._c[key]

    # ---- base retrievals (cached) ---------------------------------------
    def dense(self) -> ResultSet:
        if self.emb is not None:
            return self._memo("dense", lambda: self.s.store.query_vector(self.emb, top_k=self.P))
        return self._memo("dense", lambda: self.s.search(self.q, self.P, mode="dense"))

    def keyword(self) -> ResultSet:
        return self._memo("keyword", lambda: self.s.search(self.q, self.P, mode="keyword"))

    def hybrid(self, alpha=0.5) -> ResultSet:
        return self._memo(f"hybrid{alpha}", lambda: self.s.search(self.q, self.P, mode="hybrid", alpha=alpha))

    # ---- LLM query-side ops (cached; degrade to dense/hybrid if no generator)
    def hyde(self) -> ResultSet:
        if not self.use_llm:
            return self.dense()
        return self._memo("hyde", lambda: self.s.hyde_search(self.q, self.P))

    def decompose(self) -> ResultSet:
        if not self.use_llm:
            return self.dense()
        return self._memo("decompose", lambda: self.s.decompose_search(self.q, self.P))

    def rephrased(self) -> ResultSet:
        if not self.use_llm:
            return self.hybrid()
        return self._memo("rephrased", lambda: self.s.rephrase_search(self.q, self.P, mode="hybrid"))

    def expanded(self, n=4) -> ResultSet:
        if not self.use_llm:
            return self.hybrid()
        return self._memo(f"expand{n}", lambda: self.s.expand_search(self.q, self.P, n=n, mode="hybrid"))

    def prf(self) -> ResultSet:
        return self._memo("prf", lambda: self.s.prf_search(self.q, self.P))

    # ---- part-number / code pools ---------------------------------------
    def exact(self) -> ResultSet:
        def _go():
            out = ResultSet()
            qp = getattr(self.s.store, "query_phrase", None)
            for c in extract_codes(self.q):
                out = P.fuse([out, qp(c, top_k=self.P)]) if qp else \
                      P.fuse([out, self.s.search(c, self.P, mode="keyword")])
            return out
        return self._memo("exact", _go)

    def regex(self) -> ResultSet:
        def _go():
            out = ResultSet()
            for c in extract_codes(self.q):
                out = P.fuse([out, self.s.search(re.escape(c), self.P, mode="regex")])
            return out
        return self._memo("regex", _go)

    def has_codes(self) -> bool:
        return bool(extract_codes(self.q))

    # ---- ops -------------------------------------------------------------
    def fuse(self, *sets, weights=None) -> ResultSet:
        nz = [r for r in sets if r]
        return P.fuse(nz, weights=weights) if nz else ResultSet()

    def rerank(self, rs: ResultSet) -> ResultSet:
        if not self.use_rerank or not rs:
            return rs
        key = ("rr", tuple(rs.ids()[:40]))
        if key not in self._c:
            try:
                if self._rr_lock is not None:
                    with self._rr_lock:
                        self._c[key] = self.s.rerank(self.q, rs, top_k=self.top_k)
                else:
                    self._c[key] = self.s.rerank(self.q, rs, top_k=self.top_k)
            except Exception:
                self._c[key] = rs
        return self._c[key]

    def mmr(self, rs: ResultSet) -> ResultSet:
        try:
            return self.s.mmr(self.q, rs, top_k=self.top_k)
        except Exception:
            return rs

    def weak(self, rs: ResultSet, min_margin=0.05) -> bool:
        """Did the score 'fall off' — empty, thin, or ambiguous top hit?"""
        return (not rs) or len(rs) < 3 or _margin(rs) < min_margin


# --------------------------------------------------------------------------- #
# the templates: ctx -> ResultSet                                              #
# --------------------------------------------------------------------------- #
def _t_light_dense(c):      return c.dense()
def _t_light_keyword(c):    return c.keyword()
def _t_light_hybrid(c):     return c.hybrid()
def _t_rephrase_rerank(c):  return c.rerank(c.rephrased())            # one-hop: rephrase->retrieve->rerank

def _t_dense_rerank(c):     return c.rerank(c.dense())
def _t_hyde_rerank(c):      return c.rerank(c.hyde())
def _t_mmr_diverse(c):      return c.mmr(c.dense())
def _t_prf_rerank(c):       return c.rerank(c.prf())
def _t_multi_rephrase(c):   return c.rerank(c.expanded(n=4))          # many rephrases -> fuse -> rerank
def _t_exact_partnum(c):    return c.rerank(c.fuse(c.exact(), c.regex(), c.dense())) if c.has_codes() \
                                   else c.rerank(c.hybrid())

def _t_decompose_rerank(c): return c.rerank(c.decompose())            # deep: fan-out sub-queries
def _t_deep_hyde_decomp(c): return c.rerank(c.fuse(c.hyde(), c.decompose(), c.dense()))
def _t_deep_all(c):
    fused = c.fuse(c.dense(), c.keyword(), c.hyde(), c.decompose())
    return c.mmr(c.rerank(fused))


def _t_score_guarded(c):
    """Adaptive: cheap hybrid; if the top score fell off, escalate to the deep strategy."""
    base = c.hybrid()
    if c.weak(base):
        return c.rerank(c.fuse(c.hyde(), c.decompose(), base))
    return c.rerank(base)


def _t_escalating(c):
    """Cascade light -> medium -> deep, stopping as soon as a tier looks confident."""
    d = c.dense()
    if not c.weak(d):
        return c.rerank(d)
    h = c.hybrid()
    if not c.weak(h):
        return c.rerank(h)
    return c.rerank(c.fuse(c.hyde(), c.decompose(), h))


def _t_confidence_gated_exact(c):
    """For part-number queries: try exact; escalate to dense+hyde only if it's weak."""
    if c.has_codes():
        ex = c.fuse(c.exact(), c.regex())
        if not c.weak(ex):
            return c.rerank(ex)
        return c.rerank(c.fuse(ex, c.dense(), c.hyde()))
    return c.rerank(c.hyde())


TEMPLATES: dict[str, Callable[[StrategyContext], ResultSet]] = {
    # light (one-hop)
    "light_dense": _t_light_dense,
    "light_keyword": _t_light_keyword,
    "light_hybrid": _t_light_hybrid,
    "rephrase_rerank": _t_rephrase_rerank,
    # medium
    "dense_rerank": _t_dense_rerank,
    "hyde_rerank": _t_hyde_rerank,
    "mmr_diverse": _t_mmr_diverse,
    "prf_rerank": _t_prf_rerank,
    "multi_rephrase": _t_multi_rephrase,
    "exact_partnum": _t_exact_partnum,
    # deep
    "decompose_rerank": _t_decompose_rerank,
    "deep_hyde_decompose": _t_deep_hyde_decomp,
    "deep_all": _t_deep_all,
    # adaptive
    "score_guarded": _t_score_guarded,
    "escalating": _t_escalating,
    "confidence_gated_exact": _t_confidence_gated_exact,
}

TEMPLATE_NAMES = list(TEMPLATES)

# effort cost per tier — used to pick the CHEAPEST template that still solves a query
# (recall@k), so the router learns the lightest strategy that works.
_TIER_COST = {"light": 0, "medium": 1, "adaptive": 2, "deep": 3}

# What each strategy does and why it is a DISTINCT choice (tier, does, differs).
# Loaded by the router/prompt and rendered to docs/TEMPLATES.md.
TEMPLATE_DOCS: dict[str, dict] = {
    "light_dense": {"tier": "light",
        "does": "single vector search, no rerank",
        "differs": "cheapest path; baseline for easy semantic matches"},
    "light_keyword": {"tier": "light",
        "does": "BM25 term search only",
        "differs": "exact term matching — wins on rare tokens/IDs where embeddings blur"},
    "light_hybrid": {"tier": "light",
        "does": "RRF fuse of dense + keyword, one hop",
        "differs": "balances semantics and terms without any LLM or rerank cost"},
    "rephrase_rerank": {"tier": "light",
        "does": "rephrase the query once -> hybrid -> cross-encoder rerank",
        "differs": "the classic one-hop 'clean the query, then precision-rerank' flow"},
    "dense_rerank": {"tier": "medium",
        "does": "dense pool -> cross-encoder rerank",
        "differs": "adds precision on top of pure semantics; no lexical/LLM signal"},
    "hyde_rerank": {"tier": "medium",
        "does": "generate a hypothetical answer -> dense on it -> rerank",
        "differs": "bridges vocabulary gap when the query wording != the doc wording"},
    "mmr_diverse": {"tier": "medium",
        "does": "dense -> MMR diversification",
        "differs": "kills near-duplicate hits; for broad queries needing coverage not redundancy"},
    "prf_rerank": {"tier": "medium",
        "does": "pseudo-relevance feedback (enrich from top hits) -> rerank",
        "differs": "no LLM — uses the corpus itself to expand an under-specified query"},
    "multi_rephrase": {"tier": "medium",
        "does": "generate N rephrasings -> search each -> fuse -> rerank",
        "differs": "query-variation ensemble; catches docs that match only one phrasing"},
    "exact_partnum": {"tier": "medium",
        "does": "extract part-number tokens -> exact+regex, fuse with dense -> rerank",
        "differs": "the only strategy built for identifiers/pins/codes where exact beats semantics"},
    "decompose_rerank": {"tier": "deep",
        "does": "decompose into sub-queries -> fan-out -> fuse -> rerank",
        "differs": "for multi-part/multi-hop questions no single query retrieves"},
    "deep_hyde_decompose": {"tier": "deep",
        "does": "fuse(hyde, decompose, dense) -> rerank",
        "differs": "combines vocabulary-bridging + multi-hop; the heavy generalist for hard prose"},
    "deep_all": {"tier": "deep",
        "does": "fuse(dense, keyword, hyde, decompose) -> rerank -> MMR",
        "differs": "maximum recall net + precision + diversity; most expensive, last resort"},
    "score_guarded": {"tier": "adaptive",
        "does": "hybrid; if the top score falls off, escalate to hyde+decompose, else return",
        "differs": "spends deep effort ONLY when the cheap hop looks uncertain — the latency saver"},
    "escalating": {"tier": "adaptive",
        "does": "cascade dense -> hybrid -> deep, stop at the first confident tier",
        "differs": "finest-grained effort control; pays for exactly the depth each query needs"},
    "confidence_gated_exact": {"tier": "adaptive",
        "does": "part-number queries: try exact; escalate to dense+hyde only if weak",
        "differs": "adaptive around identifiers — cheap exact when confident, semantic backup when not"},
}

# templates whose strategy issues an LLM query-side op (rephrase/hyde/decompose/expand) — the
# real cost driver at label time. An LLM template is always dearer than any non-LLM one.
TEMPLATE_USES_LLM: set = {
    "rephrase_rerank", "hyde_rerank", "multi_rephrase", "decompose_rerank",
    "deep_hyde_decompose", "deep_all", "score_guarded", "escalating", "confidence_gated_exact",
}

# cost rank per template (cheapest first): tier + a large penalty for using the LLM. Drives both
# the winner policy (cheapest template that solves) and the cascade labeling order.
TEMPLATE_COST: dict[str, int] = {
    name: _TIER_COST.get(TEMPLATE_DOCS[name]["tier"], 5) + (10 if name in TEMPLATE_USES_LLM else 0)
    for name in TEMPLATE_NAMES
}


def run_template(name: str, ctx: StrategyContext, top_k: int = 10) -> list[str]:
    rs = TEMPLATES[name](ctx)
    return rs.ids()[:top_k] if rs else []
