"""Skills — Anthropic-style agent skills for retrieval, with a registry + progressive disclosure.

A **skill** is a named, self-describing retrieval capability: ``name``, ``when_to_use`` (a few dozen
tokens the agent reads to decide), ``tags``, ``cost``, and a ``run(session, query, top_k) -> ids``.
The **registry** exposes only the short summaries by default (progressive disclosure — full detail
loads on demand), and can ``find`` the best skills for a query (semantic over ``when_to_use``, else
lexical). This is the open Agent-Skills pattern applied to retrieval, so an enterprise skill library
can grow without bloating the prompt.

Built-in skills are thin, composable wrappers over the SDK ``Session`` primitives — the same cheap
primitives our experiments showed carry the value (decompose→fuse for multi-hop; exact for IDs;
HyDE for vocabulary gaps).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .._genutil import gen_lines
from .triage import extract_codes


@dataclass
class Skill:
    name: str
    when_to_use: str                    # short trigger description (progressive-disclosure summary)
    run: Callable                       # (session, query, top_k=10, **kw) -> list[str] ids
    tags: list = field(default_factory=list)
    cost: int = 0                       # 0 cheap ... 3 deep (tier), for budget-aware planning
    description: str = ""

    def summary(self) -> str:
        return f"{self.name} (cost {self.cost}): {self.when_to_use}"


def _ids(rs, top_k):
    if rs is None:
        return []
    if hasattr(rs, "ids"):
        return rs.ids()[:top_k]
    return [str(x) for x in rs][:top_k]


# --------------------------------------------------------------------------- #
# built-in retrieval skills (thin wrappers over Session primitives)           #
# --------------------------------------------------------------------------- #
def _dense(session, query, top_k=10, **_):
    return _ids(session.search(query, top_k=top_k, mode="dense"), top_k)


def _hybrid(session, query, top_k=10, **_):
    return _ids(session.search(query, top_k=top_k, mode="hybrid"), top_k)


def _keyword(session, query, top_k=10, **_):
    return _ids(session.search(query, top_k=top_k, mode="keyword"), top_k)


def _exact(session, query, top_k=10, **_):
    """Error-codes / IDs: exact-phrase + regex on the code tokens, fused with a keyword pass."""
    from .. import primitives as P
    codes = extract_codes(query)
    if not codes:
        return _hybrid(session, query, top_k)
    pools, qp = [], getattr(session.store, "query_phrase", None)
    for c in codes:
        try:
            pools.append(qp(c, top_k=top_k) if qp else session.search(c, top_k, mode="keyword"))
        except Exception:
            pass
    pools.append(session.search(query, top_k, mode="keyword"))
    return _ids(P.fuse([p for p in pools if p]), top_k)


def _decompose_fuse(session, query, top_k=10, **_):
    """Multi-hop: decompose into sub-questions, retrieve each, RRF-fuse (coverage-preserving)."""
    try:
        return _ids(session.decompose_search(query, top_k=top_k), top_k)
    except Exception:
        return _dense(session, query, top_k)


def _decompose_fielded(session, query, top_k=10, **_):
    """Multi-hop via per-sub-fact FIELDED match: decompose → query_fielded(sub, [title,text]) + dense
    per sub → fuse. The diagnostic showed multi-hop golds ARE reachable by title/text fielded match;
    this composes that per sub-fact so each entity gets its own strong retrieval, then fuses for
    coverage (the fix for 4-hop queries a question-level dense pass misses)."""
    from .. import primitives as P
    gen = getattr(session, "generator", None)
    subs = []
    if gen is not None:
        try:
            out = gen("Break this question into the distinct factual sub-questions needed to answer "
                      "it — each targets a DIFFERENT entity/document. One per line, 2-6.\n\nQ: " + query)
            subs = gen_lines(out, max_items=6, min_len=3)
        except Exception:
            pass
    subs = subs or [query]
    qf = getattr(session.store, "query_fielded", None)
    pools = []
    for s in subs + [query]:
        try:
            pools.append(qf(s, ["title", "text"], top_k=max(top_k, 20)) if qf
                         else session.search(s, top_k=max(top_k, 20), mode="keyword"))
        except Exception:
            pass
        try:
            pools.append(session.search(s, top_k=max(top_k, 20), mode="dense"))
        except Exception:
            pass
    return _ids(P.fuse([p for p in pools if p]), top_k) if pools else _dense(session, query, top_k)


def _arsenal_single(session, query, top_k=10, **_):
    """Full single-query arsenal: hybrid + HyDE + fielded(title,text), RRF-fused. HyDE bridges a
    GENERIC description (hallucinate the answer doc, search its embedding — the fix for entities the
    query only describes), fielded catches named entities, hybrid balances semantics+terms."""
    from .. import primitives as P
    pools = []
    for fn in (lambda: session.search(query, top_k=max(top_k, 30), mode="hybrid"),
               lambda: session.hyde_search(query, top_k=max(top_k, 30)),
               lambda: (session.store.query_fielded(query, ["title", "text"], top_k=max(top_k, 30))
                        if getattr(session.store, "query_fielded", None) else None)):
        try:
            rs = fn()
            if rs:
                pools.append(rs)
        except Exception:
            pass
    return _ids(P.fuse(pools), top_k) if pools else _dense(session, query, top_k)


def _decompose_arsenal(session, query, top_k=10, **_):
    """MULTI-HOP (validated): decompose → the full arsenal (hybrid+HyDE+fielded) per sub-fact AND the
    whole query → RRF fuse. Recovers 4-hop golds a dense/keyword decompose misses (HyDE reaches the
    generically-described entity)."""
    from .. import primitives as P
    gen = getattr(session, "generator", None)
    subs = []
    if gen is not None:
        try:
            out = gen("Break this question into the distinct factual sub-questions needed to answer "
                      "it — each targets a DIFFERENT entity/document. One per line, 2-6.\n\nQ: " + query)
            subs = gen_lines(out, max_items=6, min_len=3)
        except Exception:
            pass
    subs = subs or [query]
    qf = getattr(session.store, "query_fielded", None)
    pools = []
    for x in subs + [query]:
        for fn in (lambda x=x: session.search(x, top_k=max(top_k, 30), mode="hybrid"),
                   lambda x=x: session.hyde_search(x, top_k=max(top_k, 30)),
                   lambda x=x: (qf(x, ["title", "text"], top_k=max(top_k, 30)) if qf else None)):
            try:
                rs = fn()
                if rs:
                    pools.append(rs)
            except Exception:
                pass
    return _ids(P.fuse(pools), top_k) if pools else _dense(session, query, top_k)


def _hyde(session, query, top_k=10, **_):
    try:
        return _ids(session.hyde_search(query, top_k=top_k), top_k)
    except Exception:
        return _dense(session, query, top_k)


def _prf(session, query, top_k=10, **_):
    try:
        return _ids(session.prf_search(query, top_k=top_k), top_k)
    except Exception:
        return _dense(session, query, top_k)


def _rerank_precise(session, query, top_k=10, pool_k=50, **_):
    """Wide dense pool → cross-encoder rerank (single-answer precision; may drop multi-gold coverage)."""
    try:
        pool = session.search(query, top_k=pool_k, mode="dense")
        return _ids(session.rerank(query, pool, top_k=top_k), top_k)
    except Exception:
        return _dense(session, query, top_k)


def _diversify(session, query, top_k=10, pool_k=40, **_):
    try:
        pool = session.search(query, top_k=pool_k, mode="dense")
        return _ids(session.mmr(query, pool, top_k=top_k), top_k)
    except Exception:
        return _dense(session, query, top_k)


BUILTIN_SKILLS = [
    Skill("dense_lookup", "default semantic search for a single focused question", _dense, ["core"], 0),
    Skill("definition_lookup", "a 'what is / define X' question with one clear answer", _hybrid, ["core"], 0),
    Skill("hybrid_search", "broad/open-ended query; balance semantics + terms", _hybrid, ["core"], 0),
    Skill("keyword_search", "rare exact tokens where embeddings blur", _keyword, ["core"], 0),
    Skill("exact_lookup", "error/status codes, part numbers, IDs — exact match beats semantics", _exact, ["ids"], 1),
    Skill("decompose_fuse", "MULTI-HOP: needs several docs; split into sub-facts, retrieve each, fuse", _decompose_fuse, ["multihop"], 3),
    Skill("decompose_fielded", "MULTI-HOP over named entities: split into sub-facts, fielded title+text match + dense per sub, fuse", _decompose_fielded, ["multihop"], 3),
    Skill("arsenal_single", "hard single lookup: hybrid + HyDE + fielded, fused (HyDE for generic descriptions)", _arsenal_single, ["arsenal"], 2),
    Skill("decompose_arsenal", "MULTI-HOP (best): decompose, then hybrid+HyDE+fielded per sub-fact, RRF-fused", _decompose_arsenal, ["multihop", "arsenal"], 3),
    Skill("hyde_bridge", "vocabulary gap: the query wording differs from the corpus wording", _hyde, ["vocab"], 2),
    Skill("prf_expand", "under-specified query; expand from the corpus's own top hits (no LLM)", _prf, ["expand"], 1),
    Skill("rerank_precise", "single best answer needed; rerank a wide pool for precision", _rerank_precise, ["precision"], 2),
    Skill("diversify", "broad query needing coverage of distinct facets, not near-duplicates", _diversify, ["coverage"], 1),
]


class SkillRegistry:
    """A registry of skills with progressive disclosure + semantic lookup (the skill 'search API')."""

    def __init__(self, skills=None, embedder=None):
        self._skills: dict[str, Skill] = {}
        self._embed = embedder.embed if (embedder is not None and hasattr(embedder, "embed")) else embedder
        self._vecs: Optional[np.ndarray] = None
        for s in (skills if skills is not None else BUILTIN_SKILLS):
            self.register(s)

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        self._vecs = None                       # invalidate cache

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def names(self) -> list:
        return list(self._skills)

    def summaries(self, tags: Optional[set] = None) -> str:
        """Progressive-disclosure catalog — one short line per skill (a few dozen tokens each)."""
        return "\n".join(s.summary() for s in self._skills.values()
                         if tags is None or (set(s.tags) & tags))

    def find(self, query: str, k: int = 3) -> list:
        """Best skills for a query — semantic over when_to_use if an embedder is set, else lexical."""
        skills = list(self._skills.values())
        if self._embed:
            try:
                if self._vecs is None:
                    self._vecs = np.asarray(self._embed([s.when_to_use for s in skills]), dtype=np.float32)
                qv = np.asarray(self._embed([query])[0], dtype=np.float32)
                qv = qv / (np.linalg.norm(qv) + 1e-9)
                M = self._vecs / (np.linalg.norm(self._vecs, axis=1, keepdims=True) + 1e-9)
                order = np.argsort(-(M @ qv))[:k]
                return [skills[i] for i in order]
            except Exception:
                pass
        def _tok(t):
            return set(re.findall(r"[a-z0-9]+", (t or "").lower()))

        qtok = _tok(query)
        return sorted(skills, key=lambda s: -len(qtok & _tok(s.when_to_use + " " + " ".join(s.tags))))[:k]
