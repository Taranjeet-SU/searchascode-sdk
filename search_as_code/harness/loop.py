"""The control loop — a bounded Plan–Execute–Verify cycle, plus helpers for subagents.

Plan–Execute–Verify is the standard agent-harness loop: try the planned skills in order, verify each
result against a (pluggable) reward, keep the best, stop as soon as one is accepted. Bounded by
``max_steps`` so it can't run away (context-drift / cost — the top harness failure modes).

``verify`` is intentionally pluggable: pass a gold-based check for eval, a teacher-reranker score in
production, or the default non-empty heuristic. This is where a *reliable* reward replaces the
unreliable LLM-self-judge.
"""
from __future__ import annotations

import re
from typing import Callable

from .._genutil import gen_lines
from .context import StepResult


def default_verify(ctx, ids) -> tuple:
    """Default reward — deliberately WEAK, and now honest about it.

    This returns a *graded* score from the only evidence available without gold: how full the
    result list is relative to what was asked for. It is **not** a relevance signal.

    It used to return ``1.0`` for any non-empty list, which had three consequences the
    docstrings called "evidence-backed" online learning (SDK-A3):
      1. ``plan_execute_verify`` always broke on the first skill (score >= accept), so the
         bounded Plan-Execute-Verify loop was permanently single-step and ``max_steps`` never
         bound;
      2. ``post_write_memory`` wrote "skill X worked" on every non-empty run, biasing later
         plans toward whatever ran first;
      3. ``reflect()`` gated forging on ``score < threshold`` (0.5) — never true — so every
         non-empty multi-hop run forged a skill and a subagent.
    A graded score below the 0.75 accept bar keeps the loop actually iterating, and callers
    that have real signal should pass their own ``verify`` (gold, a teacher reranker, or
    ``DiagnosticJudge``). ``Harness(verify=...)`` and ``plan_execute_verify(verify=...)`` are
    the injection points.
    """
    if not ids:
        return (False, 0.0)
    want = getattr(ctx, "top_k", None) or 10
    coverage = min(1.0, len(ids) / float(want))
    # Cap below `accept` (0.75): with no relevance signal we can never *confirm* success.
    return (True, round(min(0.7, 0.2 + 0.5 * coverage), 3))


def plan_execute_verify(ctx, execute: Callable, verify: Callable, max_steps: int = 3,
                        accept: float = 0.75, verified: bool = False) -> tuple:
    """Try ctx.plan skills in order (bounded); return (best StepResult, [all steps]).

    ``verified`` says whether ``verify`` is a real reward signal (gold / teacher / judge) as
    opposed to :func:`default_verify`. It is recorded on each step so the online-learning
    hooks can refuse to learn from noise (SDK-A3).
    """
    best = StepResult(skill="none", ids=[], ok=False, score=-1.0)
    steps = []
    for skill in ctx.plan[:max_steps]:
        ids = execute(skill)
        ok, score = verify(ids)
        sr = StepResult(skill=skill, ids=ids, ok=ok, score=score, verified=verified)
        steps.append(sr)
        if score > best.score:
            best = sr
        if ok and score >= accept:
            break
    return best, steps


def fuse_ids(lists, k: int = 60) -> list:
    """Reciprocal-rank fusion over ranked **id lists** — the coverage-preserving union.

    THE single implementation for id-list RRF. The audit counted eight copies of
    ``1.0 / (k + rank + 1)`` across the repo (SDK-R2 / P1-11 / LEG-6); ``playbook._rrf``,
    ``forge._safe_globals._fuse_ids`` and ``agentic``'s rebinding now all delegate here, so a
    change to the fusion rule takes effect everywhere instead of in one of eight places.
    (``primitives.fuse`` is the sibling that fuses ResultSets and keeps scores/documents.)
    """
    scores: dict = {}
    for lst in lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


_SPLIT_RE = re.compile(r"\band\b|\bor\b|[,;]|\bboth\b|\bcompared? (?:to|with)\b|\bversus\b|\bvs\.?\b", re.I)


def decompose_query(query: str, generator=None, max_subs: int = 4) -> list:
    """Split a multi-hop query into sub-questions — via the generator if available, else lexically.

    Each sub-question becomes a subagent task (retrieve its own docs), then results are fused."""
    if generator is not None:
        try:
            prompt = (f"Split this multi-hop question into the distinct factual sub-questions needed "
                      f"to answer it (each targets a DIFFERENT document). One per line, {max_subs} max.\n\nQ: {query}")
            out = generator(prompt)
            subs = gen_lines(out, max_items=max_subs, min_len=3)
            if len(subs) >= 2:
                return subs
        except Exception:
            pass
    parts = [p.strip(" ?.") for p in _SPLIT_RE.split(query) if p and len(p.strip()) > 3]
    return (parts[:max_subs] or [query]) if len(parts) >= 2 else [query]
