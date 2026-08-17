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
    """Default reward: accept a non-empty result. Replace with gold / teacher-reranker for real signal."""
    return (bool(ids), 1.0 if ids else 0.0)


def plan_execute_verify(ctx, execute: Callable, verify: Callable, max_steps: int = 3,
                        accept: float = 0.75) -> tuple:
    """Try ctx.plan skills in order (bounded); return (best StepResult, [all steps])."""
    best = StepResult(skill="none", ids=[], ok=False, score=-1.0)
    steps = []
    for skill in ctx.plan[:max_steps]:
        ids = execute(skill)
        ok, score = verify(ids)
        sr = StepResult(skill=skill, ids=ids, ok=ok, score=score)
        steps.append(sr)
        if score > best.score:
            best = sr
        if ok and score >= accept:
            break
    return best, steps


def fuse_ids(lists, k: int = 60) -> list:
    """Reciprocal-rank fusion over ranked id lists (coverage-preserving union for subagent results)."""
    s: dict = {}
    for lst in lists:
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(s, key=lambda i: -s[i])


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
