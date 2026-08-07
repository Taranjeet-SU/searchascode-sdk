"""The Harness — a self-improving agentic retrieval harness.

Ties the pieces together into one Plan–Execute–Verify runtime with the four necessary harness
elements (loop, tool/skill interface, context+memory, control) plus the upgrades our experiments and
the 2026 harness-engineering / Anthropic-Agent-Skills / prime-agent Continual-Harness patterns point
to:

  - **triage** the query (single lookup vs multi-hop vs error-code) — right effort per query;
  - **skills** with progressive disclosure (a growable enterprise skill library);
  - **memory** that recalls what worked (cross-session) and writes new wins (compounding);
  - **dynamic prompt** assembled pre-loop from intent + memory + skill catalog;
  - **subagents** for multi-hop — one child harness per decomposed sub-question, results fused;
  - **pluggable verify** (reward) so a gold/teacher signal can replace the unreliable self-judge.

    from search_as_code.harness import Harness
    h = Harness(session)                       # memory + skills auto-wired from the session
    r = h.run("Which two films share a director, and what year did each release?")
    r.ids            # fused doc ids
    r.skill          # 'subagents' (multi-hop) or the winning skill name
    r.subagents      # per-sub-question traces
    r.dynamic_prompt # the pre-loop prompt an LLM agent would consume
"""
from __future__ import annotations

from typing import Optional

from .context import HarnessContext, HarnessResult
from .hooks import DEFAULT_POST_HOOKS, DEFAULT_PRE_HOOKS
from .loop import decompose_query, default_verify, fuse_ids, plan_execute_verify
from .memory import AgentMemory
from .skills import SkillRegistry

BASE_PROMPT = ("You are a retrieval agent. Choose the cheapest skill that fits the query intent; "
               "for multi-hop questions, cover each sub-fact and fuse — do not fan out blindly.")


class Harness:
    def __init__(self, session=None, *, memory: Optional[AgentMemory] = None,
                 skills: Optional[SkillRegistry] = None, generator=None, verify=None,
                 pre_hooks=None, post_hooks=None, max_steps: int = 3, base_prompt: str = BASE_PROMPT,
                 use_subagents: bool = True, memory_path: Optional[str] = None,
                 _depth: int = 0, max_depth: int = 1):
        self.session = session
        emb = getattr(session, "embedder", None) if session is not None else None
        self.memory = memory or AgentMemory(path=memory_path, embedder=emb)
        self.skills = skills or SkillRegistry(embedder=emb)
        self.generator = generator or getattr(session, "generator", None)
        self.verify = verify or default_verify
        self.pre_hooks = pre_hooks if pre_hooks is not None else DEFAULT_PRE_HOOKS
        self.post_hooks = post_hooks if post_hooks is not None else DEFAULT_POST_HOOKS
        self.max_steps = max_steps
        self.base_prompt = base_prompt
        self.use_subagents = use_subagents
        self._depth = _depth
        self.max_depth = max_depth

    # ---- public ------------------------------------------------------------
    def run(self, query: str, top_k: int = 10) -> HarnessResult:
        ctx = HarnessContext(query=query, session=self.session, memory=self.memory,
                             skills=self.skills, generator=self.generator, top_k=top_k)
        ctx.scratch["base_prompt"] = self.base_prompt
        for hook in self.pre_hooks:          # PRE: triage → recall memory → select skills → prompt
            hook(ctx)
        dynamic_prompt = ctx.scratch.get("dynamic_prompt", "")

        if (self.use_subagents and ctx.intent is not None and ctx.intent.depth == "multi"
                and self._depth < self.max_depth):
            result = self._run_subagents(ctx, top_k)          # multi-hop → subagents
        else:
            result = self._run_loop(ctx, top_k)               # single → Plan-Execute-Verify

        result.dynamic_prompt = dynamic_prompt
        result.intent = ctx.intent.kind if ctx.intent is not None else ""
        ctx.result = result
        for hook in self.post_hooks:         # POST: write memory / refine
            hook(ctx)
        return result

    def child(self) -> "Harness":
        """A subagent — a child harness sharing the same session, skills, and memory (bounded depth)."""
        return Harness(self.session, memory=self.memory, skills=self.skills, generator=self.generator,
                       verify=self.verify, pre_hooks=self.pre_hooks, post_hooks=[], max_steps=self.max_steps,
                       base_prompt=self.base_prompt, use_subagents=self.use_subagents,
                       _depth=self._depth + 1, max_depth=self.max_depth)

    def spawn(self, task: str, top_k: int = 10) -> HarnessResult:
        """Spawn a subagent on a sub-task (function-call style)."""
        return self.child().run(task, top_k)

    # ---- internal ----------------------------------------------------------
    def _run_loop(self, ctx, top_k) -> HarnessResult:
        def execute(name):
            sk = self.skills.get(name)
            try:
                return sk.run(self.session, ctx.query, top_k=top_k) if sk else []
            except Exception:
                return []
        best, steps = plan_execute_verify(ctx, execute, lambda ids: self.verify(ctx, ids),
                                          max_steps=self.max_steps)
        return HarnessResult(ids=best.ids, skill=best.skill, score=best.score, steps=steps)

    def _run_subagents(self, ctx, top_k) -> HarnessResult:
        subs = decompose_query(ctx.query, self.generator)
        sub_traces, pools = [], []
        for sub in subs:
            r = self.spawn(sub, top_k=top_k)
            pools.append(r.ids)
            sub_traces.append({"query": sub, "ids": r.ids, "skill": r.skill})
            # cross-hop memory: write each sub-question's FINDING into the shared memory so the NEXT
            # subagent (and later hops) recall it — not just the query text.
            if self.memory is not None:
                self.memory.observe(f"sub-question \"{sub[:70]}\" -> found {r.ids[:3]} via {r.skill}",
                                    kind="finding")
        pools.append(self._run_loop(ctx, top_k).ids)          # also the full question
        fused = fuse_ids(pools)[:top_k]
        return HarnessResult(ids=fused, skill="subagents", score=1.0 if fused else 0.0,
                             subagents=sub_traces, meta={"n_subagents": len(subs)})
