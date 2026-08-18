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
from .forge import HarnessForge, HarnessStore, reflect
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
                 store: Optional[HarnessStore] = None, store_path: Optional[str] = None,
                 learn: bool = False, _depth: int = 0, max_depth: int = 1):
        self.session = session
        emb = getattr(session, "embedder", None) if session is not None else None
        self.memory = memory or AgentMemory(path=memory_path, embedder=emb)
        self.skills = skills or SkillRegistry(embedder=emb)
        # self-modifiable state: load forged skills/subagents/rules + register them (usable online)
        self.store = store if store is not None else HarnessStore(store_path)
        self.forge = HarnessForge(self.store, self.skills, self.memory)
        self.learn = learn
        self.generator = generator or getattr(session, "generator", None)
        # `verified` marks whether the reward is a REAL signal. default_verify is not one
        # (it cannot see relevance), so online learning must not treat its score as
        # evidence — see loop.default_verify and reflect() (SDK-A3).
        self.verify = verify or default_verify
        self.verify_is_real = verify is not None
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
        # immutable base + self-modifiable supplemental prompt (learned rules) — Continual-Harness style
        learn_block = self.store.learnings_block() if self.store else ""
        ctx.scratch["base_prompt"] = self.base_prompt + (("\n\n" + learn_block) if learn_block else "")
        for hook in self.pre_hooks:          # PRE: triage → recall memory → select skills → prompt
            hook(ctx)
        dynamic_prompt = ctx.scratch.get("dynamic_prompt", "")

        # if memory recalled a FORGED skill to the top of the plan, use it (replicate the learned
        # module) instead of re-deriving via generic subagents.
        top = ctx.plan[0] if ctx.plan else None
        prefer_forged = top is not None and top in getattr(self.store, "skills", {})
        if (self.use_subagents and ctx.intent is not None and ctx.intent.depth == "multi"
                and self._depth < self.max_depth and not prefer_forged):
            result = self._run_subagents(ctx, top_k)          # multi-hop → subagents
        else:
            result = self._run_loop(ctx, top_k)               # single / forged-skill → Plan-Execute-Verify

        result.dynamic_prompt = dynamic_prompt
        result.intent = ctx.intent.kind if ctx.intent is not None else ""
        ctx.result = result
        for hook in self.post_hooks:         # POST: write memory / refine
            hook(ctx)
        if self.learn:                       # ONLINE self-improvement: forge skills/subagents/rules
            result.meta["forged"] = reflect(ctx, result, self.forge)
        return result

    def child(self) -> "Harness":
        """A subagent — a child harness sharing the same session, skills, and memory (bounded depth)."""
        return Harness(self.session, memory=self.memory, skills=self.skills, generator=self.generator,
                       verify=self.verify, pre_hooks=self.pre_hooks, post_hooks=[], max_steps=self.max_steps,
                       base_prompt=self.base_prompt, use_subagents=self.use_subagents,
                       store=self.store, learn=False,       # subagents reuse forged skills; only parent learns
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
                                          max_steps=self.max_steps,
                                          verified=self.verify_is_real)
        return HarnessResult(ids=best.ids, skill=best.skill, score=best.score, steps=steps,
                             verified=self.verify_is_real)

    def _run_subagents(self, ctx, top_k) -> HarnessResult:
        subs = decompose_query(ctx.query, self.generator)
        arsenal = self.skills.get("arsenal_single")           # each subagent uses the full arsenal on its sub-fact
        wide = max(top_k * 3, 30)                              # keep sub-pools WIDE so the fuse preserves coverage
        sub_traces, pools = [], []
        for sub in subs:
            if arsenal is not None:
                ids = arsenal.run(self.session, sub, top_k=wide)
                skill = "arsenal_single"
            else:
                r = self.spawn(sub, top_k=top_k)
                ids, skill = r.ids, r.skill
            pools.append(ids)
            sub_traces.append({"query": sub, "ids": ids[:top_k], "skill": skill})
            # cross-hop memory: write each sub-fact's FINDING so later hops recall it (not just the query)
            if self.memory is not None:
                self.memory.observe(f"sub-question \"{sub[:70]}\" -> found {ids[:3]} via {skill}",
                                    kind="finding")
        if arsenal is not None:
            pools.append(arsenal.run(self.session, ctx.query, top_k=wide))    # whole query, full arsenal
        fused = fuse_ids(pools)[:top_k]
        # Score the subagent path through the SAME verifier as the loop path. It used to
        # fabricate `1.0 if fused else 0.0`, which made every non-empty multi-hop run look
        # like a confirmed win to reflect() and post_write_memory (SDK-A3).
        ok, score = self.verify(ctx, fused)
        return HarnessResult(ids=fused, skill="subagents", score=score,
                             verified=self.verify_is_real,
                             subagents=sub_traces, meta={"n_subagents": len(subs)})
