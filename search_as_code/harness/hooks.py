"""Pre/post-loop hooks + dynamic prompt assembly.

Hooks are ``Callable[[HarnessContext], None]`` run before (pre) and after (post) the control loop —
the standard extension points of a good harness. The defaults implement:

  PRE  : triage the query → recall relevant long-term memory → select skills → assemble a
         **dynamic prompt** (base + intent + memory + skill catalog) an LLM agent can consume.
  POST : write the outcome to memory (working + long-term "what worked"), so the harness improves
         across runs (the compounding-memory property a static router lacks).
"""
from __future__ import annotations

from .triage import triage


# ---- PRE-loop hooks --------------------------------------------------------
def pre_triage(ctx) -> None:
    ctx.intent = triage(ctx.query)
    ctx.prompt_parts.append(f"QUERY INTENT: {ctx.intent}")


def pre_recall_memory(ctx) -> None:
    if ctx.memory is None:
        return
    ctx.recalled = ctx.memory.recall(ctx.query, k=4)
    if ctx.recalled:
        lines = "\n".join(f"- {m.content}" for m in ctx.recalled)
        ctx.prompt_parts.append(f"RELEVANT MEMORY (what worked on similar queries):\n{lines}")
    # cross-hop: surface in-session findings from earlier hops/subagents into this run's prompt
    findings = ctx.memory.working_context(max_chars=600, kinds={"finding", "outcome"})
    if findings:
        ctx.prompt_parts.append(f"IN-SESSION FINDINGS (earlier hops/subagents):\n{findings}")
    ctx.memory.observe(ctx.query, kind="query")


def pre_select_skills(ctx) -> None:
    """Plan: the triage-recommended skill first, then the best skills the registry finds."""
    plan = []
    if ctx.intent is not None:
        plan.append(ctx.intent.recommended_skill)
    if ctx.skills is not None:
        for s in ctx.skills.find(ctx.query, k=3):
            if s.name not in plan:
                plan.append(s.name)
        ctx.prompt_parts.append("AVAILABLE SKILLS:\n" + ctx.skills.summaries())
    # bias toward a memory-recalled winning skill, if any
    if ctx.memory is not None:
        for m in ctx.recalled:
            w = m.meta.get("skill")
            if w and w in (ctx.skills.names() if ctx.skills else []):
                plan = [w] + [p for p in plan if p != w]
                break
    ctx.plan = plan or ["dense_lookup"]


def pre_assemble_prompt(ctx) -> None:
    header = ctx.scratch.get("base_prompt", "").strip()
    ctx.scratch["dynamic_prompt"] = ("\n\n".join([header] + ctx.prompt_parts)).strip()


DEFAULT_PRE_HOOKS = [pre_triage, pre_recall_memory, pre_select_skills, pre_assemble_prompt]


# ---- POST-loop hooks -------------------------------------------------------
def post_write_memory(ctx) -> None:
    if ctx.memory is None or ctx.result is None:
        return
    r = ctx.result
    ctx.memory.observe(f"{ctx.intent.kind if ctx.intent else '?'} -> {r.skill} "
                       f"(score {r.score:.2f}, {len(r.ids)} hits)", kind="outcome")
    # long-term: remember the winning strategy so future similar queries recall it
    if r.ids and r.score >= 0.5:
        ctx.memory.remember(f"For a '{ctx.intent.kind if ctx.intent else '?'}' query like "
                            f"\"{ctx.query[:120]}\", the skill '{r.skill}' worked.",
                            kind="skill_win", skill=r.skill,
                            intent=(ctx.intent.kind if ctx.intent else ""))


DEFAULT_POST_HOOKS = [post_write_memory]
