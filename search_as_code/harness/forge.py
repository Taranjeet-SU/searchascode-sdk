"""The Forge — the harness's self-modification toolbox (the paper's "second iteration").

After the agent solves a problem, it can **create/modify what it needs from that learning** and
**persist** it so future queries use it online:

  - ``create_skill``      : compose existing retrievers into a NEW named skill (a recipe) + register it
  - ``create_primitive``  : alias of create_skill (a reusable composed retrieval primitive)
  - ``create_subagent``   : define a specialized subagent (a skill-plan + base prompt) for a query type
  - ``refine_prompt``     : append an evidence-backed rule to the self-modifiable supplemental prompt
  - ``remember``          : write a durable fact/win to long-term memory

Everything is persisted to a :class:`HarnessStore` directory (``skills.jsonl``, ``subagents.jsonl``,
``learnings.md``, memory alongside) and reloaded next session — **online, cross-session self-improvement**.
Created skills are *compositions of existing retrievers* (a safe recipe DSL — no arbitrary code exec):
the agent's novelty is in *which primitives to combine*, exactly the lever our experiments found.

``reflect()`` is the online-learning step run after each solve: rule-based (deterministic, testable)
or LLM-proposed — it decides what to forge from the trajectory.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .loop import fuse_ids
from .skills import Skill

# retriever aliases → built-in skill names the recipe composes
_ALIAS = {"dense": "dense_lookup", "keyword": "keyword_search", "hybrid": "hybrid_search",
          "hyde": "hyde_bridge", "prf": "prf_expand", "exact": "exact_lookup",
          "decompose": "decompose_fuse", "rerank": "rerank_precise", "mmr": "diversify"}


@dataclass
class LearnedSkill:
    name: str
    when_to_use: str
    retrievers: list                # e.g. ["dense", "keyword", "hyde"] — composed via fuse
    combine: str = "fuse"           # fuse | first
    cost: int = 1
    origin: str = "forged"          # forged | llm | user
    tags: list = field(default_factory=lambda: ["learned"])

    def to_skill(self, registry) -> Skill:
        retrievers, combine = self.retrievers, self.combine

        def run(session, query, top_k=10, **_):
            pools = []
            for r in retrievers:
                sk = registry.get(_ALIAS.get(r, r))
                if sk is not None:
                    try:
                        pools.append(sk.run(session, query, top_k=max(top_k, 20)))
                    except Exception:
                        pass
            if not pools:
                return []
            ids = fuse_ids(pools) if combine == "fuse" else pools[0]
            return ids[:top_k]

        return Skill(self.name, self.when_to_use, run, tags=self.tags, cost=self.cost,
                     description=f"forged: {combine}({'+'.join(retrievers)})")


@dataclass
class LearnedSubagent:
    name: str
    when_to_use: str
    plan: list                      # ordered skill names the subagent tries
    base_prompt: str = ""
    tags: list = field(default_factory=lambda: ["learned"])


class HarnessStore:
    """Persisted self-modifiable state: forged skills, subagents, and learned prompt rules."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.skills: dict[str, LearnedSkill] = {}
        self.subagents: dict[str, LearnedSubagent] = {}
        self.learnings: list[str] = []
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        p = self.path
        if (p / "skills.jsonl").exists():
            for ln in (p / "skills.jsonl").read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln); self.skills[d["name"]] = LearnedSkill(**d)
        if (p / "subagents.jsonl").exists():
            for ln in (p / "subagents.jsonl").read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln); self.subagents[d["name"]] = LearnedSubagent(**d)
        if (p / "learnings.md").exists():
            self.learnings = [x[2:].strip() for x in (p / "learnings.md").read_text().splitlines()
                              if x.strip().startswith("- ")]

    def save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "skills.jsonl").open("w") as f:
            for s in self.skills.values():
                f.write(json.dumps(asdict(s)) + "\n")
        with (self.path / "subagents.jsonl").open("w") as f:
            for s in self.subagents.values():
                f.write(json.dumps(asdict(s)) + "\n")
        (self.path / "learnings.md").write_text(
            "# Learned rules (self-modifiable supplemental prompt)\n\n"
            + "\n".join(f"- {r}" for r in self.learnings))

    def learnings_block(self, max_rules: int = 12) -> str:
        return ("LEARNED RULES (refined from past runs):\n"
                + "\n".join(f"- {r}" for r in self.learnings[-max_rules:])) if self.learnings else ""


class HarnessForge:
    """The primitives the agent calls to create/modify itself, then persist — online self-improvement."""

    def __init__(self, store: HarnessStore, registry, memory=None):
        self.store = store
        self.registry = registry
        self.memory = memory
        for ls in store.skills.values():            # register any previously-forged skills
            self.registry.register(ls.to_skill(self.registry))

    def create_skill(self, name: str, when_to_use: str, retrievers, combine: str = "fuse",
                     cost: int = 1, origin: str = "forged") -> str:
        ls = LearnedSkill(name=name, when_to_use=when_to_use, retrievers=list(retrievers),
                          combine=combine, cost=cost, origin=origin)
        self.store.skills[name] = ls
        self.registry.register(ls.to_skill(self.registry))       # available online, this run
        self.store.save()
        return name

    # a forged "primitive" is a composed reusable retrieval recipe — same mechanism as a skill
    create_primitive = create_skill

    def create_subagent(self, name: str, when_to_use: str, plan, base_prompt: str = "") -> str:
        self.store.subagents[name] = LearnedSubagent(name=name, when_to_use=when_to_use,
                                                     plan=list(plan), base_prompt=base_prompt)
        self.store.save()
        return name

    def refine_prompt(self, rule: str) -> None:
        rule = rule.strip()
        if rule and rule not in self.store.learnings:
            self.store.learnings.append(rule)
            self.store.save()

    def remember(self, fact: str, **meta):
        if self.memory is not None:
            return self.memory.remember(fact, **meta)


def reflect(ctx, result, forge: HarnessForge, threshold: float = 0.5) -> list:
    """Online-learning step: after a solve, create/modify skills/subagents/rules/memory from evidence.

    Rule-based + deterministic (LLM proposals can extend it). Returns the names of forged artifacts."""
    created = []
    intent = getattr(ctx, "intent", None)
    kind = intent.kind if intent is not None else "unknown"
    if not result.ids or result.score < threshold:
        return created

    # 1) always remember the win (durable memory)
    forge.remember(f"query like \"{ctx.query[:100]}\" ({kind}) -> '{result.skill}' worked",
                   kind="skill_win", skill=result.skill, intent=kind)

    # 2) a multi-hop plan that worked → forge a named composed skill for this intent (if novel)
    if result.skill == "subagents":
        name = f"learned_multihop_{kind}"
        if name not in forge.store.skills:
            forge.create_skill(name, f"multi-hop {kind} queries needing several docs",
                               retrievers=["decompose", "dense", "keyword"], combine="fuse", cost=2)
            forge.create_subagent(f"sub_{kind}", f"a sub-question of a {kind} query",
                                  plan=["dense_lookup", "keyword_search"])
            created += [name, f"sub_{kind}"]

    # 3) error-code / definition wins → refine the supplemental prompt with a rule
    if kind == "error_code":
        forge.refine_prompt("For error-code / ID queries, try exact_lookup before semantic search.")
    elif kind == "multi_hop":
        forge.refine_prompt("For multi-hop queries, decompose into sub-questions and FUSE — do not "
                            "rerank the fused union (it drops per-sub-fact coverage).")
    return created
