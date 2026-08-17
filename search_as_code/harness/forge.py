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
import re
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


def _safe_globals():
    """Restricted globals for executing an LLM-authored primitive (retrieval logic only)."""
    import importlib
    import re as _re

    import search_as_code as sac
    from search_as_code import primitives as P
    _bi = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    allowed = {k: _bi[k] for k in ("len", "range", "list", "dict", "set", "tuple", "sorted", "min",
               "max", "enumerate", "zip", "str", "int", "float", "bool", "sum", "any", "all", "map",
               "filter", "reversed", "abs", "round", "isinstance", "getattr", "hasattr", "print")
               if k in _bi}
    _mods = {"re", "math", "collections", "itertools", "statistics", "json"}

    def _imp(name, *a, **k):
        if name.split(".")[0] not in _mods:
            raise ImportError(f"import '{name}' not allowed in a primitive")
        return importlib.import_module(name)
    allowed["__import__"] = _imp

    def _fuse_ids(lists, k=60):                    # RRF over id lists (so authored code is self-contained)
        s: dict = {}
        for lst in lists:
            for r, i in enumerate(list(lst)):
                s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
        return sorted(s, key=lambda i: -s[i])

    def _rerank(session, query, ids, top_k=10):    # cross-encoder rerank an id list
        docs = session.store.get(list(ids)[:60])
        texts = [d.text or "" for d in docs]
        if not texts:
            return list(ids)[:top_k]
        scores = session.reranker(query, texts) if getattr(session, "reranker", None) else list(range(len(texts), 0, -1))
        order = sorted(range(len(docs)), key=lambda j: -scores[j])
        return [str(docs[j].id) for j in order[:top_k]]

    return {"__builtins__": allowed, "sac": sac, "P": P, "fuse": P.fuse, "re": _re,
            "fuse_ids": _fuse_ids, "rerank": _rerank}


@dataclass
class CodePrimitive:
    """A genuinely NEW retrieval primitive the LLM authored as code (not a composition of existing
    retrievers). Persisted as source; executed in a restricted sandbox as a registered skill."""
    name: str
    when_to_use: str
    code: str                           # defines `def run(session, query, top_k): -> ids | ResultSet`
    origin: str = "llm_code"
    tags: list = field(default_factory=lambda: ["learned", "code"])

    def to_skill(self, _registry=None) -> "Skill":
        ns: dict = {}
        exec(compile(self.code, f"<primitive:{self.name}>", "exec"), _safe_globals(), ns)  # noqa: S102
        fn = ns.get("run")
        if not callable(fn):
            raise ValueError(f"authored primitive '{self.name}' has no run(session, query, top_k)")

        def run(session, query, top_k=10, **_):
            try:
                out = fn(session, query, top_k)
            except Exception:
                return []
            if out is None:
                return []
            return out.ids()[:top_k] if hasattr(out, "ids") else [str(x) for x in out][:top_k]

        return Skill(self.name, self.when_to_use, run, tags=self.tags, cost=2,
                     description="LLM-authored code primitive")


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
        self.code_primitives: dict[str, CodePrimitive] = {}
        self.learnings: list[str] = []
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        p = self.path
        if p is None:
            return
        if (p / "skills.jsonl").exists():
            for ln in (p / "skills.jsonl").read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    self.skills[d["name"]] = LearnedSkill(**d)
        if (p / "subagents.jsonl").exists():
            for ln in (p / "subagents.jsonl").read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    self.subagents[d["name"]] = LearnedSubagent(**d)
        if (p / "code_primitives.jsonl").exists():
            for ln in (p / "code_primitives.jsonl").read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    self.code_primitives[d["name"]] = CodePrimitive(**d)
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
            for sub in self.subagents.values():
                f.write(json.dumps(asdict(sub)) + "\n")
        with (self.path / "code_primitives.jsonl").open("w") as f:
            for c in self.code_primitives.values():
                f.write(json.dumps(asdict(c)) + "\n")
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
        for cp in store.code_primitives.values():   # register previously-authored code primitives
            try:
                self.registry.register(cp.to_skill())
            except Exception:
                pass

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

    def create_code_primitive(self, name: str, when_to_use: str, code: str) -> str:
        """TRUE primitive creation: register + persist an LLM-AUTHORED retrieval function (arbitrary
        code, not a composition). Compiles + smoke-instantiates before accepting."""
        cp = CodePrimitive(name=name, when_to_use=when_to_use, code=code)
        self.registry.register(cp.to_skill())        # raises if code doesn't compile / lacks run()
        self.store.code_primitives[name] = cp
        self.store.save()
        return name

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


def author_code_primitive(gen, patterns: str, forge, session, test_query: str, gold,
                          name: str = "llm_authored", tries: int = 3, log=print) -> tuple:
    """TRUE primitive creation with validate-and-retry: the LLM AUTHORS a retrieval function, we run
    it on a held query, and if it errors / returns nothing we feed the failure back and re-author —
    accepting only a primitive that actually retrieves. Returns (code, accepted)."""
    goldset = set(map(str, gold))
    err = ""
    code = ""
    base = ("Write a reusable Python retrieval primitive. Signature EXACTLY: def run(session, query, top_k):\n"
            "Available: session.search(q, top_k=k, mode='hybrid'|'dense'|'keyword'); "
            "session.hyde_search(q, top_k=k) (USE for generically-DESCRIBED entities); "
            "session.store.query_fielded(q, ['title','text'], top_k=k) (named entities); "
            "fuse([resultsets]) -> a fused ResultSet (RRF). Decompose the query (split on ' and ' / commas), "
            "pick the right mode per sub-part, ALWAYS include a hyde pass for coverage, fuse everything, and "
            "return the fused ids (or the ResultSet). Return ONLY a ```python block```.\n\nWINNING PATTERNS:\n" + patterns)
    for i in range(tries):
        prompt = base + (f"\n\nYOUR PREVIOUS CODE FAILED — {err}\nReturn corrected code." if err else "")
        raw = gen.complete(prompt) if hasattr(gen, "complete") else (gen(prompt)[0] if isinstance(gen(prompt), list) else str(gen(prompt)))
        m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        code = (m.group(1) if m else raw).strip()
        try:
            ns: dict = {}
            exec(compile(code, f"<primitive:{name}>", "exec"), _safe_globals(), ns)  # noqa: S102
            fn = ns.get("run")
            if not callable(fn):
                raise ValueError("no run(session, query, top_k) defined")
            out = fn(session, test_query, 10)                # real exceptions surface here (not swallowed)
            ids = out.ids()[:10] if hasattr(out, "ids") else [str(x) for x in (out or [])][:10]
            found = len(goldset & set(map(str, ids)))
            log(f"  attempt {i+1}: {len(ids)} ids, {found}/{len(goldset)} golds")
            if ids and found > 0:
                forge.create_code_primitive(name, "LLM-authored, validated multi-hop primitive", code)
                return code, True
            err = (f"it returned {len(ids)} ids and found {found}/{len(goldset)} golds. Actually CALL the "
                   "retrieval functions (session.search / session.hyde_search / session.store.query_fielded), "
                   "include a hyde pass, fuse the ResultSets, and return the fused ids.")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            log(f"  attempt {i+1}: error — {err[:110]}")
    return code, False


def reflect(ctx, result, forge: HarnessForge, threshold: float = 0.5) -> list:
    """Online-learning step: after a solve, create/modify skills/subagents/rules/memory from evidence.

    Rule-based + deterministic (LLM proposals can extend it). Returns the names of forged artifacts."""
    created: list = []
    intent = getattr(ctx, "intent", None)
    kind = intent.kind if intent is not None else "unknown"
    # Forge only from a VERIFIED win. The gate used to be `score < threshold` alone, and with
    # the old default_verify (1.0 for any non-empty list) it was never true — so every
    # non-empty multi-hop run forged a skill and a subagent from no evidence at all (SDK-A3).
    if not result.ids or result.score < threshold or not getattr(result, "verified", False):
        return created

    # 1) always remember the win (durable memory)
    forge.remember(f"query like \"{ctx.query[:100]}\" ({kind}) -> '{result.skill}' worked",
                   kind="skill_win", skill=result.skill, intent=kind)

    # 2) a multi-hop plan that worked → forge a named composed skill for this intent (if novel)
    if result.skill == "subagents":
        name = f"learned_multihop_{kind}"
        if name not in forge.store.skills:
            forge.create_skill(name, f"multi-hop {kind} queries needing several docs",
                               retrievers=["decompose_arsenal"], combine="fuse", cost=2)
            forge.create_subagent(f"sub_{kind}", f"a sub-question of a {kind} query",
                                  plan=["arsenal_single"])
            created += [name, f"sub_{kind}"]

    # 3) error-code / definition wins → refine the supplemental prompt with a rule
    if kind == "error_code":
        forge.refine_prompt("For error-code / ID queries, try exact_lookup before semantic search.")
    elif kind == "multi_hop":
        forge.refine_prompt("For multi-hop: decompose into sub-facts and use the full arsenal per sub "
                            "— hybrid + HyDE (for generically-described entities) + fielded — then FUSE "
                            "(don't rerank the union).")
    return created
