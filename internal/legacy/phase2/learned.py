"""Runtime: pull a dataset's LEARNED PROFILE from the DB and fuse it with the generic
primitives — "standard code + custom learned code executed at runtime".

A profile (mined offline by learn_rules / judge_calibrate / fewshot) holds:
  aliases, glossary (acronyms), synonyms, routes, exemplars, judge_threshold.
This module injects them into normalize_query, expand, and the SAC prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import search_as_code as sac
from phase1 import common

LEARN_INDEX = "sac_learned"


@dataclass
class LearnedProfile:
    dataset: str
    aliases: dict = field(default_factory=dict)
    glossary: dict = field(default_factory=dict)
    synonyms: dict = field(default_factory=dict)
    routes: list = field(default_factory=list)
    exemplars: list = field(default_factory=list)          # [{query, arm, recipe}]
    judge_threshold: dict = field(default_factory=dict)    # {min_top, min_gap}

    # ---- load from DB (falls back to file) ----
    @classmethod
    def load(cls, dataset: str, hosts=None) -> "LearnedProfile":
        src = None
        try:
            store = sac.connect("opensearch", index="_meta", hosts=hosts or [common.OS_HOST])
            src = store.client.get(index=LEARN_INDEX, id=dataset)["_source"]
        except Exception:
            f = common.REPO / "phase2" / "runs" / f"learned_{dataset}.json"
            if f.exists():
                src = json.loads(f.read_text())
        src = src or {"dataset": dataset}
        return cls(dataset=dataset, aliases=src.get("aliases", {}), glossary=src.get("glossary", {}),
                   synonyms=src.get("synonyms", {}), routes=src.get("routes", []),
                   exemplars=src.get("exemplars", []), judge_threshold=src.get("judge_threshold", {}))

    # ---- inject into primitives ----
    def alias_map(self) -> dict:
        m = dict(sac.__dict__.get("_", {}))
        # domain aliases + acronym glossary both normalize the query surface
        m = {}
        m.update({k: v for k, v in self.aliases.items()})
        m.update({k: v for k, v in self.glossary.items()})
        return m

    def normalize(self, query: str) -> str:
        return sac.normalize_query(query, aliases=self.alias_map())

    def expand_seeds(self, query: str) -> list[str]:
        """Learned synonym/euphemism variants for terms present in the query."""
        ql = query.lower()
        out = [query]
        for term, syns in self.synonyms.items():
            if term in ql:
                for s in syns:
                    out.append(query.replace(term, s) if term in query else f"{query} {s}")
        return list(dict.fromkeys(out))

    def prompt_addendum(self) -> str:
        """A 'Learned for this dataset' block to append to SAC_SYSTEM."""
        lines = ["\n## Learned for this dataset (pulled at runtime)"]
        if self.glossary:
            lines.append("Expand these terms: " + "; ".join(f"{k}→{v}" for k, v in list(self.glossary.items())[:12]))
        if self.aliases:
            lines.append("Normalize spellings: " + "; ".join(f"{k}→{v}" for k, v in list(self.aliases.items())[:12]))
        if self.synonyms:
            top = list(self.synonyms.items())[:8]
            lines.append("Useful synonyms: " + "; ".join(f"{k}→{'/'.join(v[:3])}" for k, v in top))
        if self.routes:
            lines.append("Routing hints: " + "; ".join(f"{r['when']}→{r['use']}" for r in self.routes[:6]))
        if self.exemplars:
            lines.append("Recipes that worked on similar queries:")
            for e in self.exemplars[:5]:
                lines.append(f"  - if query like {e['query'][:50]!r} → {e.get('recipe', e.get('arm'))}")
        return "\n".join(lines) if len(lines) > 1 else ""


if __name__ == "__main__":
    import sys
    p = LearnedProfile.load(sys.argv[1] if len(sys.argv) > 1 else "fiqa")
    print(f"profile[{p.dataset}] aliases={len(p.aliases)} glossary={len(p.glossary)} "
          f"synonyms={len(p.synonyms)} routes={len(p.routes)} exemplars={len(p.exemplars)} "
          f"judge_threshold={p.judge_threshold}")
    print("normalize('deposit a cheque'):", p.normalize("deposit a cheque"))
    print("prompt_addendum:\n", p.prompt_addendum()[:500])
