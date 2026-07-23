"""Finetuning/learning phase for Altera — mines a domain profile from the KB ONLY
(no sheet access -> no test leakage). Produces learned_altera.json:
  - facets / fact_types / entity_types  (domain structure, from aggregations)
  - glossary  (FPGA acronym -> expansion, LLM-mined from sampled KB cards)
  - synonyms  (domain term -> variants)
  - routes    (query-shape -> primitive policy, MCP/altera-kg-aligned)

    ALTERA_OS=... python -m phase4.altera_learn --cards 60
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

from phase1 import common
from phase1.llm import LLM
from phase4 import altera

OUT = Path(common.REPO) / "phase4" / "runs" / "learned_altera.json"

MINE_SYS = ("You are building a glossary for an Altera/Intel FPGA support assistant. From the KB "
            "snippets, extract domain vocabulary that helps search. Output ONLY JSON: "
            '{"glossary":{"ACRONYM":"expansion",...},"synonyms":{"term":["variant1","variant2"],...}}. '
            "Focus on FPGA acronyms (e.g. HPS, EMIF, RSU, PDN, PMA, PCS, ALM, SERDES, AVMM) and "
            "device/tool synonyms. Only include items clearly supported by the snippets.")


def aggs():
    body = {"size": 0, "aggs": {
        "fact_type": {"terms": {"field": "fact_type", "size": 15}},
        "facet": {"terms": {"field": "facet", "size": 20}},
        "entity_type": {"terms": {"field": "entity_type.keyword", "size": 15}}}}
    r = requests.post(f"{altera.OS_URL}/{altera.KG}/_search", json=body, timeout=30).json()
    return {a: [b["key"] for b in v["buckets"] if b["key"]] for a, v in r["aggregations"].items()}


def sample_cards(n, facets):
    """Sample KB cards spread across facets (diverse domain coverage)."""
    out = []
    per = max(3, n // max(1, len(facets)))
    for f in facets:
        body = {"size": per, "query": {"term": {"facet": f}},
                "_source": ["answer", "evidence", "content", "doc_title"]}
        try:
            hits = requests.post(f"{altera.OS_URL}/{altera.KG}/_search", json=body, timeout=30).json()["hits"]["hits"]
        except Exception:
            continue
        for h in hits:
            s = h["_source"]
            t = s.get("answer") or s.get("evidence") or s.get("content") or ""
            if t:
                out.append(t[:300])
        if len(out) >= n:
            break
    return out[:n]


def mine_glossary(gen, cards):
    glossary, synonyms = {}, {}
    for i in range(0, len(cards), 6):
        batch = "\n---\n".join(cards[i:i + 6])
        raw = gen.complete(f"KB snippets:\n{batch}", system=MINE_SYS)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            continue
        try:
            j = json.loads(m.group(0))
        except Exception:
            continue
        for k, v in (j.get("glossary") or {}).items():
            if k and v and len(k) <= 20:
                glossary[k.strip().upper()] = str(v).strip()
        for k, v in (j.get("synonyms") or {}).items():
            if k and v:
                synonyms.setdefault(k.strip().lower(), [])
                synonyms[k.strip().lower()] += [str(x) for x in (v if isinstance(v, list) else [v])]
    return glossary, synonyms


# MCP / altera-kg-aligned routing policy (derived from the KB's structure, not the sheet)
ROUTES = [
    {"when": "exact part number / ordering code / register / signal name",
     "use": "keyword() for the exact token + kb(fact_type:spec) to verify"},
    {"when": "definition / 'what is' / component or capability overview",
     "use": "kb() curated cards first (authoritative), then dense() for context"},
    {"when": "multi-part or comparison question",
     "use": "subqueries() then fan-out dense()+kb() per part, fuse()"},
    {"when": "any answer with a numeric spec/limit",
     "use": "VERIFY the value against kb() altera-kg cards before answering (MCP-style)"},
]


def main(cards=60):
    gen = LLM()
    structure = aggs()
    print(f"[learn] facets={len(structure['facet'])} fact_types={len(structure['fact_type'])}")
    cs = sample_cards(cards, structure["facet"])
    print(f"[learn] sampled {len(cs)} KB cards across facets; mining glossary...")
    glossary, synonyms = mine_glossary(gen, cs)
    profile = {"source": "altera_kg_v2 (KB only, no sheet)", "facets": structure["facet"],
               "fact_types": structure["fact_type"], "entity_types": structure["entity_type"],
               "glossary": glossary, "synonyms": synonyms, "routes": ROUTES}
    OUT.write_text(json.dumps(profile, indent=2))
    print(f"[learn] glossary={len(glossary)} synonyms={len(synonyms)}  (llm ${gen.usage.cost_usd:.4f})")
    print("[learn] sample glossary:", dict(list(glossary.items())[:10]))
    print(f"[learn] saved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cards", type=int, default=60)
    main(ap.parse_args().cards)
