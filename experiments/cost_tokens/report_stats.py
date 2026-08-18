"""Post-run stats: what the harness actually DID and LEARNED across the cost runs + pipelines.

Reports, per corpus:
  1. hop-depth distribution of the product arms (escalation rate, hops when escalated)
  2. search usage: searches/query per arm; raw-OpenSearch DSL + keyword-escalation usage
     mined from the code the memory harness stored with each skill-win
  3. forge inventory: code primitives / skills / subagents (+ provenance, superseded count)
  4. memory learned: skill-wins and findings accumulated by the runs

    python -m experiments.cost_tokens.report_stats
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DJ = HERE.parent / "deep_judge"


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.open() if ln.strip()]


def hop_stats(corpus: str):
    rows = list({r["qid"]: r for r in _rows(HERE / f"cost_{corpus}_perquery.jsonl")}.values())
    rows = [r for r in rows if "sac_product" in r]
    if not rows:
        print(f"  (no product rows yet for {corpus})")
        return
    for arm in ("sac_product", "tool_product"):
        esc = [r[arm] for r in rows if r[arm].get("escalated")]
        hops = Counter(r[arm]["turns"] for r in rows)
        print(f"  {arm}: n={len(rows)} escalation_rate={len(esc)/len(rows):.2f} "
              f"avg_searches={sum(r[arm]['searches'] for r in rows)/len(rows):.1f}")
        print(f"    hop-depth (turns incl. hop-0) distribution: "
              + " ".join(f"{k}:{v}" for k, v in sorted(hops.items())))
        if esc:
            print(f"    when escalated: avg turns={sum(e['turns'] for e in esc)/len(esc):.1f} "
                  f"avg in-tokens={sum(e['in'] for e in esc)/len(esc):.0f} "
                  f"vs PASS-at-hop-0 in-tokens="
                  f"{sum(r[arm]['in'] for r in rows if not r[arm].get('escalated'))/max(1,len(rows)-len(esc)):.0f}")


def code_usage(paths, label):
    codes = []
    for p in paths:
        for m in _rows(p):
            c = (m.get("meta") or {}).get("code") or m.get("code")
            if c:
                codes.append(c)
    if not codes:
        print(f"  {label}: no stored winning code yet")
        return
    n = len(codes)
    raw_dsl = sum(1 for c in codes if "_search(" in c)
    kw = sum(1 for c in codes if "mode='keyword'" in c or 'mode="keyword"' in c)
    dense = sum(1 for c in codes if "mode='dense'" in c or 'mode="dense"' in c)
    hyde = sum(1 for c in codes if "hyde" in c)
    fused = sum(1 for c in codes if "fuse" in c)
    phrases = Counter(t for c in codes for t in re.findall(r"match_phrase|match\b|term\b", c))
    print(f"  {label}: {n} winning strategies stored")
    print(f"    raw OpenSearch DSL: {raw_dsl}/{n} · keyword-escalation: {kw}/{n} · "
          f"dense: {dense}/{n} · hyde: {hyde}/{n} · fusion: {fused}/{n}")
    if phrases:
        print(f"    DSL clause mix: {dict(phrases)}")


def forge_inventory():
    for store in sorted(DJ.glob("forge_store_*")):
        cps = _rows(store / "code_primitives.jsonl")
        sks = _rows(store / "skills.jsonl")
        subs = _rows(store / "subagents.jsonl")
        sup = _rows(store / "superseded.jsonl")
        learn = (store / "learnings.md").read_text().count("\n- ") if (store / "learnings.md").exists() else 0
        print(f"  {store.name}: primitives={len(cps)} skills={len(sks)} subagents={len(subs)} "
              f"rules={learn} superseded_archive={len(sup)}")
        for c in cps:
            prov = c.get("provenance") or {}
            gate = prov.get("gate_baseline"), prov.get("accepted")
            print(f"    - {c['name']}: {c['when_to_use'][:60]}"
                  + (f" [gate: vs {gate[0]}, accepted={gate[1]}]" if prov else " [no provenance (pre-gate)]"))


def memory_stats():
    for p in sorted(list(DJ.glob("explore_*_memory.jsonl")) + list(HERE.glob("product_memory_*.jsonl"))):
        rows = _rows(p)
        kinds = Counter(r.get("kind", "?") for r in rows)
        print(f"  {p.name}: {len(rows)} memories — {dict(kinds)}")


if __name__ == "__main__":
    print("== 1. hop depth / escalation (product arms) ==")
    for c in ("browsecomp_qwen8b", "hotpotqa_qwen8b"):
        print(f" {c}:")
        hop_stats(c)
    print("\n== 2. what the winning code actually called ==")
    code_usage(list(HERE.glob("product_memory_*.jsonl")), "cost-run product arms")
    code_usage(list(DJ.glob("explore_*_memory.jsonl")), "explore pipelines")
    print("\n== 3. forge inventory (primitives / skills / subagents / rules) ==")
    forge_inventory()
    print("\n== 4. memory learned ==")
    memory_stats()
