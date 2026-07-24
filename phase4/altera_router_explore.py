"""Router exploration (XGB step 1): run several retrieval strategies on the 300 KB-derived
synthetic queries and record which one retrieved the gold card. Produces labeled data
(query embedding + facet + per-arm hit + best arm) to train the learned router.

No Qwen rerank -> GPU-light (own gte model); runs parallel to the GPU evals.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_router_explore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase1 import common
from phase4 import altera
from phase4.altera_eval import rrf
from phase4.altera_eval_tuned import expand_query

SYNTH = Path(common.REPO) / "phase4" / "runs" / "synth_altera.json"
OUT = Path(common.REPO) / "phase4" / "runs" / "router_explore_altera.json"
ARMS = ["dense", "keyword", "kb", "kb_expanded", "hybrid"]


def hit(docs, gold_docid, k=10):
    g = str(gold_docid)
    for d in docs[:k]:
        blob = f"{d.get('url','')} {d.get('id','')} {d.get('docid','')}"
        if g in blob:
            return 1
    return 0


def run_arms(q):
    qe = expand_query(q)
    dense = altera.dense(q, 12)          # uses gte embed (CPU)
    kw = altera.bm25_doc(q, 12)
    kb = altera.bm25_kg(q, 12)
    kbx = altera.bm25_kg(qe, 12)
    hyb = rrf([dense, kw, kb])
    return {"dense": dense, "keyword": kw, "kb": kb, "kb_expanded": kbx, "hybrid": hyb}


def main(n=300):
    rows = json.loads(SYNTH.read_text())[:n]
    data = []
    per_arm = {a: 0 for a in ARMS}
    solved = 0
    for i, r in enumerate(rows):
        q, gold = r["question"], r["gold_docid"]
        res = run_arms(q)
        hits = {a: hit(res[a], gold) for a in ARMS}
        for a in ARMS:
            per_arm[a] += hits[a]
        winners = [a for a in ARMS if hits[a]]
        if winners:
            solved += 1
            # best arm = highest-priority hitter (kb-family cheap; prefer the most specific single arm)
            best = winners[0]
        else:
            best = "none"
        data.append({"question": q, "facet": r.get("facet", ""), "gold_docid": gold,
                     "emb": altera.embed(q), "hits": hits, "best": best})
        if (i + 1) % 25 == 0:
            print(f"[explore] {i+1}/{len(rows)}  solved={solved}  "
                  + " ".join(f"{a}={per_arm[a]}" for a in ARMS), flush=True)

    OUT.write_text(json.dumps(data))
    print(f"\n===== router exploration (n={len(rows)}) =====")
    print(f"  gold retrieved by ANY arm: {solved}/{len(rows)} ({solved/len(rows):.1%})")
    for a in ARMS:
        print(f"  {a:12s} hit@10 = {per_arm[a]/len(rows):.3f}")
    # routing structure? how often arms DISAGREE (some hit, some miss)
    disagree = sum(1 for d in data if 0 < sum(d["hits"].values()) < len(ARMS))
    print(f"  arms disagree on {disagree}/{len(rows)} ({disagree/len(rows):.1%})  <- routing headroom")
    print(f"  saved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300)
    main(ap.parse_args().n)
