"""Synthetic training-set generator for the learned router (KB ONLY, no sheet).

Samples altera_kg_v2 cards across facets and asks the LLM to write a natural question
that each card answers. gold = the source card (its docid). This labeled set (no sheet
leakage) is used to (a) explore which primitive/arm retrieves the gold, and (b) train
the XGB router. LLM + tunnel only -> no GPU (runs parallel to the GPU evals).

    ALTERA_OS=... python -m phase4.altera_synth --n 300 --workers 12
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from phase1 import common
from phase1.llm import LLM
from phase4 import altera

OUT = Path(common.REPO) / "phase4" / "runs" / "synth_altera.json"
Q_SYS = ("Write ONE natural support question that a user would ask, whose answer is exactly the "
         "given FPGA knowledge snippet. Be specific (mention the device/feature). Output only the "
         "question, no preamble.")


def sample_cards(n, facets, seed=0):
    random.seed(seed)
    out = []
    per = max(4, (n * 2) // max(1, len(facets)))
    for f in facets:
        body = {"size": per, "query": {"function_score": {
                    "query": {"term": {"facet": f}}, "random_score": {"seed": seed, "field": "_seq_no"}}},
                "_source": ["answer", "evidence", "content", "doc_title", "docid", "family"]}
        try:
            hits = requests.post(f"{altera.OS_URL}/{altera.KG}/_search", json=body, timeout=30).json()["hits"]["hits"]
        except Exception:
            continue
        for h in hits:
            s = h["_source"]
            txt = (s.get("answer") or s.get("evidence") or s.get("content") or "").strip()
            if len(txt) > 60 and s.get("docid"):
                out.append({"card_id": h["_id"], "docid": str(s["docid"]), "facet": f,
                            "family": s.get("family", ""), "text": txt[:400],
                            "title": s.get("doc_title", "")})
    random.shuffle(out)
    return out[:n]


def gen_q(gen, card):
    q = gen.complete(f"Knowledge snippet ({card.get('family','')}, {card['facet']}):\n{card['text']}",
                     system=Q_SYS).strip().strip('"')
    return {"question": q, "gold_docid": card["docid"], "gold_card_id": card["card_id"],
            "facet": card["facet"], "family": card.get("family", "")}


def main(n=300, workers=12):
    gen = LLM()
    facets = [b["key"] for b in requests.post(f"{altera.OS_URL}/{altera.KG}/_search",
              json={"size": 0, "aggs": {"f": {"terms": {"field": "facet", "size": 20}}}},
              timeout=30).json()["aggregations"]["f"]["buckets"] if b["key"]]
    cards = sample_cards(n, facets)
    print(f"[synth] sampled {len(cards)} cards across {len(facets)} facets; generating questions...", flush=True)
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(gen_q, gen, c) for c in cards]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception:
                pass
            done += 1
            if done % 25 == 0:
                print(f"[synth] {done}/{len(cards)}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"[synth] wrote {len(rows)} synthetic (question, gold) pairs -> {OUT}  (${gen.usage.cost_usd:.4f})")
    print("[synth] samples:")
    for r in rows[:4]:
        print(f"   [{r['facet']}] {r['question'][:90]}  -> docid {r['gold_docid']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); main(a.n, a.workers)
