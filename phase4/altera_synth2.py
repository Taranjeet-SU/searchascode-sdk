"""Expanded stratified synthetic generator (KB only, no sheet) for the learned router.

Design goal: REAL routing structure. Sources are drawn from BOTH indices so primitives
genuinely compete:
  - ft_document chunks  -> gold = chunk _id  (dense/keyword can hit; kb usually can't)
  - altera_kg cards     -> gold = card docid (kb hits; dense/keyword usually can't)
Difficulty-stratified:
  - easy   : exact-term lookup (favors keyword/kb)
  - medium : paraphrased/conceptual (favors dense/expand)
  - hard   : multi-evidence across 2 related sources (favors fan-out/agglom; multi-gold)

Bulk-samples sources in a few big queries (tunnel-light), then LLM-generates in parallel.

    ALTERA_OS=... python -m phase4.altera_synth2 --n 5000 --workers 16
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

OUT = Path(common.REPO) / "phase4" / "runs" / "synth2_altera.json"
FTP = altera.FTP

SYS = {
 "easy": ("Write ONE EASY lookup question a user would ask, answerable by a single exact fact in the "
          "snippet. Use the exact device/feature/part names from the snippet. Output only the question."),
 "medium": ("Write ONE MEDIUM question that paraphrases the concept in the snippet WITHOUT reusing its "
            "exact keywords (use synonyms/rewording), still answerable from it. Output only the question."),
 "hard": ("Write ONE HARD question that requires COMBINING the two snippets (multi-part / multi-evidence). "
          "It should not be answerable from either snippet alone. Output only the question."),
}


def sample_docs(n):
    body = {"size": n, "query": {"function_score": {
                "query": {"exists": {"field": FTP + "content"}}, "random_score": {}}},
            "_source": [FTP + "content", FTP + "ft_title"]}
    hits = requests.post(f"{altera.OS_URL}/{altera.FT_DOC}/_search", json=body, timeout=120).json()["hits"]["hits"]
    out = []
    for h in hits:
        c = (h["_source"].get(FTP + "content") or "").strip()
        if len(c) > 80:
            out.append({"src": "doc", "gold": h["_id"], "text": c[:400],
                        "title": h["_source"].get(FTP + "ft_title", ""), "facet": ""})
    return out


def sample_cards(n, facets, seed=0):
    out = []
    per = max(5, (n * 2) // max(1, len(facets)))
    for f in facets:
        body = {"size": per, "query": {"function_score": {
                    "query": {"term": {"facet": f}}, "random_score": {"seed": seed}}},
                "_source": ["answer", "evidence", "content", "docid", "family"]}
        try:
            hits = requests.post(f"{altera.OS_URL}/{altera.KG}/_search", json=body, timeout=60).json()["hits"]["hits"]
        except Exception:
            continue
        for h in hits:
            s = h["_source"]; txt = (s.get("answer") or s.get("evidence") or s.get("content") or "").strip()
            if len(txt) > 80 and s.get("docid"):
                out.append({"src": "kg", "gold": str(s["docid"]), "text": txt[:400],
                            "family": s.get("family", ""), "facet": f})
    return out


def gen_one(gen, spec):
    diff, srcs = spec["difficulty"], spec["sources"]
    if diff == "hard":
        snip = f"Snippet A:\n{srcs[0]['text']}\n\nSnippet B:\n{srcs[1]['text']}"
        gold = [s["gold"] for s in srcs]
    else:
        snip = srcs[0]["text"]; gold = [srcs[0]["gold"]]
    q = gen.complete(snip, system=SYS[diff]).strip().strip('"')
    return {"question": q, "difficulty": diff, "src": srcs[0]["src"], "gold": gold,
            "facet": srcs[0].get("facet", ""), "family": srcs[0].get("family", "")}


def main(n=5000, workers=16, seed=0):
    random.seed(seed)
    gen = LLM()
    facets = [b["key"] for b in requests.post(f"{altera.OS_URL}/{altera.KG}/_search",
              json={"size": 0, "aggs": {"f": {"terms": {"field": "facet", "size": 20}}}},
              timeout=30).json()["aggregations"]["f"]["buckets"] if b["key"]]
    # bulk sample sources (few big queries -> tunnel-light)
    need = n
    docs = sample_docs(min(2000, need)); cards = sample_cards(min(2000, need), facets, seed)
    pool = docs + cards
    random.shuffle(pool)
    print(f"[synth2] sampled sources: {len(docs)} doc-chunks + {len(cards)} kg-cards = {len(pool)}", flush=True)

    # build specs: 40% easy, 35% medium, 25% hard (hard needs 2 same-src sources)
    specs = []
    i = 0
    while len(specs) < n and i < len(pool):
        r = random.random()
        if r < 0.40:
            specs.append({"difficulty": "easy", "sources": [pool[i]]}); i += 1
        elif r < 0.75:
            specs.append({"difficulty": "medium", "sources": [pool[i]]}); i += 1
        else:
            same = [pool[j] for j in range(i, min(i + 8, len(pool))) if pool[j]["src"] == pool[i]["src"]]
            if len(same) >= 2:
                specs.append({"difficulty": "hard", "sources": same[:2]})
            i += 1
    print(f"[synth2] {len(specs)} specs; generating with {workers} workers...", flush=True)

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(gen_one, gen, s) for s in specs]
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception:
                pass
            done += 1
            if done % 250 == 0:
                print(f"[synth2] {done}/{len(specs)}  (${gen.usage.cost_usd:.2f})", flush=True)
    OUT.write_text(json.dumps(rows, indent=2))
    from collections import Counter
    print(f"[synth2] wrote {len(rows)} queries -> {OUT}  (${gen.usage.cost_usd:.4f})")
    print("  difficulty:", dict(Counter(r["difficulty"] for r in rows)))
    print("  source:", dict(Counter(r["src"] for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); main(a.n, a.workers)
