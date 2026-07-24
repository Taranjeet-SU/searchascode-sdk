"""Combo-ranking exploration (expanded router step 1): for each stratified synth query,
run a set of primitive COMBOS and record which retrieved the gold (all_found for multi-
gold hard queries). Produces labeled data (query emb + difficulty + src + per-combo hit +
best combo) for the XGB router. Base retrievals are computed once and reused across combos
(tunnel-efficient). Parallel but tunnel-safe (default 6 workers).

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_router_explore2 --n 1200 --workers 6
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import decompose, rrf
from phase4.altera_eval_tuned import expand_query

SYNTH = Path(common.REPO) / "phase4" / "runs" / "synth2_altera.json"
OUT = Path(common.REPO) / "phase4" / "runs" / "router_explore2_altera.json"
COMBOS = ["dense", "keyword", "kb", "hybrid", "kb_expanded", "fanout", "fanout_rerank"]


def _ids(docs):
    s = set()
    for d in docs:
        s.add(str(d.get("id"))); s.add(str(d.get("docid"))); s.add(str(d.get("url")))
    return s


def all_found(docs, gold, k=10):
    top = _ids(docs[:k])
    return 1 if all(any(g in t for t in top) for g in gold) else 0


def rephrase(gen, q, n=2):
    r = gen.complete(f"Rephrase this FPGA query {n} ways (synonyms), one per line:\n{q}",
                     system="You rephrase search queries.")
    return [s.strip("-• ").strip() for s in r.splitlines() if s.strip()][:n]


def build_combos(gen, reranker, rr_lock, q):
    # base retrievals (4 tunnel calls), reused across combos
    dense = altera.dense(q, 12); kw = altera.bm25_doc(q, 12); kb = altera.bm25_kg(q, 12)
    kbx = altera.bm25_kg(expand_query(q), 12)
    hybrid = rrf([dense, kw, kb])
    # fanout: decompose but cap at 2 sub-queries x (dense+kb) = 4 tunnel calls (tunnel-light)
    subs = decompose(gen, q)[:2]
    fo_pools = [dense, kb]
    for sq in subs:
        fo_pools += [altera.dense(sq, 10), altera.bm25_kg(sq, 10)]
    fanout = rrf(fo_pools)
    pool = fanout[:30]
    texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in pool]
    if texts:
        with rr_lock:
            sc = reranker(q, texts)
        fanout_rerank = [d for _, d in sorted(zip(sc, pool), key=lambda x: -x[0])]
    else:
        fanout_rerank = pool
    return {"dense": dense, "keyword": kw, "kb": kb, "hybrid": hybrid, "kb_expanded": kbx,
            "fanout": fanout, "fanout_rerank": fanout_rerank}


def process(r, gen, reranker, rr_lock):
    q, gold = r["question"], r["gold"]
    combos = build_combos(gen, reranker, rr_lock, q)
    hits = {c: all_found(combos[c], gold) for c in COMBOS}
    winners = [c for c in COMBOS if hits[c]]
    best = winners[0] if winners else "none"       # COMBOS ordered cheap->rich; first hit = cheapest that works
    return {"question": q, "difficulty": r["difficulty"], "src": r["src"], "facet": r.get("facet", ""),
            "emb": altera.embed(q), "hits": hits, "best": best}


def stratified(rows, n):
    by = {}
    for r in rows:
        by.setdefault((r["difficulty"], r["src"]), []).append(r)
    per = max(1, n // max(1, len(by)))
    out = []
    for k, v in by.items():
        out += v[:per]
    return out[:n]


def main(n=1200, workers=6):
    rows = stratified(json.loads(SYNTH.read_text()), n)
    print(f"[explore2] {len(rows)} queries  dist={dict(Counter((r['difficulty'],r['src']) for r in rows))}", flush=True)
    gen = LLM(); reranker = sac.QwenReranker(); reranker("warm", ["a", "b"]); altera.embedder()
    rr_lock = threading.Lock()
    data, per = [], {c: 0 for c in COMBOS}; solved = 0; done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, r, gen, reranker, rr_lock) for r in rows]
        for f in as_completed(futs):
            try:
                d = f.result()
            except Exception:
                done += 1; continue
            data.append(d); done += 1
            for c in COMBOS:
                per[c] += d["hits"][c]
            solved += (d["best"] != "none")
            if done % 25 == 0:
                print(f"[explore2] {done}/{len(rows)} solved={solved} " +
                      " ".join(f"{c}={per[c]}" for c in COMBOS), flush=True)
    OUT.write_text(json.dumps(data))
    print(f"\n===== combo exploration (n={len(data)}) =====")
    print(f"  solved (any combo) = {solved}/{len(data)} ({solved/max(1,len(data)):.1%})")
    for c in COMBOS:
        print(f"  {c:14s} all_found = {per[c]/max(1,len(data)):.3f}")
    print(f"  best-combo dist: {dict(Counter(d['best'] for d in data))}")
    print(f"  saved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); main(a.n, a.workers)
