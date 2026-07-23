"""Production latency of the SAC answer path against the Altera KB (NO judge — this is
what a deployed query would actually cost). Breaks the wall-clock into stages:
decompose -> fan-out retrieval -> Qwen rerank -> answer generation.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_latency --n 8
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import ANS_SYS, ctx_text, decompose, rrf

SHEET = Path(common.REPO) / "SearchUnify_Evaluation_Package_2026_06_27 (1).xlsx - 5. All Questions.csv"


def sac_answer_timed(gen, reranker, q, k=6):
    t = {}
    t0 = time.time()
    subs = decompose(gen, q); t["decompose"] = time.time() - t0

    t0 = time.time()
    pools = []
    for sq in subs:
        pools.append(altera.dense(sq, 10)); pools.append(altera.bm25_doc(sq, 10))
    pools.append(altera.bm25_kg(q, 10))
    fused = rrf(pools)[:30]; t["retrieval"] = time.time() - t0

    t0 = time.time()
    texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in fused]
    if texts:
        scores = reranker(q, texts)
        fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: -x[0])]
    t["rerank"] = time.time() - t0

    t0 = time.time()
    ctx = ctx_text(fused, k)
    prompt = ("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx else f"Question: {q}\n\nAnswer:"
    gen.complete(prompt, system=ANS_SYS); t["generate"] = time.time() - t0

    t["TOTAL"] = sum(t.values())
    t["n_subqueries"] = len(subs)
    return t


def main(n=8, k=6):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question")][:n]
    gen = LLM()
    reranker = sac.QwenReranker(); reranker("warm", ["a", "b"])
    altera.embedder()  # warm the CPU embedder (excluded from per-query timing)

    stages = ["decompose", "retrieval", "rerank", "generate", "TOTAL"]
    acc = {s: [] for s in stages}
    print("per-query SAC latency (seconds):")
    for i, r in enumerate(rows):
        t = sac_answer_timed(gen, reranker, r["Question"], k)
        for s in stages:
            acc[s].append(t[s])
        print(f"  Q{i+1}: total={t['TOTAL']:.2f}s  (decompose={t['decompose']:.2f} "
              f"retrieval={t['retrieval']:.2f} rerank={t['rerank']:.2f} generate={t['generate']:.2f}) "
              f"subq={t['n_subqueries']}", flush=True)

    print(f"\n===== SAC production latency (n={len(rows)}, KB=altera, gte-alt-v1 CPU + Qwen GPU) =====")
    for s in stages:
        a = acc[s]
        print(f"  {s:10s} mean={np.mean(a):.2f}s  p50={np.median(a):.2f}s  p90={np.percentile(a,90):.2f}s")
    print("  (excludes LLM-as-judge; = real per-query cost if deployed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args(); main(a.n, a.k)
