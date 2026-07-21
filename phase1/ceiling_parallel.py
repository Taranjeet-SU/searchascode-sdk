"""Parallel ceiling sweep over 50% of FiQA — individual primitives + combinations,
with the per-query best-recall distribution. Spawns multiple worker processes.

    python -m phase1.ceiling_parallel --frac 0.5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np

from phase1 import common

# ---- combos: model-free so the sweep is fast & reproducible -----------------
INDIVIDUAL = ["dense", "keyword", "hybrid_rrf", "hybrid_.8", "prf"]
COMBINATION = ["dense+rerank", "hybrid_.8+rerank", "prf+rerank", "dense+mmr",
               "fuse(dense,prf)", "fuse(dense,kw,prf)", "fuse(dense,kw,prf)+rerank"]
COMBOS = INDIVIDUAL + COMBINATION


def _recall(ids, gold, k=10):
    return len(set(ids[:k]) & gold) / len(gold) if gold else 0.0


def run_shard(args):
    """One worker process: build its own Session (own GPU models), score its qids."""
    qids, queries, qrels = args
    import torch
    from sentence_transformers import SentenceTransformer
    import search_as_code as sac

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()
    rr = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    s = sac.Session("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
                    embedder=embed, reranker=rr)
    rr("warm", ["a", "b"])

    out = {}
    for qid in qids:
        q = queries[qid]
        gold = {d for d, sc in qrels[qid].items() if sc > 0}
        qv = np.asarray(embed([q])[0], dtype=np.float32)
        dense = s.store.query_vector(qv.tolist(), top_k=100)
        kw = s.store.query_keyword(q, top_k=100)
        prf = s.prf_search(q, top_k=50)
        d50, k50 = dense.top(50), kw.top(50)
        rrf = sac.fuse([d50, k50]); w8 = sac.fuse([d50, k50], weights=[0.8, 0.2])
        dkp = sac.fuse([d50, k50, prf])
        res = {
            "dense": dense.top(10).ids(),
            "keyword": kw.top(10).ids(),
            "hybrid_rrf": rrf.top(10).ids(),
            "hybrid_.8": w8.top(10).ids(),
            "prf": prf.top(10).ids(),
            "dense+rerank": sac.rerank(q, d50, reranker=rr, top_k=10).ids(),
            "hybrid_.8+rerank": sac.rerank(q, w8.top(50), reranker=rr, top_k=10).ids(),
            "prf+rerank": sac.rerank(q, prf, reranker=rr, top_k=10).ids(),
            "dense+mmr": sac.mmr(qv.tolist(), d50, lambda_=0.5, top_k=10).ids(),
            "fuse(dense,prf)": sac.fuse([d50, prf]).top(10).ids(),
            "fuse(dense,kw,prf)": dkp.top(10).ids(),
            "fuse(dense,kw,prf)+rerank": sac.rerank(q, dkp.top(50), reranker=rr, top_k=10).ids(),
        }
        rec = {c: _recall(res[c], gold) for c in COMBOS}
        rec["_dense@100"] = _recall(dense.ids(), gold, 100)
        rec["_hybrid@100"] = _recall(rrf.ids(), gold, 100)
        out[qid] = rec
    return out


def _hist(vals, edges=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Distribution of recall values into buckets."""
    buckets = {"=0": 0, "(0,.25]": 0, "(.25,.5]": 0, "(.5,.75]": 0, "(.75,1)": 0, "=1.0": 0}
    for v in vals:
        if v == 0: buckets["=0"] += 1
        elif v <= .25: buckets["(0,.25]"] += 1
        elif v <= .5: buckets["(.25,.5]"] += 1
        elif v <= .75: buckets["(.5,.75]"] += 1
        elif v < 1.0: buckets["(.75,1)"] += 1
        else: buckets["=1.0"] += 1
    return buckets


def main(frac=0.5, workers=4, seed=42):
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    pool = [q for q in qrels if any(s > 0 for s in qrels[q].values())]
    random.seed(seed)
    n = max(1, int(len(pool) * frac))
    qids = random.sample(pool, n)
    shards = [qids[i::workers] for i in range(workers)]
    print(f"[ceiling|| ] {n} queries ({frac:.0%}) across {workers} processes", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        parts = p.map(run_shard, [(sh, queries, qrels) for sh in shards])
    merged = {}
    for part in parts:
        merged.update(part)
    print(f"[ceiling|| ] done in {time.time()-t0:.0f}s", flush=True)

    # aggregate
    means = {c: float(np.mean([merged[q][c] for q in merged])) for c in COMBOS}
    r100 = {k: float(np.mean([merged[q][k] for q in merged])) for k in ("_dense@100", "_hybrid@100")}
    oracle = [max(merged[q][c] for c in COMBOS) for q in merged]
    dense_vals = [merged[q]["dense"] for q in merged]
    best_combo = max(means, key=means.get)

    print("\n===== INDIVIDUAL primitives (recall@10) =====")
    for c in INDIVIDUAL:
        print(f"  {c:26s} {means[c]:.4f}")
    print("\n===== COMBINATIONS (recall@10) =====")
    for c in sorted(COMBINATION, key=lambda c: -means[c]):
        print(f"  {c:26s} {means[c]:.4f}")
    print(f"\n  best single strategy : {best_combo}  ({means[best_combo]:.4f})")
    print(f"  ORACLE (per-query best): {np.mean(oracle):.4f}")
    print(f"  retrievability  dense@100={r100['_dense@100']:.4f}  hybrid@100={r100['_hybrid@100']:.4f}")

    print("\n===== RECALL DISTRIBUTION (per-query) =====")
    print(f"  {'bucket':10s} {'dense':>8} {'ORACLE':>8}")
    hd, ho = _hist(dense_vals), _hist(oracle)
    for b in hd:
        print(f"  {b:10s} {hd[b]:>8} {ho[b]:>8}")

    res = {"n": n, "means": {c: round(means[c], 4) for c in COMBOS},
           "recall@100": {k: round(v, 4) for k, v in r100.items()},
           "oracle@10": round(float(np.mean(oracle)), 4),
           "dist_dense": hd, "dist_oracle": ho}
    (common.RUNS_DIR / "ceiling_parallel.json").write_text(json.dumps(res, indent=2))
    print("\n[ceiling|| ] wrote", common.RUNS_DIR / "ceiling_parallel.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(frac=args.frac, workers=args.workers)
