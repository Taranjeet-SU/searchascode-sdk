"""Parallel ceiling sweep for a chosen embedder — reports BOTH recall@10 and
nDCG@10 (the BEIR/SOTA metric), plus recall@100, oracle, and the distribution.

    python -m phase2.ceiling_model --model e5-large --frac 0.5 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np

from phase1 import common, metrics
from phase2 import embed_models

INDIVIDUAL = ["dense", "keyword", "hybrid_rrf", "hybrid_.8", "prf"]
COMBINATION = ["dense+rerank", "hybrid_.8+rerank", "fuse(dense,kw,prf)", "fuse(dense,kw,prf)+rerank"]
COMBOS = INDIVIDUAL + COMBINATION

_KEY = None  # set per run; workers read the module global


def run_shard(args):
    qids, queries, qrels, key = args
    import torch
    import search_as_code as sac

    q_embed, _p, dim, index = embed_models.build(key, "cuda" if torch.cuda.is_available() else "cpu")
    rr = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    s = sac.Session("opensearch", index=index, dim=dim, hosts=[common.OS_HOST], embedder=q_embed, reranker=rr)
    rr("warm", ["a", "b"])

    def rec(ids, gold, k=10):
        g = {d for d in gold}
        return len(set(ids[:k]) & g) / len(g) if g else 0.0

    out = {}
    for qid in qids:
        q = queries[qid]
        goldd = {d: sc for d, sc in qrels[qid].items() if sc > 0}
        gold = set(goldd)
        qv = np.asarray(q_embed([q])[0], dtype=np.float32)
        dense = s.store.query_vector(qv.tolist(), top_k=100)
        kw = s.store.query_keyword(q, top_k=100)
        prf = s.prf_search(q, top_k=50)
        d50, k50 = dense.top(50), kw.top(50)
        rrf = sac.fuse([d50, k50]); w8 = sac.fuse([d50, k50], weights=[0.8, 0.2]); dkp = sac.fuse([d50, k50, prf])
        R = {
            "dense": dense.top(10).ids(), "keyword": kw.top(10).ids(),
            "hybrid_rrf": rrf.top(10).ids(), "hybrid_.8": w8.top(10).ids(), "prf": prf.top(10).ids(),
            "dense+rerank": sac.rerank(q, d50, reranker=rr, top_k=10).ids(),
            "hybrid_.8+rerank": sac.rerank(q, w8.top(50), reranker=rr, top_k=10).ids(),
            "fuse(dense,kw,prf)": dkp.top(10).ids(),
            "fuse(dense,kw,prf)+rerank": sac.rerank(q, dkp.top(50), reranker=rr, top_k=10).ids(),
        }
        rec10 = {c: rec(R[c], gold) for c in COMBOS}
        ndcg10 = {c: metrics.ndcg_at_k(R[c], goldd, 10) for c in COMBOS}
        out[qid] = {"rec10": rec10, "ndcg10": ndcg10,
                    "dense@100": rec(dense.ids(), gold, 100), "hybrid@100": rec(rrf.ids(), gold, 100)}
    return out


def main(key: str, frac=0.5, workers=4, seed=42):
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    pool = [q for q in qrels if any(s > 0 for s in qrels[q].values())]
    random.seed(seed)
    qids = random.sample(pool, max(1, int(len(pool) * frac)))
    shards = [qids[i::workers] for i in range(workers)]
    print(f"[{key}] {len(qids)} queries, {workers} procs, index={embed_models.index_name(key)}", flush=True)

    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as p:
        parts = p.map(run_shard, [(sh, queries, qrels, key) for sh in shards])
    merged = {}
    for pt in parts:
        merged.update(pt)
    print(f"[{key}] done in {time.time()-t0:.0f}s", flush=True)

    rec = {c: float(np.mean([merged[q]["rec10"][c] for q in merged])) for c in COMBOS}
    ndcg = {c: float(np.mean([merged[q]["ndcg10"][c] for q in merged])) for c in COMBOS}
    r100 = {k: float(np.mean([merged[q][k] for q in merged])) for k in ("dense@100", "hybrid@100")}
    oracle_rec = float(np.mean([max(merged[q]["rec10"][c] for c in COMBOS) for q in merged]))
    oracle_ndcg = float(np.mean([max(merged[q]["ndcg10"][c] for c in COMBOS) for q in merged]))
    best = max(ndcg, key=ndcg.get)

    print(f"\n===== {key} — recall@10 / nDCG@10 (n={len(qids)}) =====")
    print(f"  {'combo':28s} {'recall@10':>10} {'nDCG@10':>10}")
    for c in COMBOS:
        print(f"  {c:28s} {rec[c]:>10.4f} {ndcg[c]:>10.4f}")
    print(f"\n  best by nDCG@10: {best}  (nDCG@10={ndcg[best]:.4f}, recall@10={rec[best]:.4f})")
    print(f"  ORACLE  recall@10={oracle_rec:.4f}  nDCG@10={oracle_ndcg:.4f}")
    print(f"  recall@100  dense={r100['dense@100']:.4f}  hybrid={r100['hybrid@100']:.4f}")

    res = {"model": key, "n": len(qids),
           "recall@10": {c: round(rec[c], 4) for c in COMBOS},
           "ndcg@10": {c: round(ndcg[c], 4) for c in COMBOS},
           "recall@100": {k: round(v, 4) for k, v in r100.items()},
           "oracle": {"recall@10": round(oracle_rec, 4), "ndcg@10": round(oracle_ndcg, 4)},
           "best_by_ndcg": best}
    (common.RUNS_DIR / f"ceiling_{key}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[{key}] wrote", common.RUNS_DIR / f"ceiling_{key}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().model, frac=ap.parse_args().frac, workers=ap.parse_args().workers)
