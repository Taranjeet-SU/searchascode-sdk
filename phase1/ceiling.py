"""Ceiling sweep: which primitive / combination maximizes recall@10 on FiQA, and
what is the retrieval ceiling?

Model-free combos (no LLM) over 500 random queries. Also reports:
- recall@100 (dense/hybrid) = retrievability ceiling (are gold docs even found?)
- per-query ORACLE = best achievable if you picked the perfect combo per query
  (the headroom an adaptive agent/router could capture).
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

import search_as_code as sac
from phase1 import common, metrics


def main(n: int = 500, seed: int = 42):
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    pool = [q for q in qrels if any(s > 0 for s in qrels[q].values())]
    random.seed(seed)
    qids = pool if len(pool) <= n else random.sample(pool, n)
    print(f"[ceiling] {len(qids)} queries")

    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()
    rr = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    s = sac.Session("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
                    embedder=embed, reranker=rr)
    rr("warm", ["a", "b"])

    def R(ids, gold, k=10):
        return len(set(ids[:k]) & gold) / len(gold) if gold else 0.0

    combos = ["dense", "keyword", "hybrid_rrf", "hybrid_w.8/.2", "dense+rerank",
              "hybrid+rerank", "dense+mmr", "prf", "prf+rerank", "fuse(dense,prf)"]
    acc = {c: [] for c in combos}
    r100 = {"dense@100": [], "hybrid@100": []}
    per_q_best = []
    t0 = time.time()

    for i, qid in enumerate(qids):
        q = queries[qid]
        gold = {d for d, sc in qrels[qid].items() if sc > 0}
        qv = np.asarray(embed([q])[0], dtype=np.float32)
        dense = s.store.query_vector(qv, top_k=100)
        kw = s.store.query_keyword(q, top_k=100)
        prf = s.prf_search(q, top_k=50)                      # Rocchio (shifts neighborhood)
        fused = sac.fuse([dense.top(50), kw.top(50)])
        fused_w = sac.fuse([dense.top(50), kw.top(50)], weights=[0.8, 0.2])

        res = {
            "dense": dense.top(10).ids(),
            "keyword": kw.top(10).ids(),
            "hybrid_rrf": fused.top(10).ids(),
            "hybrid_w.8/.2": fused_w.top(10).ids(),
            "dense+rerank": sac.rerank(q, dense.top(50), reranker=rr, top_k=10).ids(),
            "hybrid+rerank": sac.rerank(q, fused.top(50), reranker=rr, top_k=10).ids(),
            "dense+mmr": sac.mmr(qv.tolist(), dense.top(50), lambda_=0.5, top_k=10).ids(),
            "prf": prf.top(10).ids(),
            "prf+rerank": sac.rerank(q, prf.top(50), reranker=rr, top_k=10).ids(),
            "fuse(dense,prf)": sac.fuse([dense.top(50), prf.top(50)]).top(10).ids(),
        }
        best = 0.0
        for c in combos:
            r = R(res[c], gold)
            acc[c].append(r)
            best = max(best, r)
        per_q_best.append(best)
        r100["dense@100"].append(R(dense.ids(), gold, 100))
        r100["hybrid@100"].append(R(fused.ids(), gold, 100))
        if (i + 1) % 100 == 0:
            print(f"[ceiling] {i+1}/{len(qids)}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n===== FiQA ceiling sweep — recall@10 by primitive combo =====")
    for c in sorted(combos, key=lambda c: -np.mean(acc[c])):
        print(f"  {c:16s} {np.mean(acc[c]):.4f}")
    print("\n  --- retrievability ceiling (recall@100) ---")
    for k, v in r100.items():
        print(f"  {k:16s} {np.mean(v):.4f}")
    print(f"\n  ORACLE per-query best@10 (perfect routing): {np.mean(per_q_best):.4f}")
    out = {"n": len(qids),
           "recall@10": {c: round(float(np.mean(acc[c])), 4) for c in combos},
           "recall@100": {k: round(float(np.mean(v)), 4) for k, v in r100.items()},
           "oracle@10": round(float(np.mean(per_q_best)), 4)}
    (common.RUNS_DIR / "ceiling.json").write_text(json.dumps(out, indent=2))
    print("\n[ceiling] wrote", common.RUNS_DIR / "ceiling.json")


if __name__ == "__main__":
    main()
