"""Router Step 1 — exploration → tagged dataset + feature analysis.

For each exploration query, run the primitive palette (dense/keyword/hybrid/prf/
+rerank + the multihop expand-fuse arm), score each arm's recall@10 against qrels
(the LABELS), and extract features (text repr + primitive-probe signals). Saves the
tagged data for modelling (Step 2) and prints the feature analysis.

    python -m phase2.router_explore --n 300 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import re
import time

import numpy as np

from phase1 import common

ARMS = ["dense", "keyword", "hybrid_.8", "prf", "dense+rerank", "hybrid+rerank",
        "expand_fuse", "expand_fuse+rerank"]


def _feats(query: str, dense, kw):
    sc = [h.score for h in dense]
    toks = query.split()
    d_ids, k_ids = set(dense.top(50).ids()), set(kw.top(50).ids())
    return {
        "q_words": len(toks),
        "q_chars": len(query),
        "has_digit": int(any(c.isdigit() for c in query)),
        "has_quote": int('"' in query or "'" in query),
        "caps_tokens": sum(1 for t in toks if t[:1].isupper()),
        "is_question": int(query.strip().endswith("?") or query.lower().split()[0] in
                           {"what", "how", "why", "who", "when", "where", "which", "can", "is", "are", "do"}),
        "dense_top1": float(sc[0]) if sc else 0.0,
        "dense_gap_1_10": float(sc[0] - sc[9]) if len(sc) >= 10 else 0.0,   # flatness
        "dense_gap_1_2": float(sc[0] - sc[1]) if len(sc) >= 2 else 0.0,
        "dense_std_100": float(np.std(sc)) if sc else 0.0,
        "dense_kw_overlap": len(d_ids & k_ids),
        "kw_top1": float(kw[0].score) if len(kw) else 0.0,
        "n_kw": len(kw),
    }


def run_shard(args):
    qids, queries, qrels = args
    import torch
    from sentence_transformers import SentenceTransformer
    import search_as_code as sac
    from phase1.llm import LLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()
    gen = LLM()
    rr = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    s = sac.Session("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
                    embedder=embed, reranker=rr, generator=gen.as_generator())
    rr("warm", ["a", "b"])

    def rec(ids, g):
        return len(set(ids[:10]) & g) / len(g) if g else 0.0

    out = {}
    for qid in qids:
        q = queries[qid]
        g = {d for d, sc in qrels[qid].items() if sc > 0}
        qv = np.asarray(embed([q])[0], dtype=np.float32)
        dense = s.store.query_vector(qv.tolist(), top_k=100)
        kw = s.store.query_keyword(q, top_k=100)
        d50, k50 = dense.top(50), kw.top(50)
        hyb = sac.fuse([d50, k50], weights=[0.8, 0.2])
        prf = s.prf_search(q, top_k=50)
        variants = sac.expand(q, gen.as_generator(), n=3)          # multihop / fan-out arm
        exp = s.search_many(variants, top_k=50, mode="dense")
        arms = {
            "dense": dense.top(10).ids(),
            "keyword": kw.top(10).ids(),
            "hybrid_.8": hyb.top(10).ids(),
            "prf": prf.top(10).ids(),
            "dense+rerank": sac.rerank(q, d50, reranker=rr, top_k=10).ids(),
            "hybrid+rerank": sac.rerank(q, hyb.top(50), reranker=rr, top_k=10).ids(),
            "expand_fuse": exp.top(10).ids(),
            "expand_fuse+rerank": sac.rerank(q, exp.top(50), reranker=rr, top_k=10).ids(),
        }
        out[qid] = {"feats": _feats(q, dense, kw), "qvec": qv.tolist(),
                    "arms": {a: rec(arms[a], g) for a in ARMS}, "n_gold": len(g)}
    return out


def main(n=300, workers=4, seed=7):
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    pool = [q for q in qrels if any(s > 0 for s in qrels[q].values())]
    random.seed(seed)
    qids = random.sample(pool, min(n, len(pool)))
    shards = [qids[i::workers] for i in range(workers)]
    print(f"[explore] {len(qids)} queries, {workers} procs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as p:
        parts = p.map(run_shard, [(sh, queries, qrels) for sh in shards])
    data = {}
    for pt in parts:
        data.update(pt)
    print(f"[explore] done in {time.time()-t0:.0f}s", flush=True)

    (common.REPO / "phase2" / "runs").mkdir(exist_ok=True)
    outp = common.REPO / "phase2" / "runs" / "router_data.json"
    outp.write_text(json.dumps(data))

    # ---- feature analysis ----
    means = {a: np.mean([data[q]["arms"][a] for q in data]) for a in ARMS}
    best_fixed = max(means, key=means.get)
    oracle = np.mean([max(data[q]["arms"].values()) for q in data])
    wins = {a: 0 for a in ARMS}
    for q in data:
        best = max(ARMS, key=lambda a: data[q]["arms"][a])
        wins[best] += 1
    print("\n== per-arm mean recall@10 ==")
    for a in sorted(ARMS, key=lambda a: -means[a]):
        print(f"  {a:20s} {means[a]:.4f}   (best-per-query {wins[a]:3d}/{len(data)})")
    print(f"\n  best fixed arm : {best_fixed} ({means[best_fixed]:.4f})")
    print(f"  ORACLE (router ceiling): {oracle:.4f}   -> routing headroom +{oracle-means[best_fixed]:.4f}")

    # does the 'flat -> expand/rerank helps' rule show up in the tagged data?
    flat = sorted(data, key=lambda q: data[q]["feats"]["dense_gap_1_10"])
    t = len(flat) // 3
    def bk(name, qs):
        d = np.mean([data[q]["arms"]["dense"] for q in qs])
        rr_ = np.mean([data[q]["arms"]["dense+rerank"] for q in qs])
        ex = np.mean([data[q]["arms"]["expand_fuse"] for q in qs])
        orc = np.mean([max(data[q]["arms"].values()) for q in qs])
        print(f"  {name:16s} dense={d:.3f}  dense+rerank={rr_:.3f}  expand_fuse={ex:.3f}  oracle={orc:.3f}")
    print("\n== by flatness bucket (learnable rule check) ==")
    bk("FLAT", flat[:t]); bk("medium", flat[t:2*t]); bk("PEAKED", flat[2*t:])
    print(f"\n[explore] tagged data -> {outp}  ({len(data)} rows, {len(ARMS)} arms, "
          f"{len(data[qids[0]]['feats'])} probe feats + 768-d qvec)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().n, ap.parse_args().workers)
