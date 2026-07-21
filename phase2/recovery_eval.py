"""Quantify the new primitives: how many gold docs missed@100 by dense do the new
pipelines pull into the top-10?  dense vs smart_search vs retrieve_rerank(500,Qwen)."""
from __future__ import annotations

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common


def main(n=80, pool=500):
    q = json.loads((common.DATA_DIR / "queries.json").read_text())
    qr = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
    s = sac.Session("opensearch", index="fiqa", dim=768, hosts=[common.OS_HOST],
                    embedder=embed, reranker=sac.QwenReranker())
    s.reranker("warm", ["a", "b"])

    def R(ids, g): return len(set(ids[:10]) & g) / len(g) if g else 0.0

    dense, smart, rr = [], [], []
    miss_total = miss_recovered = 0
    for i, x in enumerate(qids):
        g = {d for d, v in qr[x].items() if v > 0}
        d = s.search(q[x], top_k=100, mode="dense")
        d10 = d.top(10).ids()
        missed = g - set(d10)
        miss_total += len(missed)
        sm = s.smart_search(q[x], top_k=10).ids()
        rk = s.retrieve_rerank(q[x], pool_k=pool, top_k=10).ids()
        miss_recovered += len(missed & set(rk))       # gold that dense missed@10 but rerank got
        dense.append(R(d10, g)); smart.append(R(sm, g)); rr.append(R(rk, g))
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{n}", flush=True)

    print(f"\n=== recovery on {n} FiQA queries (pool={pool}, Qwen rerank) — recall@10 ===")
    print(f"  dense@10            {np.mean(dense):.4f}")
    print(f"  smart_search        {np.mean(smart):.4f}   (normalize + rare-term boost)")
    print(f"  retrieve_rerank     {np.mean(rr):.4f}   (wide pool -> Qwen rerank)")
    print(f"\n  gold docs missed by dense@10 that retrieve_rerank pulled into top-10: "
          f"{miss_recovered}/{miss_total}  ({100*miss_recovered/max(1,miss_total):.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=80); ap.add_argument("--pool", type=int, default=500)
    a = ap.parse_args(); main(a.n, a.pool)
