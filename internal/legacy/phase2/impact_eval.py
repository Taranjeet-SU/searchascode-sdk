"""Deterministic impact of the LEARNED PROFILE (no LLM at eval time), for ANY dataset:
dense(raw query) vs dense(learned-normalized) vs learned-synonym-expand-fused.
Reports recall@10 and all_found@10 (multi-hop-sensitive).

    python -m phase2.impact_eval --dataset fiqa --n 150
    python -m phase2.impact_eval --dataset hotpotqa --n 150
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from internal.legacy.phase2 import beir
from internal.legacy.phase2.learned import LearnedProfile


def main(dataset="fiqa", n=150):
    q, qr, index = beir.eval_data(dataset)
    qids = [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
    s = sac.Session("opensearch", index=index, dim=common.DIM, hosts=[common.OS_HOST], embedder=embed)
    prof = LearnedProfile.load(dataset)

    def R(ids, g): return len(set(ids[:10]) & g) / len(g) if g else 0.0
    def A(ids, g): return 1.0 if g and g <= set(ids[:10]) else 0.0

    base_r, norm_r, syn_r = [], [], []
    base_a, norm_a, syn_a = [], [], []
    changed_norm = changed_syn = 0
    for x in qids:
        g = {d for d, v in qr[x].items() if v > 0}
        bids = s.search(q[x], 10, mode="dense").ids()
        base_r.append(R(bids, g)); base_a.append(A(bids, g))
        nq = prof.normalize(q[x]); changed_norm += (nq != q[x])
        nids = s.search(nq, 10, mode="dense").ids()
        norm_r.append(R(nids, g)); norm_a.append(A(nids, g))
        variants = prof.expand_seeds(nq); changed_syn += (len(variants) > 1)
        sids = s.search_many(variants, top_k=10, mode="dense").top(10).ids()
        syn_r.append(R(sids, g)); syn_a.append(A(sids, g))

    m = np.mean
    print(f"=== {dataset} learned-profile impact (n={len(qids)}) ===")
    print(f"  dense (raw)                 recall@10={m(base_r):.4f}  all_found@10={m(base_a):.4f}")
    print(f"  dense (learned-normalized)  recall@10={m(norm_r):.4f}  all_found@10={m(norm_a):.4f}"
          f"   ({changed_norm} normalized)")
    print(f"  learned synonym-expand+fuse recall@10={m(syn_r):.4f}  all_found@10={m(syn_a):.4f}"
          f"   ({changed_syn} expanded)")
    (common.REPO / "phase2" / "runs" / f"impact_{dataset}.json").write_text(json.dumps({
        "dataset": dataset, "n": len(qids),
        "base": {"r": float(m(base_r)), "a": float(m(base_a))},
        "normalized": {"r": float(m(norm_r)), "a": float(m(norm_a)), "changed": changed_norm},
        "synonym_expand": {"r": float(m(syn_r)), "a": float(m(syn_a)), "changed": changed_syn}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa"); ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args(); main(a.dataset, a.n)
