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
from phase2 import beir
from phase2.learned import LearnedProfile


def main(dataset="fiqa", n=150, split="test"):
    q, qr, index = beir.eval_data(dataset)
    # Evaluate on the split the rules were NOT mined from. Reporting BOTH makes the
    # in-sample/held-out gap visible instead of hiding it (P2-1).
    from phase2.splits import pick
    qids = pick(qr, split, n=None)[:n] if split != "all" else \
        [x for x in qr if any(v > 0 for v in qr[x].values())][:n]
    print(f"[impact] dataset={dataset} split={split} n_qids={len(qids)}", flush=True)
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
    from search_as_code.metrics import compare, bootstrap_ci, format_ci
    print(f"=== {dataset} learned-profile impact  split={split}  (n={len(qids)}) ===")
    print(f"  dense (raw)                 recall@10={m(base_r):.4f}  all_found@10={m(base_a):.4f}")
    print(f"  dense (learned-normalized)  recall@10={m(norm_r):.4f}  all_found@10={m(norm_a):.4f}"
          f"   ({changed_norm} normalized)")
    print(f"  learned synonym-expand+fuse recall@10={m(syn_r):.4f}  all_found@10={m(syn_a):.4f}"
          f"   ({changed_syn} expanded)")
    # deltas WITH intervals — a lift whose CI includes 0 is not a lift
    d_norm_r = compare(norm_r, base_r); d_syn_r = compare(syn_r, base_r)
    d_norm_a = compare(norm_a, base_a); d_syn_a = compare(syn_a, base_a)
    print(f"  base recall@10  {format_ci(*bootstrap_ci(base_r))}")
    for nm, d in (("normalized  recall", d_norm_r), ("synonym-exp recall", d_syn_r),
                  ("normalized  all@10", d_norm_a), ("synonym-exp all@10", d_syn_a)):
        print(f"  delta {nm}: {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] "
              f"{'SIGNIFICANT' if d['significant'] else 'ns'}")
    (common.REPO / "phase2" / "runs" / f"impact_{dataset}_{split}.json").write_text(json.dumps({
        "dataset": dataset, "split": split, "n": len(qids),
        "deltas_vs_base": {"normalized_recall": d_norm_r, "synonym_recall": d_syn_r,
                           "normalized_allfound": d_norm_a, "synonym_allfound": d_syn_a},
        "base": {"r": float(m(base_r)), "a": float(m(base_a))},
        "normalized": {"r": float(m(norm_r)), "a": float(m(norm_a)), "changed": changed_norm},
        "synonym_expand": {"r": float(m(syn_r)), "a": float(m(syn_a)), "changed": changed_syn}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    a = ap.parse_args(); main(a.dataset, a.n, a.split)
