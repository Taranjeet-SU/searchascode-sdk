"""HotpotQA multi-hop retrieval eval: dense vs hybrid vs SAC vs tool-calling.

Multi-hop needs BOTH supporting docs, so we report recall@10 AND all_found@10
(fraction of queries where EVERY gold doc is in the top-10) — the metric where a
decompose/fan-out agent should beat a single dense query.

    python -m phase2.hotpot_eval --n 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM

DATA = Path(common.REPO) / "phase2" / "data"
INDEX = "hotpotqa"
DIM = 768


def main(n=60):
    queries = json.loads((DATA / "hotpot_queries.json").read_text())
    qrels = json.loads((DATA / "hotpot_qrels.json").read_text())
    qids = list(queries)
    from sentence_transformers import SentenceTransformer
    import torch
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()
    gen = LLM()
    s = sac.Session("opensearch", index=INDEX, dim=DIM, hosts=[common.OS_HOST], embedder=embed,
                    reranker=sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2"),
                    generator=gen.as_generator())
    chat = agents.lc_chat()
    s.reranker("warm", ["a", "b"])

    def rec(ids, gold):
        return len(set(ids[:10]) & gold) / len(gold) if gold else 0.0

    def allf(ids, gold):
        return 1.0 if gold and gold <= set(ids[:10]) else 0.0

    # dense/hybrid on ALL sampled queries (stable baseline); SAC/tool on first n
    full = {"dense_r": [], "dense_a": [], "hybrid_r": [], "hybrid_a": []}
    sub = {k: [] for k in ["dense_r", "dense_a", "hybrid_r", "hybrid_a",
                            "sac_r", "sac_a", "tool_r", "tool_a"]}
    for i, qid in enumerate(qids):
        q = queries[qid]
        gold = {c for c, sc in qrels[qid].items() if sc > 0}
        d = s.search(q, 10, mode="dense").ids()
        h = s.search(q, 10, mode="hybrid", alpha=0.8).ids()
        full["dense_r"].append(rec(d, gold)); full["dense_a"].append(allf(d, gold))
        full["hybrid_r"].append(rec(h, gold)); full["hybrid_a"].append(allf(h, gold))
        if i < n:
            sacr = agents.run_sac(s, q, chat=chat, max_retries=1)
            tl = agents.run_tool_calling(s, q, chat=chat, max_retries=1)
            sub["dense_r"].append(rec(d, gold)); sub["dense_a"].append(allf(d, gold))
            sub["hybrid_r"].append(rec(h, gold)); sub["hybrid_a"].append(allf(h, gold))
            sub["sac_r"].append(rec(sacr["ids"], gold)); sub["sac_a"].append(allf(sacr["ids"], gold))
            sub["tool_r"].append(rec(tl["ids"], gold)); sub["tool_a"].append(allf(tl["ids"], gold))
            print(f"{i+1}/{n} dense_r={sub['dense_r'][-1]:.2f} sac_r={sub['sac_r'][-1]:.2f} "
                  f"dense_all={sub['dense_a'][-1]:.0f} sac_all={sub['sac_a'][-1]:.0f} hops={sacr['hops']}", flush=True)

    m = lambda k, D: float(np.mean(D[k])) if D[k] else 0.0
    print(f"\n===== HotpotQA multi-hop (dense/hybrid on {len(qids)}) =====")
    print(f"  dense   recall@10={m('dense_r',full):.4f}  all_found@10={m('dense_a',full):.4f}")
    print(f"  hybrid  recall@10={m('hybrid_r',full):.4f}  all_found@10={m('hybrid_a',full):.4f}")
    print(f"\n===== SAC vs tool vs dense on the same {n} queries =====")
    for name, r, a in [("dense", "dense_r", "dense_a"), ("hybrid", "hybrid_r", "hybrid_a"),
                       ("SAC", "sac_r", "sac_a"), ("tool", "tool_r", "tool_a")]:
        print(f"  {name:7s} recall@10={m(r,sub):.4f}  all_found@10={m(a,sub):.4f}")
    print(f"\n  llm cost: ${gen.usage.cost_usd:.4f}")
    (common.REPO / "phase2" / "runs").mkdir(exist_ok=True)
    (common.REPO / "phase2" / "runs" / "hotpot.json").write_text(json.dumps(
        {"n_full": len(qids), "n_sub": n,
         "full": {k: m(k, full) for k in full}, "sub": {k: m(k, sub) for k in sub}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60)
    main(ap.parse_args().n)
