"""Generic per-dataset campaign: ingest a BEIR dataset into OpenSearch and eval
dense vs hybrid vs SAC vs tool-calling. Uniform across all datasets.

    python -m phase2.beir_run --dataset scifact --ingest --n 40
    python -m phase2.beir_run --dataset scifact --n 40        # eval only (already ingested)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from phase2 import beir

RUNS = Path(common.REPO) / "phase2" / "runs"
DIM = 768


def ingest(name, em):
    corpus, _, _ = beir.load(name)
    store = sac.connect("opensearch", index=name, dim=DIM, hosts=[common.OS_HOST])
    store.client.indices.delete(index=name, ignore=[404])
    store.ensure_index(DIM)
    ids = list(corpus)
    B, t0 = 1000, time.time()
    for s in range(0, len(ids), B):
        chunk = ids[s:s + B]
        txt = [(corpus[i]["title"] + ". " + corpus[i]["text"]).strip(". ") for i in chunk]
        vecs = em.encode(txt, normalize_embeddings=True, convert_to_numpy=True,
                         batch_size=256, show_progress_bar=False)
        store.upsert([sac.Document(id=chunk[j], text=corpus[chunk[j]]["text"],
                                   vector=vecs[j].astype(np.float32).tolist(),
                                   metadata={"title": corpus[chunk[j]]["title"]})
                      for j in range(len(chunk))])
        if (s // B) % 20 == 0:
            print(f"[{name}] indexed {s+len(chunk)}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(1)
    print(f"[{name}] ingest done, count={store.count()} ({time.time()-t0:.0f}s)", flush=True)


def main(name, do_ingest, n):
    _, queries, qrels = beir.load(name)
    qids = [q for q in queries if q in qrels and any(v > 0 for v in qrels[q].values())]
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    if do_ingest:
        ingest(name, em)
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()
    gen = LLM()
    s = sac.Session("opensearch", index=name, dim=DIM, hosts=[common.OS_HOST], embedder=embed,
                    reranker=sac.QwenReranker(), generator=gen.as_generator())
    chat = agents.lc_chat()
    s.reranker("warm", ["a", "b"])

    def rec(ids, g): return len(set(ids[:10]) & g) / len(g) if g else 0.0
    def allf(ids, g): return 1.0 if g and g <= set(ids[:10]) else 0.0

    full = {k: [] for k in ["dense_r", "dense_a", "hybrid_r", "hybrid_a"]}
    sub = {k: [] for k in ["dense_r", "hybrid_r", "sac_r", "tool_r",
                           "dense_a", "hybrid_a", "sac_a", "tool_a"]}
    for i, qid in enumerate(qids):
        q, g = queries[qid], {c for c, v in qrels[qid].items() if v > 0}
        d = s.search(q, 10, mode="dense").ids()
        h = s.search(q, 10, mode="hybrid", alpha=0.7).ids()
        full["dense_r"].append(rec(d, g)); full["dense_a"].append(allf(d, g))
        full["hybrid_r"].append(rec(h, g)); full["hybrid_a"].append(allf(h, g))
        if i < n:
            sr = agents.run_sac(s, q, chat=chat, max_retries=1)
            tl = agents.run_tool_calling(s, q, chat=chat, max_retries=1)
            for k, v in [("dense_r", rec(d, g)), ("hybrid_r", rec(h, g)),
                         ("sac_r", rec(sr["ids"], g)), ("tool_r", rec(tl["ids"], g)),
                         ("dense_a", allf(d, g)), ("hybrid_a", allf(h, g)),
                         ("sac_a", allf(sr["ids"], g)), ("tool_a", allf(tl["ids"], g))]:
                sub[k].append(v)
            print(f"[{name}] {i+1}/{n} dense_r={sub['dense_r'][-1]:.2f} "
                  f"sac_r={sub['sac_r'][-1]:.2f} tool_r={sub['tool_r'][-1]:.2f}", flush=True)

    m = lambda k, D: float(np.mean(D[k])) if D[k] else 0.0
    print(f"\n===== {name}: dense/hybrid on {len(qids)} queries =====")
    print(f"  dense  recall@10={m('dense_r',full):.4f}  all_found@10={m('dense_a',full):.4f}")
    print(f"  hybrid recall@10={m('hybrid_r',full):.4f}  all_found@10={m('hybrid_a',full):.4f}")
    print(f"===== {name}: SAC vs tool vs dense on {min(n,len(qids))} queries =====")
    for nm, r, a in [("dense", "dense_r", "dense_a"), ("hybrid", "hybrid_r", "hybrid_a"),
                     ("SAC", "sac_r", "sac_a"), ("tool", "tool_r", "tool_a")]:
        print(f"  {nm:6s} recall@10={m(r,sub):.4f}  all_found@10={m(a,sub):.4f}")
    print(f"  llm cost ${gen.usage.cost_usd:.4f}")
    RUNS.mkdir(exist_ok=True)
    avg_gold = float(np.mean([sum(1 for v in qrels[q].values() if v > 0) for q in qids])) if qids else 0.0
    try:
        n_corpus = int(sac.connect("opensearch", index=name, dim=DIM, hosts=[common.OS_HOST]).count())
    except Exception:
        n_corpus = None
    (RUNS / f"{name}.json").write_text(json.dumps(
        {"dataset": name, "n_corpus": n_corpus, "n_queries_total": len(qids),
         "n_full": len(qids), "n_sub": min(n, len(qids)), "avg_gold_per_query": round(avg_gold, 3),
         "embedder": common.EMB_MODEL, "reranker": "Qwen3-Reranker-0.6B", "llm": "gpt-4.1-mini",
         "llm_cost_usd": round(gen.usage.cost_usd, 4),
         "full": {k: m(k, full) for k in full}, "sub": {k: m(k, sub) for k in sub}}, indent=2))
    print(f"[{name}] saved runs/{name}.json (corpus={n_corpus}, avg_gold={avg_gold:.2f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    main(a.dataset, a.ingest, a.n)
