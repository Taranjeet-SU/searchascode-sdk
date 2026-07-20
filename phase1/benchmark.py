"""100-query benchmark: base search vs MCP tool-calling vs SAC.

For each query runs all three paths, records a full per-query trace (generated
code, tool steps, ids, latency, tokens, cost) to JSONL for the UI, and reports
Recall@10 / nDCG@10 / MRR@10, average latency, token cost, and cache hit rate
per path.
"""

from __future__ import annotations

import argparse
import copy
import json
import time

from sentence_transformers import SentenceTransformer
import torch

import search_as_code as sac
from phase1 import agents, common, metrics
from phase1.llm import LLM, Usage

common.load_env()


def _merge_gen_usage(result: dict, gen: LLM, before: Usage) -> None:
    """Fold the Session generator's token delta (rephrase/expand run *inside* a
    path) into that path's usage so cost accounting is complete."""
    d_in = gen.usage.input_tokens - before.input_tokens
    d_cache = gen.usage.cached_input_tokens - before.cached_input_tokens
    d_out = gen.usage.output_tokens - before.output_tokens
    d_calls = gen.usage.calls - before.calls
    u = result["usage"]
    u["input_tokens"] += d_in
    u["cached_input_tokens"] += d_cache
    u["output_tokens"] += d_out
    u["calls"] += d_calls
    p = common.LLM_PRICE
    u["cost_usd"] = round(
        (u["input_tokens"] * p["input"] + u["cached_input_tokens"] * p["cached_input"]
         + u["output_tokens"] * p["output"]) / 1_000_000, 6)


def main(n: int = 100, k: int = 10, reranker_model: str = "BAAI/bge-reranker-base") -> None:
    queries = json.loads((common.DATA_DIR / "queries.json").read_text())
    qrels = json.loads((common.DATA_DIR / "qrels.json").read_text())
    qids = [q for q in qrels if any(s > 0 for s in qrels[q].values())][:n]
    print(f"[bench] {len(qids)} queries, reranker={reranker_model}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda ts: em.encode(list(ts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()
    gen = LLM()  # shared generator for internal rephrase/expand, usage-tracked
    session = sac.Session("opensearch", index=common.INDEX, dim=common.DIM, hosts=[common.OS_HOST],
                          embedder=embed, reranker=sac.CrossEncoderReranker(reranker_model),
                          generator=gen.as_generator())
    chat = agents.lc_chat()  # shared → warms the prompt cache across queries

    traces_path = common.RUNS_DIR / "bench_traces.jsonl"
    rankings = {"base": {}, "tool_calling": {}, "sac": {}}
    per_path_usage = {"base": [], "tool_calling": [], "sac": []}
    per_path_lat = {"base": [], "tool_calling": [], "sac": []}

    with open(traces_path, "w") as tf:
        for i, qid in enumerate(qids):
            q = queries[qid]
            gold = [d for d, s in qrels[qid].items() if s > 0]
            rec = {"qid": qid, "query": q, "gold": gold, "paths": {}}
            for path, fn in (("base", lambda: agents.run_base(session, q, k=k)),
                             ("sac", lambda: _run(agents.run_sac, session, q, chat, k, gen)),
                             ("tool_calling", lambda: _run(agents.run_tool_calling, session, q, chat, k, gen))):
                r = fn()
                rankings[path][qid] = r["ids"]
                per_path_usage[path].append(r["usage"])
                per_path_lat[path].append(r["latency_s"])
                r["recall@10"] = metrics.recall_at_k(r["ids"], gold, k)
                rec["paths"][path] = r
            tf.write(json.dumps(rec) + "\n"); tf.flush()
            if (i + 1) % 10 == 0:
                print(f"[bench] {i+1}/{len(qids)}")

    # ---- aggregate ----
    summary = {}
    for path in ("base", "tool_calling", "sac"):
        m = metrics.evaluate(rankings[path], qrels, k=k)
        us = per_path_usage[path]
        tot_uncached = sum(u["input_tokens"] for u in us)         # billed at full rate
        tot_cache = sum(u["cached_input_tokens"] for u in us)     # billed at cached rate
        tot_prompt = tot_uncached + tot_cache                    # total input tokens sent
        tot_out = sum(u["output_tokens"] for u in us)
        summary[path] = {
            **{kk: round(v, 4) for kk, v in m.items()},
            "avg_latency_s": round(sum(per_path_lat[path]) / len(per_path_lat[path]), 3),
            "total_cost_usd": round(sum(u["cost_usd"] for u in us), 4),
            "avg_calls": round(sum(u["calls"] for u in us) / len(us), 2),
            "total_input_tokens": tot_prompt, "total_output_tokens": tot_out,
            "cached_input_tokens": tot_cache,
            "cache_hit_rate": round(tot_cache / tot_prompt, 3) if tot_prompt else 0.0,
        }
    (common.RUNS_DIR / "bench_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Phase 1 benchmark: base vs tool-calling vs SAC (FiQA, "
          f"{summary['base']['n_queries']} queries) ===")
    cols = ["recall@10", "ndcg@10", "mrr@10", "avg_latency_s", "avg_calls",
            "total_input_tokens", "cache_hit_rate", "total_cost_usd"]
    hdr = f"{'path':14s} " + " ".join(f"{c:>18s}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for path in ("base", "tool_calling", "sac"):
        s = summary[path]
        print(f"{path:14s} " + " ".join(f"{s[c]:>18}" for c in cols))
    print("\n[bench] traces ->", traces_path)
    print("[bench] summary ->", common.RUNS_DIR / "bench_summary.json")


def _run(fn, session, q, chat, k, gen):
    before = copy.copy(gen.usage)
    r = fn(session, q, chat=chat, k=k)
    _merge_gen_usage(r, gen, before)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    args = ap.parse_args()
    main(n=args.n, reranker_model=args.reranker)
