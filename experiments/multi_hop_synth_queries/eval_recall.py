"""recall@10 benchmark on the multi-hop datasets: dense vs tool-calling vs SAC code-mode.

Metric per query: recall@10 = |gold ∩ top10| / N  and  all_golds@10 = (all N gold in top10).
Arms (same retriever = gte-base dense over hotpotqa, same model = gpt-4.1-mini):
  dense  : one dense search, top-10.
  tool   : agent, ONE search per turn — LLM sees each result set and proposes the next query
           (intermediate results IN context); RRF-accumulate; top-10. (budget turns)
  sac    : code-mode — LLM plans all sub-queries in one shot, batch-search each, fuse in code,
           intermediate results NEVER go back to the model; top-10.

    python -m experiments.multi_hop_synth_queries.eval_recall [per_dataset=150] [workers=6]
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM

DATA = Path(__file__).parent / "data"
K = 10


def _rrf(id_lists, k=60):
    score = {}
    for lst in id_lists:
        for rank, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


def dense_ids(session, q, top_k=K):
    return session.search(q, top_k=top_k, mode="dense").ids()


def _titles(session, q, m=5):
    res = session.search(q, top_k=m, mode="dense")
    return [(h.document.metadata or {}).get("title", "?") for h in res]


def tool_arm(session, gen, q, budget=4):
    """One search per turn; the model sees results and proposes the next query (tool mode)."""
    pools, history, cur = [], [], q
    for _ in range(budget):
        res = session.search(cur, top_k=K, mode="dense")
        pools.append(res.ids())
        history.append({"query": cur, "top_titles": [(h.document.metadata or {}).get("title", "?")
                                                      for h in res[:5]]})
        nxt = gen.complete(
            f"Question: {q}\n\nSearches and their top results so far:\n{json.dumps(history, indent=1)}\n\n"
            "This question needs several DIFFERENT supporting documents. Propose ONE more search "
            "query to find a still-missing supporting document, or reply STOP if you have enough. "
            "Reply with just the query on one line.",
            system="You are an iterative search agent.")
        if nxt.strip().upper().startswith("STOP"):
            break
        cur = nxt.strip().splitlines()[0][:200]
    return _rrf(pools)[:K]


def sac_arm(session, gen, q):
    """Code-mode: plan all sub-queries once, batch-search, fuse in code (state out of context)."""
    plan = gen.complete(
        f"Break this question into the distinct factual sub-questions needed to answer it — each "
        f"should target a DIFFERENT entity/document. One per line, no numbering, 2-6 lines.\n\nQ: {q}",
        system="You decompose multi-hop questions into retrieval sub-queries.")
    subs = [s.strip("-•* ").strip() for s in plan.splitlines() if s.strip()][:6] or [q]
    pools = [session.search(sq, top_k=K, mode="dense").ids() for sq in subs]
    pools.append(dense_ids(session, q))          # also keep the whole-question search
    return _rrf(pools)[:K]


def recall(gold, ids):
    g = set(gold); got = g & set(ids[:K])
    return len(got) / len(g), int(g <= set(ids[:K]))


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=128).tolist()

    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed)
    arms = ["dense", "tool", "sac"]
    lock = threading.Lock()
    results = {}

    for ds in (2, 3, 4):
        rows = [json.loads(l) for l in (DATA / f"multihop_{ds}docs_queries.jsonl").open()][:per]
        agg = {a: {"recall": 0.0, "all": 0, "n": 0} for a in arms}

        def one(r):
            q, gold = r["query"], r["gold_ids"]
            out = {}
            out["dense"] = recall(gold, dense_ids(session, q))
            out["tool"] = recall(gold, tool_arm(session, gen, q))
            out["sac"] = recall(gold, sac_arm(session, gen, q))
            with lock:
                for a in arms:
                    agg[a]["recall"] += out[a][0]; agg[a]["all"] += out[a][1]; agg[a]["n"] += 1
                    done = agg["dense"]["n"]
                if done % 25 == 0:
                    print(f"[eval {ds}hop] {done}/{len(rows)}  " +
                          " ".join(f"{a} r@10={agg[a]['recall']/done:.2f}/all={agg[a]['all']/done:.2f}"
                                   for a in arms), flush=True)
            return True

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(as_completed([ex.submit(one, r) for r in rows]))
        n = agg["dense"]["n"]
        results[f"{ds}hop"] = {a: {"recall@10": round(agg[a]["recall"] / n, 4),
                                   "all_golds@10": round(agg[a]["all"] / n, 4)} for a in arms}
        print(f"\n===== {ds}-hop (n={n}) =====")
        for a in arms:
            print(f"  {a:6s}  recall@10={results[f'{ds}hop'][a]['recall@10']:.3f}  "
                  f"all_golds@10={results[f'{ds}hop'][a]['all_golds@10']:.3f}")

    (DATA.parent / "recall_benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"\n[eval] cost ${gen.usage.cost_usd:.2f} | saved recall_benchmark.json")


if __name__ == "__main__":
    main()
