"""3-arm BrowseComp-Plus benchmark: dense vs tool-calling vs SAC code-mode.

Reuses the FAIR harness (Tools / tool_harness / code_harness) from
experiments.multi_hop_synth_queries.eval_fair — identical toolset, matched search
budget; only the harness differs. Retrieval depth is 20 so we can report recall@10
and recall@20 from the same run.

    python -m experiments.browsecomp.eval [n_sample=60] [workers=4] [budget=8]
"""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from phase1 import agents
from phase1.llm import LLM
from experiments.browsecomp import bc_common as B

# reuse the fair harness, but retrieve/return depth 20 (report @10 and @20)
from experiments.multi_hop_synth_queries import eval_fair as EF
EF.K = 20
from experiments.multi_hop_synth_queries.eval_fair import Tools, tool_harness, code_harness

ARMS = ["dense", "tool", "sac"]


def recall_at(gold, ids, k):
    g = set(gold)
    top = set(ids[:k])
    return len(g & top) / len(g), int(g <= top)


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    golds = B.load_golds()
    queries = B.load_queries()
    corpus_ids = set(json.loads(B.IDS_JSON.read_text()))
    print(f"[eval] queries={len(queries)} golds={len(golds)} corpus={len(corpus_ids)}", flush=True)

    # eligible = query has text + ALL golds present in corpus (so all_golds achievable)
    eligible = [qid for qid, gs in golds.items()
                if qid in queries and queries[qid]
                and all(g in corpus_ids for g in gs)]
    random.seed(0)
    random.shuffle(eligible)
    sample = eligible[:n_sample]
    print(f"[eval] eligible={len(eligible)} sampling {len(sample)} queries", flush=True)

    gen = LLM()
    session = B.load_session(generator=gen.as_generator())
    print("[eval] pre-building keyword (BM25) index...", flush=True)
    t_kw = time.time()
    session.store.build_kw()
    print(f"[eval] keyword index ready in {time.time()-t_kw:.0f}s", flush=True)
    chat = agents.lc_chat()

    lock = threading.Lock()
    records = []
    t0 = time.time()
    done = 0

    def one(qid):
        nonlocal done
        q, gold = queries[qid], golds[qid]
        res = {"qid": qid, "n_gold": len(gold)}

        # arm 1: dense single-shot (top-20)
        dids = session.search(q, top_k=20, mode="dense").ids()
        r10, a10 = recall_at(gold, dids, 10)
        r20, a20 = recall_at(gold, dids, 20)
        res["dense"] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                        "searches": 1, "turns": 0, "in": 0, "out": 0}

        # arm 2: tool-calling
        tgen = LLM(); tt = Tools(session, tgen, budget)
        tids, tm = tool_harness(chat, tt, q)
        r10, a10 = recall_at(gold, tids, 10)
        r20, a20 = recall_at(gold, tids, 20)
        res["tool"] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                       "searches": tt.searches, "turns": tm["steps"],
                       "in": tm["lc_in"] + tgen.usage.input_tokens,
                       "out": tm["lc_out"] + tgen.usage.output_tokens}

        # arm 3: SAC code-mode
        sgen = LLM(); st = Tools(session, sgen, budget)
        sids, sm = code_harness(sgen, st, q)
        r10, a10 = recall_at(gold, sids, 10)
        r20, a20 = recall_at(gold, sids, 20)
        res["sac"] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                      "searches": st.searches, "turns": sm["steps"],
                      "in": sgen.usage.input_tokens, "out": sgen.usage.output_tokens}

        with lock:
            records.append(res)
            done += 1
            if done % 5 == 0:
                print(f"[eval] {done}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, qid) for qid in sample]))

    # aggregate
    n = len(records)
    metrics = ["recall@10", "recall@20", "all@10", "all@20", "searches", "turns", "in", "out"]
    out = {"config": {"n_sample": n, "workers": workers, "budget": budget,
                      "corpus_size": len(corpus_ids), "k_list": [10, 20]}, "arms": {}}
    for a in ARMS:
        agg = {m: sum(r[a][m] for r in records) / n for m in metrics}
        out["arms"][a] = {
            "recall@10": round(agg["recall@10"], 4),
            "recall@20": round(agg["recall@20"], 4),
            "all_golds@10": round(agg["all@10"], 4),
            "all_golds@20": round(agg["all@20"], 4),
            "avg_searches": round(agg["searches"], 2),
            "avg_turns": round(agg["turns"], 2),
            "avg_in_tokens": int(agg["in"]),
            "avg_out_tokens": int(agg["out"]),
        }

    (B.HERE / "bc_recall.json").write_text(json.dumps(out, indent=2))
    with (B.HERE / "bc_perquery.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n===== BrowseComp-Plus 3-arm (n={n}, budget={budget}, corpus={len(corpus_ids)}) =====")
    hdr = f"  {'arm':6s} {'r@10':>6s} {'r@20':>6s} {'all@10':>7s} {'srch':>5s} {'turns':>6s} {'in_tok':>8s} {'out_tok':>8s}"
    print(hdr)
    for a in ARMS:
        r = out["arms"][a]
        print(f"  {a:6s} {r['recall@10']:>6.3f} {r['recall@20']:>6.3f} {r['all_golds@10']:>7.3f} "
              f"{r['avg_searches']:>5.1f} {r['avg_turns']:>6.1f} {r['avg_in_tokens']:>8d} {r['avg_out_tokens']:>8d}")
    print(f"\n[eval] saved bc_recall.json + bc_perquery.jsonl ({n} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
