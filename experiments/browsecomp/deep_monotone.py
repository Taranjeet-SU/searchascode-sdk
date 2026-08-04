"""BrowseComp deep-SAC (monotone) vs one-shot — filling the gap (HotpotQA/SU got this, BrowseComp didn't).

Reuses the FIXED session/reranker plumbing from eval_rr2 (no embedder monkeypatch; Qwen reranker
locked + text-truncated). Arms per query, on the SAME eligible sample:
  oneshot   = run_sac(deep=False, max_retries=0)          # single lean pass
  deep_mono = run_sac(deep=True,  monotone=True)           # hop-0 one-shot + RRF-fuse all hops

    python -m experiments.browsecomp.deep_monotone [n=15] [workers=2]
Honest expectation: BrowseComp is near-floor (all_golds@10 ~ 0 over 100k), so deep likely won't help.
"""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import search_as_code as sac
from phase1 import agents
from phase1.llm import LLM
from experiments.browsecomp import bc_common as B
from experiments.browsecomp.eval_rr2 import LockedReranker, recall_at

ARMS = ["oneshot", "deep_mono"]
CALL = {"oneshot": dict(deep=False, max_retries=0), "deep_mono": dict(deep=True, max_retries=3, monotone=True)}


def run_one(store, embedder, reranker, generator, chat, q, gold):
    out = {}
    for arm in ARMS:
        session = sac.Session(store, embedder=embedder, reranker=reranker, generator=generator)
        cnt = {"n": 0}; _o = session.search
        def w(*a, **k):
            cnt["n"] += 1; return _o(*a, **k)
        session.search = w
        try:
            res = agents.run_sac(session, q, chat=chat, judge_chat=chat, k=10, **CALL[arm])
            r10, a10 = recall_at(gold, res["ids"], 10); r20, a20 = recall_at(gold, res["ids"], 20)
            u = res["usage"]
            out[arm] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                        "searches": cnt["n"], "hops": res["hops"], "in": u["input_tokens"], "out": u["output_tokens"]}
        except Exception as e:  # noqa: BLE001
            out[arm] = {"recall@10": 0, "all@10": 0, "recall@20": 0, "all@20": 0,
                        "searches": cnt["n"], "hops": 0, "in": 0, "out": 0, "err": str(e)[:80]}
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    golds = B.load_golds(); queries = B.load_queries()
    corpus_ids = set(json.loads(B.IDS_JSON.read_text()))
    eligible = [q for q, gs in golds.items() if q in queries and queries[q] and all(g in corpus_ids for g in gs)]
    random.seed(0); random.shuffle(eligible); sample = eligible[:n]
    print(f"[bc-deep] sampling {len(sample)} of {len(eligible)} eligible", flush=True)

    gen = LLM(); session = B.load_session(generator=gen.as_generator())
    session.store.build_kw()
    rr = LockedReranker(sac.QwenReranker()); session.reranker = rr
    chat = agents.lc_chat()
    lock = threading.Lock(); records = []; t0 = time.time(); done = [0]

    def one(qid):
        m = run_one(session.store, session.embedder, rr, gen.as_generator(), chat, queries[qid], golds[qid])
        with lock:
            records.append({"qid": qid, "n_gold": len(golds[qid]), **m}); done[0] += 1
            print(f"[bc-deep] {done[0]}/{len(sample)} ({time.time()-t0:.0f}s) "
                  f"oneshot={m['oneshot']['recall@10']:.2f} deep={m['deep_mono']['recall@10']:.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, q) for q in sample]))

    out = {"n": len(records), "arms": {}}
    for a in ARMS:
        agg = {k: sum(r[a][k] for r in records) / len(records) for k in
               ("recall@10", "recall@20", "all@10", "all@20", "searches", "hops", "in", "out")}
        out["arms"][a] = {"recall@10": round(agg["recall@10"], 4), "recall@20": round(agg["recall@20"], 4),
                          "all_golds@10": round(agg["all@10"], 4), "avg_searches": round(agg["searches"], 2),
                          "avg_hops": round(agg["hops"], 2), "avg_in_tokens": int(agg["in"])}
    (B.HERE / "bc_deep_monotone.json").write_text(json.dumps(out, indent=2))
    print("\n===== BrowseComp deep-SAC (monotone) vs one-shot =====")
    for a in ARMS:
        r = out["arms"][a]
        print(f"  {a:10s} r@10 {r['recall@10']:.3f} r@20 {r['recall@20']:.3f} all@10 {r['all_golds@10']:.3f} "
              f"srch {r['avg_searches']} hops {r['avg_hops']} in {r['avg_in_tokens']}")
    print("saved bc_deep_monotone.json")


if __name__ == "__main__":
    main()
