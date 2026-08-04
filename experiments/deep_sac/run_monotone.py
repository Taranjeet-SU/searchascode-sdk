"""Deep-SAC MONOTONICITY validation: does deep still lose to one-shot?

Three arms, all via phase1.agents.run_sac on the SAME fresh per-query session, so this is an
apples-to-apples deep-vs-one-shot test (unlike the eval_fair one-shot harness):
  - oneshot      = run_sac(deep=False, max_retries=0)         # single lean pass (SAC_SYSTEM)
  - deep_legacy  = run_sac(deep=True,  monotone=False)        # old: return best-confidence hop
  - deep_mono    = run_sac(deep=True,  monotone=True)         # fix: hop-0 = one-shot recipe +
                                                              #      RRF-fuse ALL hops (never lose hop-1)

Writes deep_recall_monotone.json + per-query jsonl. HotpotQA + SU multi-hop, per_hop from argv.
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import search_as_code as sac
from phase1 import agents
from phase1.llm import LLM

from experiments.deep_sac.run_deep_sac import (
    HERE, K, HOPS, make_embedder, instrument, recall,
    hotpot_store, hotpot_datasets, su_store, su_datasets,
)

ARMS = ["oneshot", "deep_legacy", "deep_mono"]
CALL = {
    "oneshot":     dict(deep=False, max_retries=0),
    "deep_legacy": dict(deep=True, max_retries=3, monotone=False),
    "deep_mono":   dict(deep=True, max_retries=3, monotone=True),
}
_KEYS = ["recall", "all", "hops", "searches", "in", "out", "n"]


def run_query(store, embedder, reranker, generator, chat, q, gold):
    out = {}
    for arm in ARMS:
        session = sac.Session(store, embedder=embedder, reranker=reranker, generator=generator)
        cnt = instrument(session)
        try:
            res = agents.run_sac(session, q, chat=chat, judge_chat=chat, k=K, **CALL[arm])
            rc, al = recall(gold, res["ids"])
            u = res["usage"]
            out[arm] = {"recall": rc, "all": al, "hops": res["hops"], "searches": cnt["n"],
                        "in": u["input_tokens"], "out": u["output_tokens"], "error": None}
        except Exception as e:  # noqa: BLE001
            out[arm] = {"recall": 0.0, "all": 0, "hops": 0, "searches": cnt["n"],
                        "in": 0, "out": 0, "error": f"{type(e).__name__}: {e}"}
            print(f"  [ERR {arm}] {e}", flush=True)
    return out


def bench(name, store, embedder, reranker, generator, datasets, per_hop, workers):
    chat = agents.lc_chat()
    lock = threading.Lock()
    result, records = {}, []
    for hop in HOPS:
        rows = datasets[hop][:per_hop]
        agg = {a: dict.fromkeys(_KEYS, 0.0) for a in ARMS}

        def one(r):
            m = run_query(store, embedder, reranker, generator, chat, r["query"], r["gold_ids"])
            with lock:
                for a in ARMS:
                    d = m[a]
                    for kk in ("recall", "all", "hops", "searches", "in", "out"):
                        agg[a][kk] += d[kk]
                    agg[a]["n"] += 1
                    records.append({"corpus": name, "hop": hop, "arm": a, **d})
                n = int(agg[ARMS[0]]["n"])
                if n % 5 == 0:
                    print(f"[{name} {hop}hop] {n}/{len(rows)} " + " ".join(
                        f"{a}=r{agg[a]['recall']/n:.2f}/all{agg[a]['all']/n:.2f}" for a in ARMS), flush=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(as_completed([ex.submit(one, r) for r in rows]))

        n = int(agg[ARMS[0]]["n"])
        result[f"{hop}hop"] = {"n": n, **{a: {
            "recall@10": round(agg[a]["recall"] / n, 4), "all_golds@10": round(agg[a]["all"] / n, 4),
            "avg_hops": round(agg[a]["hops"] / n, 2), "avg_searches": round(agg[a]["searches"] / n, 2),
            "avg_in_tokens": int(agg[a]["in"] / n), "avg_out_tokens": int(agg[a]["out"] / n),
        } for a in ARMS}}
        print(f"\n===== [{name}] {hop}-hop (n={n}) =====", flush=True)
        for a in ARMS:
            rr = result[f"{hop}hop"][a]
            print(f"  {a:12s} r{rr['recall@10']:.3f} all{rr['all_golds@10']:.3f} "
                  f"hops{rr['avg_hops']:.2f} srch{rr['avg_searches']:.1f} in{rr['avg_in_tokens']}", flush=True)
    return result, records


def main():
    per_hop = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    embedder = make_embedder()
    reranker = sac.CrossEncoderReranker()
    generator = LLM().as_generator()
    out = {}
    hres, hrec = bench("hotpotqa", hotpot_store(embedder), embedder, reranker, generator,
                       hotpot_datasets(), per_hop, workers)
    out["hotpotqa"] = hres
    sres, srec = bench("su", su_store(embedder, generator), embedder, reranker, generator,
                       su_datasets(), per_hop, workers)
    out["su"] = sres
    (HERE / "deep_recall_monotone.json").write_text(json.dumps(out, indent=2))
    with (HERE / "deep_recall_monotone_perquery.jsonl").open("w") as f:
        for r in hrec + srec:
            f.write(json.dumps(r) + "\n")
    print("\nwrote deep_recall_monotone.json")


if __name__ == "__main__":
    main()
