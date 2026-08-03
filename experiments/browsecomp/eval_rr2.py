"""BrowseComp-Plus rerun (FIXED) — dense / tool / sac / sac_deep with a real Qwen reranker.

Mirrors the PROVEN experiments/browsecomp/eval.py (whose dense arm gives non-zero recall) and
only adds: (1) a Qwen3-Reranker attached to the Session (so `rerank` is real), locked so concurrent
GPU forward passes serialize; (2) sac_deep arms via run_sac(deep=True) with oracle vs LLM judge.

The earlier eval_rr.py returned all-zeros because it ALSO monkeypatched the Session embedder
(`_lock_embedder`), which corrupted query embeddings -> every arm (incl. dense) scored 0. Here we
lock ONLY the reranker and never touch the embedder.

    python -m experiments.browsecomp.eval_rr2 [n_sample=40] [workers=2] [budget=8]
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

from experiments.multi_hop_synth_queries import eval_fair as EF
EF.K = 20                     # retrieve/return depth 20 -> report @10 and @20
from experiments.multi_hop_synth_queries.eval_fair import Tools, tool_harness, code_harness

K_FINAL = 20
GPU_LOCK = threading.Lock()   # serialize reranker forward passes only (NOT the embedder)


class LockedReranker:
    def __init__(self, rr):
        self.rr = rr

    def __call__(self, query, texts):
        # BrowseComp docs avg ~33KB; the Qwen reranker only sees 512 tokens, so truncate to ~2000
        # chars BEFORE tokenizing — same scores, ~15x faster tokenization.
        texts = [(t or "")[:2000] for t in texts]
        with GPU_LOCK:
            return self.rr(query, texts)


def recall_at(gold, ids, k):
    g = set(gold)
    top = set(ids[:k])
    return len(g & top) / len(g), int(g <= top)


def _sac_deep(store, rr, embedder, q, oracle_gold):
    dgen = LLM()
    dsess = sac.Session(store, embedder=embedder, generator=dgen.as_generator(), reranker=rr)
    cnt = {"n": 0}
    _orig = dsess.search

    def counting(*a, **kw):
        cnt["n"] += 1
        return _orig(*a, **kw)
    dsess.search = counting
    out = agents.run_sac(dsess, q, k=10, deep=True, oracle_gold=oracle_gold)
    u = out["usage"]
    ids = [str(x) for x in out["ids"]][:K_FINAL]
    return ids, {"searches": cnt["n"], "turns": out["hops"],
                 "in": u["input_tokens"] + dgen.usage.input_tokens,
                 "out": u["output_tokens"] + dgen.usage.output_tokens}


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    deep_n = int(sys.argv[4]) if len(sys.argv) > 4 else 6   # deep arms are slow; only first deep_n qids

    golds = B.load_golds()
    queries = B.load_queries()
    corpus_ids = set(json.loads(B.IDS_JSON.read_text()))
    print(f"[rr2] queries={len(queries)} golds={len(golds)} corpus={len(corpus_ids)}", flush=True)

    eligible = [qid for qid, gs in golds.items()
                if qid in queries and queries[qid] and all(g in corpus_ids for g in gs)]
    random.seed(0)
    random.shuffle(eligible)
    sample = eligible[:n_sample]
    print(f"[rr2] eligible={len(eligible)} sampling {len(sample)}", flush=True)

    gen = LLM()
    session = B.load_session(generator=gen.as_generator())
    print("[rr2] building keyword index...", flush=True)
    session.store.build_kw()
    print("[rr2] loading Qwen3-Reranker-0.6B...", flush=True)
    rr = LockedReranker(sac.QwenReranker())
    session.reranker = rr
    chat = agents.lc_chat()

    # --- DENSE SANITY CHECK (no reranker involved): confirm gold-matching before the full run ---
    sq = sample[0]
    sdids = session.search(queries[sq], top_k=20, mode="dense").ids()
    hit = set(golds[sq]) & set(sdids)
    print(f"[rr2] SANITY qid={sq} n_gold={len(golds[sq])} dense_top5={sdids[:5]} "
          f"gold_sample={golds[sq][:3]} intersect={len(hit)}", flush=True)

    lock = threading.Lock()
    records = []
    t0 = time.time()
    done = 0

    def one(idx, qid):
        nonlocal done
        q, gold = queries[qid], golds[qid]
        res = {"qid": qid, "n_gold": len(gold)}

        dids = session.search(q, top_k=20, mode="dense").ids()
        r10, a10 = recall_at(gold, dids, 10); r20, a20 = recall_at(gold, dids, 20)
        res["dense"] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                        "searches": 1, "turns": 0, "in": 0, "out": 0}

        tgen = LLM(); tt = Tools(session, tgen, budget)
        tids, tm = tool_harness(chat, tt, q)
        r10, a10 = recall_at(gold, tids, 10); r20, a20 = recall_at(gold, tids, 20)
        res["tool"] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                       "searches": tt.searches, "turns": tm["steps"],
                       "in": tm["lc_in"] + tgen.usage.input_tokens,
                       "out": tm["lc_out"] + tgen.usage.output_tokens}

        sgen = LLM(); st = Tools(session, sgen, budget)
        sids, sm = code_harness(sgen, st, q)
        r10, a10 = recall_at(gold, sids, 10); r20, a20 = recall_at(gold, sids, 20)
        res["sac"] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20,
                      "searches": st.searches, "turns": sm["steps"],
                      "in": sgen.usage.input_tokens, "out": sgen.usage.output_tokens}

        if idx < deep_n:      # deep arms only on the first deep_n qids (they're expensive)
            for arm, og in (("sac_deep_oracle", gold), ("sac_deep_llm", None)):
                try:
                    did, dm = _sac_deep(session.store, rr, session.embedder, q, og)
                    r10, a10 = recall_at(gold, did, 10); r20, a20 = recall_at(gold, did, 20)
                    res[arm] = {"recall@10": r10, "all@10": a10, "recall@20": r20, "all@20": a20, **dm}
                except Exception as e:
                    res[arm] = {"recall@10": 0, "all@10": 0, "recall@20": 0, "all@20": 0,
                                "searches": 0, "turns": 0, "in": 0, "out": 0, "err": str(e)[:80]}

        with lock:
            records.append(res); done += 1
            dl = res.get("sac_deep_llm", {}).get("recall@10")
            print(f"[rr2] {done}/{len(sample)} ({time.time()-t0:.0f}s) "
                  f"dense={res['dense']['recall@10']:.2f} sac={res['sac']['recall@10']:.2f} "
                  f"deep_llm={dl if dl is not None else '-'}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, i, qid) for i, qid in enumerate(sample)]))

    arms = ["dense", "tool", "sac", "sac_deep_oracle", "sac_deep_llm"]
    n = len(records)
    out = {"config": {"n_sample": n, "deep_n": deep_n, "workers": workers, "budget": budget,
                      "corpus_size": len(corpus_ids)}, "arms": {}}
    for a in arms:
        recs = [r for r in records if a in r]     # deep arms only present on first deep_n
        nn = len(recs) or 1
        agg = {}
        for m in ("recall@10", "recall@20", "all@10", "all@20", "searches", "turns", "in", "out"):
            agg[m] = sum(r[a][m] for r in recs) / nn
        out["arms"][a] = {"n": len(recs),
                          "recall@10": round(agg["recall@10"], 4), "recall@20": round(agg["recall@20"], 4),
                          "all_golds@10": round(agg["all@10"], 4), "all_golds@20": round(agg["all@20"], 4),
                          "avg_searches": round(agg["searches"], 2), "avg_turns": round(agg["turns"], 2),
                          "avg_in_tokens": int(agg["in"]), "avg_out_tokens": int(agg["out"])}
    (B.HERE / "bc_recall_rr.json").write_text(json.dumps(out, indent=2))
    with (B.HERE / "bc_perquery_rr.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("\n===== BrowseComp-Plus rerun (Qwen reranker) =====")
    for a in arms:
        r = out["arms"][a]
        print(f"  {a:16s} r@10 {r['recall@10']:.3f}  r@20 {r['recall@20']:.3f}  all@10 {r['all_golds@10']:.3f}  "
              f"srch {r['avg_searches']:.1f}  turns {r['avg_turns']:.1f}  in {r['avg_in_tokens']}")
    print("saved bc_recall_rr.json + bc_perquery_rr.jsonl")


if __name__ == "__main__":
    main()
