"""Tool-calling vs SAC code-mode: TOKEN + LATENCY cost, on the strong retriever (fable.md WS6).

Every prior experiment led with relevance; this one leads with the cost axis the audit says is
the structural win: per-query wall-clock latency, input/cached/output tokens, model turns, and
searches — with recall alongside so nobody quotes a cost number whose quality collapsed.

Arms (the FAIR harness from eval_fair — identical Tools, identical budget; only the harness differs):
  dense   one dense search @20 (no LLM)                     — the latency/cost floor
  tool    LangChain tool-calling loop over the shared Tools — tokens grow per hop
  sac     ONE model turn writes a program over the same Tools

Corpora:
  browsecomp_qwen8b   830-query BrowseComp-Plus over the Qwen3-Embedding-8B (4096-d) index;
                      query-side uses Qwen's instruction prefix (reproduce_qwen8b.py convention).
  hotpotqa_qwen8b     the 2/3/4-doc multihop queries over a Qwen3-8B re-embed of the hotpotqa
                      corpus (build with cost_tokens.build_hotpot_qwen8b first).
  hotpotqa            same queries over the original gte-base index (comparison).

Token accounting (P1-13 made explicit): for the sac arm, tokens come from phase1.llm.Usage
(uncached input / cached input / output, separately). For the tool arm, LangChain's
usage_metadata reports TOTAL input; we record it as `in` and add the arm's direct-LLM usage.
`in_uncached_known` marks which accounting an arm uses — do not compare `in` across arms
without reading it.

    python -m experiments.cost_tokens.run_cost browsecomp_qwen8b [n=100] [workers=3] [budget=8]
    python -m experiments.cost_tokens.run_cost hotpotqa_qwen8b   [n=100] [workers=3] [budget=8]
"""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from search_as_code.metrics import bootstrap_ci

from experiments.multi_hop_synth_queries import eval_fair as EF

EF.K = 20
from experiments.multi_hop_synth_queries.eval_fair import Tools, code_harness, tool_harness  # noqa: E402

HERE = Path(__file__).parent
ARMS = ["dense", "tool", "sac"]
QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"
QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def qwen_embedder(max_tokens: int = 512):
    """Query-side Qwen3-Embedding-8B with the instruction prefix (worth +0.13 R@10 on
    BrowseComp — qwen8b_sac issues #4). Docs were indexed plain; the prefix is query-only."""
    import torch
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(QWEN_MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                             trust_remote_code=True)
    em.max_seq_length = max_tokens

    def embed(texts):
        prefixed = [f"Instruct: {QWEN_TASK}\nQuery:{t}" for t in texts]
        return em.encode(prefixed, normalize_embeddings=True, batch_size=8).tolist()
    return embed


def gte_embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return em.encode(list(texts), normalize_embeddings=True, batch_size=64).tolist()
    return embed


def load_corpus(corpus: str, gen):
    """-> (session, rows [{qid, query, gold_ids, tag}])"""
    if corpus.startswith("browsecomp"):
        from experiments.browsecomp import bc_common as B
        golds, queries = B.load_golds(), B.load_queries()
        corpus_ids = set(json.loads(B.IDS_JSON.read_text()))
        eligible = [q for q, gs in golds.items()
                    if q in queries and queries[q] and all(g in corpus_ids for g in gs)]
        random.seed(0)
        random.shuffle(eligible)
        rows = [{"qid": q, "query": queries[q], "gold_ids": golds[q], "tag": "bc"} for q in eligible]
        index = "browsecomp_qwen8b" if corpus.endswith("qwen8b") else "browsecomp"
        embed = qwen_embedder() if corpus.endswith("qwen8b") else gte_embedder()
        dim = 4096 if corpus.endswith("qwen8b") else common.DIM
        session = sac.Session("opensearch", index=index, dim=dim, hosts=[common.OS_HOST],
                              text_field="text", vector_field="vector", embedder=embed,
                              generator=gen.as_generator())
    else:
        data = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
        rows = []
        for ds in (2, 3, 4):
            for r in [json.loads(l) for l in (data / f"multihop_{ds}docs_queries.jsonl").open()][200:300]:
                rows.append({"qid": f"{ds}h_{r['seed_id']}", "query": r["query"],
                             "gold_ids": [str(g) for g in r["gold_ids"]], "tag": f"{ds}hop"})
        index = "hotpotqa_qwen8b" if corpus.endswith("qwen8b") else "hotpotqa"
        embed = qwen_embedder() if corpus.endswith("qwen8b") else gte_embedder()
        dim = 4096 if corpus.endswith("qwen8b") else common.DIM
        session = sac.Session("opensearch", index=index, dim=dim, hosts=[common.OS_HOST],
                              text_field="text", vector_field="vector", embedder=embed,
                              generator=gen.as_generator())
    session.reranker = sac.CrossEncoderReranker()   # MiniLM CE for ALL arms (VRAM budget; matched)
    return session, rows


def recall_at(gold, ids, k):
    g = set(map(str, gold))
    top = set(map(str, ids[:k]))
    return len(g & top) / len(g), int(g <= top)


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "browsecomp_qwen8b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    gen = LLM()
    session, rows = load_corpus(corpus, gen)
    if corpus.startswith("browsecomp"):
        rows = rows[:n]
    else:
        per = {}
        rows = [r for r in rows if per.setdefault(r["tag"], []) is not None
                and len(per[r["tag"]]) < n and (per[r["tag"]].append(r) or True)]
        rows = [r for tag in sorted(per) for r in per[tag]]
    chat = agents.lc_chat()
    print(f"[cost] corpus={corpus} n={len(rows)} workers={workers} budget={budget}", flush=True)

    lock = threading.Lock()
    records, done, t0 = [], 0, time.time()

    def one(r):
        nonlocal done
        q, gold = r["query"], r["gold_ids"]
        res = {"qid": r["qid"], "tag": r["tag"], "n_gold": len(gold)}

        t = time.monotonic()
        dids = session.search(q, top_k=20, mode="dense").ids()
        dt = time.monotonic() - t
        r10, a10 = recall_at(gold, dids, 10)
        res["dense"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                        "searches": 1, "turns": 0, "in": 0, "in_cached": 0, "out": 0,
                        "in_uncached_known": True}

        tgen = LLM()
        tt = Tools(session, tgen, budget)
        t = time.monotonic()
        tids, tm = tool_harness(chat, tt, q)
        dt = time.monotonic() - t
        r10, a10 = recall_at(gold, tids, 10)
        res["tool"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                       "searches": tt.searches, "turns": tm["steps"],
                       "in": tm["lc_in"] + tgen.usage.input_tokens,
                       "in_cached": tgen.usage.cached_input_tokens,
                       "out": tm["lc_out"] + tgen.usage.output_tokens,
                       "in_uncached_known": False}   # lc_in is TOTAL input (P1-13)

        sgen = LLM()
        st = Tools(session, sgen, budget)
        t = time.monotonic()
        sids, sm = code_harness(sgen, st, q)
        dt = time.monotonic() - t
        r10, a10 = recall_at(gold, sids, 10)
        res["sac"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                      "searches": st.searches, "turns": sm["steps"],
                      "in": sgen.usage.input_tokens, "in_cached": sgen.usage.cached_input_tokens,
                      "out": sgen.usage.output_tokens, "in_uncached_known": True}

        with lock:
            records.append(res)
            done += 1
            if done % 10 == 0:
                print(f"[cost] {done}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in rows]):
            fut.result()                              # re-raise worker exceptions (EXP-1)

    metrics = ["recall@10", "all@10", "latency_s", "searches", "turns", "in", "in_cached", "out"]
    out = {"config": {"corpus": corpus, "n": len(records), "workers": workers, "budget": budget,
                      "reranker": "ms-marco MiniLM (all arms)", "k": 20,
                      "embedder": QWEN_MODEL if corpus.endswith("qwen8b") else common.EMB_MODEL,
                      "date": "2026-08-18",
                      "caveats": ["latency measured under worker concurrency — arms face the same "
                                  "contention; compare ratios, not absolutes",
                                  "tool arm's `in` is TOTAL input incl. cached (P1-13); sac arm's "
                                  "`in` is uncached with `in_cached` separate"]},
           "arms": {}, "by_tag": {}}
    tags = sorted({r["tag"] for r in records})
    for a in ARMS:
        agg = {m: sum(r[a][m] for r in records) / len(records) for m in metrics}
        lat = [r[a]["latency_s"] for r in records]
        mean, lo, hi = bootstrap_ci(lat)
        out["arms"][a] = {**{m: round(agg[m], 4) for m in metrics},
                          "latency_ci": [round(mean, 2), round(lo, 2), round(hi, 2)]}
        for tag in tags:
            sub = [r for r in records if r["tag"] == tag]
            out["by_tag"].setdefault(tag, {})[a] = {
                m: round(sum(r[a][m] for r in sub) / len(sub), 4) for m in metrics}

    stem = HERE / f"cost_{corpus}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2))
    with open(f"{stem}_perquery.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n===== {corpus} cost (n={len(records)}, budget={budget}) =====")
    print(f"  {'arm':6s} {'r@10':>6s} {'lat_s':>7s} {'turns':>6s} {'srch':>5s} "
          f"{'in_tok':>8s} {'cached':>7s} {'out':>6s}")
    for a in ARMS:
        r = out["arms"][a]
        print(f"  {a:6s} {r['recall@10']:>6.3f} {r['latency_s']:>7.2f} {r['turns']:>6.2f} "
              f"{r['searches']:>5.1f} {int(r['in']):>8d} {int(r['in_cached']):>7d} {int(r['out']):>6d}")
    print(f"\nwrote {stem}.json + perquery ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
