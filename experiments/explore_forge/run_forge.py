"""explore_forge — the explore pipeline as a self-improving Forge loop.

On synthetic multi-hop data (2/3/4-hop, gold KNOWN), for each TRAIN query:
  EXPLORE with raw OpenSearch queries (run_sac writes atomic queries — keyword/hybrid/hyde/
  decompose/fuse — NOT a canned recipe), **oracle-guided** (gold as the stop signal), up to 10
  depths, using the FULL harness: AgentMemory recall injected as the dynamic prompt each run, and
  sub-question decomposition. We always keep going until the gold docs are found (or 10 hops).

Then FORGE from the wins: create a composed skill + a sub-agent + a learned prompt-rule via
HarnessForge, persist to a HarnessStore. Finally REPLICATE on held-out queries with
`sac.Harness(store=...)` — no free exploration, just the forged modules (memory biases the plan to
the forged skill) — to show the forged modules reproduce the exploration's success. Also trains the
xgboost router (the existing explore capability) so explore now does BOTH.

    python -m experiments.explore_forge.run_forge [per_hop=3] [max_depth=10] [do_xgb=0]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import agents, common
from phase1.llm import LLM
from search_as_code.harness import AgentMemory, Harness, HarnessForge, HarnessStore

DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
HERE = Path(__file__).parent


def explore_hint(memory, query: str) -> str:
    recalled = memory.recall(query, k=4)
    mem = "\n".join(f"- {m.content}" for m in recalled)
    return ("EXPLORE THIS CORPUS WITH RAW QUERIES — do not rely on one canned recipe. Compose atomic "
            "OpenSearch queries: sac.search(q, mode='keyword'|'hybrid'|'dense'), sac.hyde_search, and "
            "sac.decompose_search per sub-fact, then sac.fuse the pools. For a multi-hop question, "
            "DECOMPOSE into sub-facts, retrieve each with its own query, and FUSE (do NOT rerank the "
            "union — it drops per-sub-fact coverage). Iterate until ALL needed docs are found.\n"
            + (f"WHAT WORKED BEFORE (memory):\n{mem}" if mem else ""))


def all_golds(gold, ids, k=10):
    g = set(map(str, gold))
    return int(g <= set(map(str, ids[:k]))), (len(g & set(map(str, ids[:k]))) / len(g) if g else 0.0)


def main():
    per_hop = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    do_xgb = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128).tolist()  # noqa: E731
    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed,
                          generator=gen.as_generator())
    session.reranker = sac.CrossEncoderReranker()
    chat = agents.lc_chat()

    memory = AgentMemory(path=str(HERE / "agent_memory.jsonl"), embedder=session.embedder)
    store = HarnessStore(path=str(HERE / "forge_store"))

    # train (explore+forge) / test (replicate) split across hops
    rows = []
    for hop in (2, 3, 4):
        rs = [json.loads(l) for l in (DATA / f"multihop_{hop}docs_queries.jsonl").open()][:per_hop * 2]
        for r in rs:
            r["n_docs"] = hop
        rows += rs
    random.seed(0); random.shuffle(rows)
    train, test = rows[:len(rows) // 2], rows[len(rows) // 2:]
    print(f"[forge] explore {len(train)} train (max_depth={max_depth}), replicate {len(test)} test", flush=True)

    # ---- PHASE 1: EXPLORE (oracle, up to max_depth, memory-driven dynamic prompt) ----
    t0 = time.time()
    ex = {"solved": 0, "n": 0, "hops": 0}
    for r in train:
        q, gold = r["query"], r["gold_ids"]
        res = agents.run_sac(session, q, chat=chat, k=10, max_retries=max_depth - 1, deep=True,
                             oracle_gold=gold, hint=explore_hint(memory, q))
        solved, frac = all_golds(gold, res["ids"])
        ex["solved"] += solved; ex["n"] += 1; ex["hops"] += res["hops"]
        memory.observe(f"{r['n_docs']}hop \"{q[:60]}\" solved={solved} frac={frac:.2f} in {res['hops']} hops",
                       kind="finding")
        if solved:   # record the win so replication's memory-recall biases the plan to the forged skill
            memory.remember(f"{r['n_docs']}-hop query like \"{q[:80]}\" solved by decompose+fuse "
                            f"in {res['hops']} hops", kind="skill_win", skill="explored_multihop",
                            intent="multi_hop", ndocs=r["n_docs"])
        print(f"[explore] {ex['n']}/{len(train)} {r['n_docs']}hop solved={solved} hops={res['hops']} "
              f"(rate {ex['solved']/ex['n']:.2f})", flush=True)

    # ---- PHASE 2: FORGE from the wins ----
    forged = []
    if ex["solved"] > 0:
        forge = HarnessForge(store, sac.SkillRegistry(embedder=session.embedder), memory)
        forge.create_skill("explored_multihop", "multi-hop queries needing several docs (learned by "
                           "oracle exploration on this corpus)", retrievers=["decompose", "dense", "keyword"],
                           combine="fuse", cost=2, origin="forged")
        forge.create_subagent("sub_fact", "a single sub-fact of a multi-hop query",
                              plan=["dense_lookup", "keyword_search"])
        forge.refine_prompt("Multi-hop: decompose into sub-facts, retrieve each, FUSE — do not rerank "
                            "the union (it drops per-sub-fact coverage).")
        forged = ["explored_multihop", "sub_fact"]
        print(f"[forge] created {forged} + {len(store.learnings)} learned rule(s)", flush=True)

    # ---- PHASE 3: REPLICATE with forged modules (no free exploration) ----
    h = Harness(session, memory=memory, store=store)     # loads forged skills + rules; memory biases plan
    rep = {"all_golds": 0, "frac": 0.0, "n": 0, "used_forged": 0}
    for r in test:
        rr = h.run(r["query"], top_k=10)
        ag, frac = all_golds(r["gold_ids"], rr.ids)
        rep["all_golds"] += ag; rep["frac"] += frac; rep["n"] += 1
        rep["used_forged"] += int(rr.skill == "explored_multihop")
    n = rep["n"] or 1

    out = {
        "config": {"per_hop": per_hop, "max_depth": max_depth, "n_train": len(train), "n_test": len(test)},
        "explore": {"solve_rate(all_golds@10, oracle)": round(ex["solved"] / (ex["n"] or 1), 4),
                    "avg_hops": round(ex["hops"] / (ex["n"] or 1), 2), "n": ex["n"],
                    "wall_s": round(time.time() - t0)},
        "forged_skills": forged, "learned_rules": store.learnings,
        "replicate": {"all_golds@10": round(rep["all_golds"] / n, 4), "avg_frac": round(rep["frac"] / n, 4),
                      "used_forged_skill": rep["used_forged"], "n": rep["n"]},
    }

    if do_xgb:   # keep the existing explore capability: train the router on the same queries
        try:
            exp = sac.explore(session, out=str(HERE / "pack"))
            labeled = [{"query": r["query"], "gold_ids": r["gold_ids"]} for r in rows]
            exp.dataset(queries=labeled, all_golds=True, label_llm=True, label_rerank=True, workers=4)
            m = exp.set_model("xgb").train(cv=3)
            out["xgboost_router"] = {"cv_accuracy": m.get("cv_accuracy"), "oracle": m.get("oracle_coverage")}
        except Exception as e:  # noqa: BLE001
            out["xgboost_router"] = {"error": str(e)[:120]}

    (HERE / "forge_results.json").write_text(json.dumps(out, indent=2))
    print("\n===== explore_forge =====")
    print(json.dumps(out, indent=2))
    print("saved forge_results.json + forge_store/ + agent_memory.jsonl")


if __name__ == "__main__":
    main()
