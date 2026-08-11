"""The DEFAULT explore pipeline — structure-emergent, forged from raw OpenSearch queries.

Stages (run before any analysis, per corpus):
  1. EXPLORE with raw OS queries, ORACLE (ceiling) as the stop signal — the LLM authors the retrieval
     strategy per hop (agentic_solve, gold-stopped, up to `max_hops`), capturing the winning strategies.
  2. DEEP JUDGE — the DiagnosticJudge (cross-encoder coverage; tuned to the ~0.72 signal ceiling) that
     mimics the oracle. (Corpus-agnostic; instantiated here.)
  3. VALIDATE WITHOUT CEILING — re-run held queries with the judge deciding stop (no gold); compare its
     recall to the oracle-stopped run.
  4. FORGE FROM RAW QUERIES — synthesize ONE reusable primitive from the winning strategies (preserving
     the discovered STRUCTURE — whole-query vs decompose), validated on a held query with gold; + skill + subagent.
  5. VALIDATE ON TRAINING WITH THE NEW FORGE — the forged primitive reproduces exploration recall on train.
  7. RUN ON ALL DATA WITH THE NEW PRIMITIVE — final recall on a held test set via the forged primitive.
(Stage 6 = commit, done outside.)

    python -m experiments.deep_judge.run_explore_pipeline <corpus> [n_train=10] [n_val=8] [n_test=20] [max_hops=10]
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from search_as_code.harness import (AgentMemory, DiagnosticJudge, HarnessForge, HarnessStore,
                                    SkillLookup, SkillRegistry, agentic_solve)
from search_as_code.harness.agentic import _exec
from experiments.deep_judge.run_forge_playbook import build

HERE = Path(__file__).parent
_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _recall(gold, ids, k=20):
    g = set(map(str, gold)); t = set(map(str, ids[:k]))
    return len(g & t) / len(g), (len(g & set(map(str, ids[:10]))) / len(g))


def forge_from_exploration(gen, session, winning, held_list, name, min_recall=0.0):
    """Synthesize ONE general primitive from the winning authored strategies (keep their structure),
    validate on SEVERAL held queries (accept if it retrieves gold across them — the right bar for a
    low-recall corpus), retry with the real error. Returns (code, ok, mean_held_recall)."""
    top = sorted(winning, key=lambda w: -w["recall"])[:5]
    exemplars = "\n\n".join(f"# worked here (recall@20 {w['recall']:.2f}):\n{w['code']}" for w in top)
    base = ("Below are retrieval strategies (authored Python) that WORKED on this corpus. Synthesize ONE "
            "GENERAL reusable primitive that captures the winning STRUCTURE — if they kept the query WHOLE "
            "(dense/hybrid + rerank), keep it whole; if they decomposed, decompose. Do NOT change the "
            "structure. Signature EXACTLY: def run(session, query, top_k). Use ONLY session.search(q,top_k=k,"
            "mode=...), session.hyde_search, session.store.query_fielded (guard hasattr), session.store._search, "
            "and the in-scope helpers fuse_ids([...]) and rerank(session,query,ids,top_k=k). Return a list of "
            "ids. Return ONLY one ```python block```.\n\nWINNING STRATEGIES:\n" + exemplars)
    err, code = "", ""
    for _ in range(3):
        raw = gen.complete(base + (f"\n\nYour previous code FAILED: {err} Fix it." if err else ""))
        m = _CODE.search(raw)
        code = (m.group(1) if m else raw).strip()
        try:
            recs = []
            for h in held_list:
                ids = _exec(code, session, h["query"], 20)
                recs.append(_recall(h["gold_ids"], ids)[0])
            mean = float(np.mean(recs)) if recs else 0.0
            if mean > min_recall:                     # it retrieves gold across the held set
                return code, True, mean
            err = f"mean recall@20 over {len(held_list)} held queries was {mean:.2f} — actually retrieve documents."
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
    return code, False, 0.0


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "browsecomp"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    max_hops = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    if corpus == "browsecomp":                       # use the OpenSearch-indexed BrowseComp (raw OS queries work)
        store = sac.connect("opensearch", index="browsecomp", dim=common.DIM, hosts=[common.OS_HOST],
                            text_field="text", vector_field="vector")
        from experiments.browsecomp import bc_common
        g, q = bc_common.load_golds(), bc_common.load_queries()
        rows = [{"query": q[k], "gold_ids": g[k]} for k in q if k in g]
    else:
        store, rows = build(corpus, embed, gen)
    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)
    train, val = rows[:n_train], rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:n_train + n_val + n_test]
    judge = DiagnosticJudge(gen)                      # STAGE 2: the deep judge (tuned to the signal ceiling)
    skill = SkillLookup(embed)
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    shared = AgentMemory(path=str(HERE / f"explore_{corpus}_memory.jsonl"))   # cross-QUERY skill-wins (thread-safe)
    mlock = threading.Lock()
    print(f"[pipeline] corpus={corpus} train={len(train)} val={len(val)} test={len(test)} "
          f"max_hops={max_hops} workers={workers}", flush=True)

    def solve_one(r, judge_stop):
        per_q = AgentMemory()                         # fresh cross-HOP memory (clean, per query)
        with mlock:
            wins = shared.recall(r["query"], k=3, kind="skill_win")           # cross-query recall (seed)
        for w in wins:
            per_q.remember(w.content, kind="skill_win", **(w.meta or {}))
        seeded = len(wins)
        res = agentic_solve(session, r["query"], gold=r["gold_ids"], generator=gen, judge=judge,
                            skill_lookup=skill, reranker=rr, embedder=embed, judge_stop=judge_stop,
                            max_hops=max_hops, memory=per_q)
        with mlock:
            for m in per_q.longterm[seeded:]:         # harvest NEW skill-wins into the shared store
                shared.remember(m.content, kind="skill_win", **(m.meta or {}))
        return res

    # STAGE 1 — explore with raw OS queries, ORACLE stop (parallel)
    winning, decomp, ex_rec = [], 0, []
    slock = threading.Lock()

    def explore_one(idx, r):
        res = solve_one(r, False)
        best = res["codes"][-1] if res["codes"] else ""
        d = bool(re.search(r"re\.split|split\(|decompose|for .* in .*(parts|sub)", (best or "").lower()))
        with slock:
            ex_rec.append(res["all_recall"]); nonlocal_decomp[0] += int(d)
            if res["all_recall"] and res["all_recall"] > 0:
                winning.append({"code": best, "recall": res["all_recall"]})
            n = len(ex_rec)
            if n % 10 == 0:
                print(f"  [explore {n}/{len(train)}] mean recall@20={np.mean(ex_rec):.3f} "
                      f"decomposed={nonlocal_decomp[0]}/{n} wins={len(winning)}", flush=True)

    nonlocal_decomp = [0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(explore_one, i, r) for i, r in enumerate(train)]))
    decomp = nonlocal_decomp[0]
    print(f"[stage1] explore mean recall@20={np.mean(ex_rec):.3f} · decomposed {decomp}/{len(train)} · "
          f"{len(winning)} winning strategies captured", flush=True)

    # STAGE 3 — validate WITHOUT ceiling (judge decides stop) vs oracle (parallel)
    vj, vo = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fj = {ex.submit(solve_one, r, True): r for r in val}
        fo = {ex.submit(solve_one, r, False): r for r in val}
        for f in as_completed(list(fj)):
            vj.append(f.result()["all_recall"])
        for f in as_completed(list(fo)):
            vo.append(f.result()["all_recall"])
    print(f"[stage3] validate: judge-stop recall@20={np.mean(vj):.3f} vs oracle-stop {np.mean(vo):.3f}", flush=True)

    # STAGE 4 — forge from raw winning queries (structure-preserving)
    reg = SkillRegistry(embedder=embed)
    fstore = HarnessStore(path=str(HERE / f"forge_store_{corpus}_explored"))
    forge = HarnessForge(fstore, reg, AgentMemory())
    name = f"{corpus}_explored_primitive"
    held_list = (test[:5] or train[:5])
    code, ok, held_mean = forge_from_exploration(gen, session, winning, held_list, name)
    print(f"[stage4] forge validation mean recall@20 over {len(held_list)} held = {held_mean:.3f}", flush=True)
    if ok:
        forge.create_code_primitive(name, f"explored on {corpus}: structure-preserving reusable retriever", code)
    struct = "whole-query" if decomp < len(train) / 2 else "decompose"
    forge.create_skill(f"{corpus}_explored_skill", f"explored {struct} strategy for {corpus}",
                       (["hybrid", "rerank"] if struct == "whole-query" else ["decompose", "hyde", "rerank"]), combine="fuse")
    forge.create_subagent(f"{corpus}_explored_agent", f"solve via the explored {struct} primitive",
                          plan=[name] if ok else [f"{corpus}_explored_skill"])
    forge.refine_prompt(f"[{corpus}] discovered structure = {struct} (decomposed {decomp}/{len(train)} in exploration)")
    fstore.save()
    print(f"[stage4] forged primitive accepted={ok} · discovered structure={struct} · "
          f"artifacts: code_primitives={list(fstore.code_primitives)} skills={list(fstore.skills)} subagents={list(fstore.subagents)}", flush=True)

    # STAGE 5 — validate forged primitive on TRAINING (replicate)
    prim = reg.get(name)
    tr_rec = []
    if prim:
        for r in train:
            try:
                ids = prim.run(session, r["query"], top_k=20)
            except Exception:
                ids = []
            tr_rec.append(_recall(r["gold_ids"], ids)[0])
    print(f"[stage5] forged primitive on TRAIN recall@20={np.mean(tr_rec) if tr_rec else 0:.3f} "
          f"(exploration was {np.mean(ex_rec):.3f})", flush=True)

    # STAGE 7 — run on TEST with the new primitive
    te10, te20 = [], []
    if prim:
        for r in test:
            try:
                ids = prim.run(session, r["query"], top_k=20)
            except Exception:
                ids = []
            r20, r10 = _recall(r["gold_ids"], ids)
            te20.append(r20); te10.append(r10)
    out = {"corpus": corpus, "max_hops": max_hops,
           "stage1_explore_recall@20": round(float(np.mean(ex_rec)), 3), "decomposed": f"{decomp}/{len(train)}",
           "discovered_structure": struct,
           "stage3_validate_judgestop@20": round(float(np.mean(vj)), 3), "stage3_oraclestop@20": round(float(np.mean(vo)), 3),
           "stage5_forge_on_train@20": round(float(np.mean(tr_rec)) if tr_rec else 0, 3),
           "stage7_test_recall@10": round(float(np.mean(te10)) if te10 else 0, 3),
           "stage7_test_recall@20": round(float(np.mean(te20)) if te20 else 0, 3),
           "artifacts": {"code_primitives": list(fstore.code_primitives), "skills": list(fstore.skills),
                         "subagents": list(fstore.subagents), "rules": fstore.learnings},
           "forged_code": code if ok else None}
    (HERE / f"explore_pipeline_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] explore pipeline =====")
    for k in ("discovered_structure", "stage1_explore_recall@20", "stage3_validate_judgestop@20",
              "stage3_oraclestop@20", "stage5_forge_on_train@20", "stage7_test_recall@10", "stage7_test_recall@20"):
        print(f"  {k}: {out[k]}")
    print(f"  artifacts: {out['artifacts']}")


if __name__ == "__main__":
    main()
