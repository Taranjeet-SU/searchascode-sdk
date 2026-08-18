"""The DEFAULT explore pipeline — structure-emergent, forged from raw OpenSearch queries.

Stages (run before any analysis, per corpus):
  1. EXPLORE with raw OS queries, ORACLE (ceiling) as the stop signal — the LLM authors the retrieval
     strategy per hop (agentic_solve, gold-stopped, up to `max_hops`), capturing the winning strategies.
  2. DEEP JUDGE — the DiagnosticJudge (cross-encoder coverage; held-out balanced acc 0.700
     [0.613, 0.789], matched by a no-LLM logistic gate — issues.md DJ-5/DJ-9) that
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
    judge = DiagnosticJudge(gen)                      # STAGE 2: the deep judge (0.700 held-out; see DJ-5/9)
    skill = SkillLookup(embed)
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    shared = AgentMemory(path=str(HERE / f"explore_{corpus}_memory.jsonl"))   # cross-QUERY skill-wins (thread-safe)
    mlock = threading.Lock()
    print(f"[pipeline] corpus={corpus} train={len(train)} val={len(val)} test={len(test)} "
          f"max_hops={max_hops} workers={workers}", flush=True)

    def solve_one(r, judge_stop, capture=None):
        per_q = AgentMemory()                         # fresh cross-HOP memory (clean, per query)
        with mlock:
            wins = shared.recall(r["query"], k=3, kind="skill_win")           # cross-query recall (seed)
        for w in wins:
            per_q.remember(w.content, kind="skill_win", **(w.meta or {}))
        seeded = len(wins)
        # EXPLORE runs with a structure-NEUTRAL author prompt (FRG-2): with the production
        # "whole" prior, "structure-emergent" was dictated by the prompt, not discovered.
        res = agentic_solve(session, r["query"], gold=r["gold_ids"], generator=gen, judge=judge,
                            skill_lookup=skill, reranker=rr, embedder=embed, judge_stop=judge_stop,
                            max_hops=max_hops, memory=per_q, capture=capture,
                            structure_prior="neutral")
        with mlock:
            for m in per_q.longterm[seeded:]:         # harvest NEW skill-wins into the shared store
                shared.remember(m.content, kind="skill_win", **(m.meta or {}))
        return res

    # STAGE 1 — explore with raw OS queries, ORACLE stop (parallel)
    winning, decomp, ex_rec = [], 0, []
    slock = threading.Lock()

    def explore_one(idx, r):
        cap: list = []
        res = solve_one(r, False, capture=cap)
        codes = res.get("codes") or []
        # WINNING-HOP attribution (FRG-2): credit the code + structure of the hop that
        # actually REACHED the final coverage — not hop 1 (the forced raw-OS probe, EXP-6)
        # and not blindly the last hop. Structure comes from the AST classifier on that hop.
        final_got = max((c.get("got", 0) for c in cap), default=0)
        win_row = next((c for c in cap if c.get("got", 0) == final_got and final_got > 0), None)
        best = (win_row or {}).get("code") or (codes[-1] if codes else "")
        d = ((win_row or {}).get("structure") == "decompose")
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
    # THE SDK GATE (fable.md WS3): the pipeline used to run its own 5-held dense-only gate —
    # thin enough to be a coin flip, and deletable by a merge (issues.md FRG-1, which
    # happened). accept_code_primitive gates on max(dense, hybrid) over up to 30 held
    # queries with a paired-bootstrap CI, persists WHICHEVER SIDE WINS under `name`, and
    # records full provenance either way.
    held_list = (test[:30] or train[:30])
    code, ok, held_mean = forge_from_exploration(gen, session, winning, held_list[:5], name)
    print(f"[stage4] forge synthesis: compiled={bool(code)} smoke over 5 held = {held_mean:.3f}", flush=True)
    gate_report = forge.accept_code_primitive(
        name, f"explored on {corpus}: structure-preserving reusable retriever", code,
        session=session, held=held_list, k=20,
        extra_provenance={"corpus": corpus, "n_winning": len(winning)}) if code else {"accepted": False}
    ok = bool(gate_report.get("accepted"))
    print(f"[stage4-gate] candidate={gate_report.get('candidate_mean')} "
          f"baselines={gate_report.get('baseline_means')} -> accepted={ok} "
          f"(delta vs {gate_report.get('gate_baseline')}: {gate_report.get('delta_vs_baseline')})", flush=True)
    struct = "whole-query" if decomp < len(train) / 2 else "decompose"
    # Derive the skill's retrievers from what the WINNING code actually called (FRG-3) —
    # the old two-branch ternary flattened whatever was discovered into a hardcoded bag.
    calls = " ".join(w["code"] for w in winning)
    retrievers = [r for r, pat in (("dense", "mode='dense'"), ("keyword", "mode='keyword'"),
                                   ("hybrid", "mode='hybrid'"), ("hyde", "hyde"),
                                   ("exact", "_search("), ("decompose", "decompose"))
                  if pat in calls] or ["dense"]
    if "rerank(" in calls:
        retrievers.append("rerank")
    forge.create_skill(f"{corpus}_explored_skill", f"explored {struct} strategy for {corpus}",
                       retrievers, combine="fuse",
                       provenance={"derived_from": f"{len(winning)} winning strategies",
                                   "call_mix": retrievers})
    forge.create_subagent(f"{corpus}_explored_agent", f"solve via the explored {struct} primitive",
                          plan=[name] if ok else [f"{corpus}_explored_skill"])
    # Supersede prior structure verdicts instead of accumulating contradictions (FRG-4).
    forge.refine_prompt(f"[{corpus}] discovered structure = {struct} (decomposed {decomp}/{len(train)} "
                        f"in exploration)", supersedes=f"[{corpus}] discovered structure")
    fstore.save()
    print(f"[stage4] forged primitive accepted={ok} · discovered structure={struct} · "
          f"artifacts: code_primitives={list(fstore.code_primitives)} skills={list(fstore.skills)} subagents={list(fstore.subagents)}", flush=True)

    # STAGE 4b — THE DENSE-DEFAULT GATE.
    # README: "adopts the forged primitive only if it BEATS plain dense on held queries;
    # otherwise it emits session.search(mode='dense'). So SAC never underperforms dense."
    # That gate did not exist (issues.md EXP-5): when the forge was rejected, `reg.get(name)`
    # returned None, stages 5 and 7 silently skipped every query, and `np.mean(...) if x else 0`
    # printed **0.000** — reporting "no primitive was produced" as "the primitive retrieved
    # nothing", which is the pipeline's headline number.
    def _dense_ids(q, k=20):
        try:
            return session.search(q, top_k=k, mode="dense").ids()
        except Exception:
            return []

    # The SDK gate persisted the WINNER under `name` (the accepted candidate, or the best
    # no-LLM baseline as fallback code) — the registry entry IS the deployable strategy.
    selected = "forged" if ok else str(gate_report.get("gate_baseline", "dense"))
    print(f"[stage4b] best-baseline gate SELECTED **{selected}** "
          f"(registry primitive '{name}' runs it either way)", flush=True)
    prim = reg.get(name)

    def _run_selected(q, k=20):
        """What the pipeline actually deploys: the gate's winner (candidate or baseline),
        with a dense last-resort. Never returns [] just because nothing was forged."""
        if prim is not None:
            try:
                return prim.run(session, q, top_k=k)
            except Exception:
                return _dense_ids(q, k)
        return _dense_ids(q, k)

    # STAGE 5 — validate the SELECTED strategy on TRAINING (replicate)
    tr_rec = [_recall(r["gold_ids"], _run_selected(r["query"]))[0] for r in train]
    print(f"[stage5] {selected} strategy on TRAIN recall@20={np.mean(tr_rec) if tr_rec else 0:.3f} "
          f"(exploration was {np.mean(ex_rec):.3f})", flush=True)

    # STAGE 7 — run on TEST with the SELECTED strategy (forged, or dense per the gate)
    te10, te20 = [], []
    for r in test:
        r20, r10 = _recall(r["gold_ids"], _run_selected(r["query"]))
        te20.append(r20); te10.append(r10)
    out = {"corpus": corpus, "max_hops": max_hops,
           "stage1_explore_recall@20": round(float(np.mean(ex_rec)), 3), "decomposed": f"{decomp}/{len(train)}",
           "discovered_structure": struct,
           "stage3_validate_judgestop@20": round(float(np.mean(vj)), 3), "stage3_oraclestop@20": round(float(np.mean(vo)), 3),
           # `selected` says WHICH strategy these numbers describe. Reporting a bare 0 when
           # nothing was forged conflated "no primitive" with "primitive retrieved nothing".
           "selected_strategy": selected,
           "gate_candidate_mean@20": gate_report.get("candidate_mean"),
           "gate_baseline_means@20": gate_report.get("baseline_means"),
           "gate_delta_vs_baseline": gate_report.get("delta_vs_baseline"),
           "forge_accepted": bool(ok),
           "stage5_train@20": round(float(np.mean(tr_rec)), 3) if tr_rec else None,
           "stage7_test_recall@10": round(float(np.mean(te10)), 3) if te10 else None,
           "stage7_test_recall@20": round(float(np.mean(te20)), 3) if te20 else None,
           "artifacts": {"code_primitives": list(fstore.code_primitives), "skills": list(fstore.skills),
                         "subagents": list(fstore.subagents), "rules": fstore.learnings},
           "forged_code": code if ok else None}
    (HERE / f"explore_pipeline_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] explore pipeline =====")
    for k in ("discovered_structure", "stage1_explore_recall@20", "stage3_validate_judgestop@20",
              "stage3_oraclestop@20", "selected_strategy", "gate_candidate_mean@20",
                 "gate_dense_on_held@20", "forge_accepted", "stage5_train@20",
                 "stage7_test_recall@10", "stage7_test_recall@20"):
        print(f"  {k}: {out[k]}")
    print(f"  artifacts: {out['artifacts']}")


if __name__ == "__main__":
    main()
