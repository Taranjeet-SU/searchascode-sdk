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
import os
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
from experiments.multi_hop_synth_queries.eval_fair import (  # noqa: E402
    TOOL_SYS,
    Tools,
    code_harness,
    tool_harness,
)

HERE = Path(__file__).parent
ARMS = ["dense", "tool", "sac", "tool_explored", "sac_explored", "sac_product", "tool_product"]
PRODUCT_MAX_HOPS = 5      # the product flow's escalation cap; explore/judge-tuning use 10
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
            for i, r in enumerate([json.loads(l) for l in (data / f"multihop_{ds}docs_queries.jsonl").open()][200:300]):
                rows.append({"qid": f"{ds}h_{200 + i}", "query": r["query"],
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


def load_explore_seed(corpus):
    """What explore→forge learned about THIS corpus: guidance text + the gate-selected forged
    primitive as a runnable. The unseeded arms get eval_fair's generic multi-hop brief — which
    says DECOMPOSE, the exact strategy explore found wrong on BrowseComp (whole-query, 0/30
    decomposed; gate selected dense). Seeding is the explore-first workflow the SDK documents."""
    from search_as_code.harness.forge import CodePrimitive, HarnessStore
    stem = "browsecomp" if corpus.startswith("browsecomp") else "hotpot"
    store_dir = Path(__file__).parents[1] / "deep_judge" / f"forge_store_{stem}_explored"
    pipe_json = Path(__file__).parents[1] / "deep_judge" / f"explore_pipeline_{stem}.json"
    if not store_dir.exists():
        return "", None
    store = HarnessStore(path=str(store_dir))
    rules = [r for r in store.learnings if "discovered structure" in r][-1:]
    selected = ""
    try:
        selected = json.loads(pipe_json.read_text()).get("selected_strategy", "")
    except Exception:
        pass
    # THE GATE'S DECISION IS THE PRODUCT of explore — deliver it, not the store's candidate.
    # v1 of this loader handed the arms the store's code primitive, which on BrowseComp is the
    # REJECTED whole-query fusion (gate: forged 0.0 = dense 0.0 -> selected dense). The seeded
    # arm then ran a recall-diluting fusion on every query and scored 0.169 vs dense 0.265.
    baseline_code = {
        "dense": "def run(session, query, top_k):\n    return session.search(query, top_k=top_k, mode='dense').ids()",
        "hybrid": "def run(session, query, top_k):\n    return session.search(query, top_k=top_k, mode='hybrid').ids()",
        "keyword": "def run(session, query, top_k):\n    return session.search(query, top_k=top_k, mode='keyword').ids()",
    }
    prim = next(iter(store.code_primitives.values()), None)
    if selected in baseline_code:
        vetted_code, vetted_desc = baseline_code[selected], f"one whole-query {selected} search"
    elif prim is not None:
        vetted_code, vetted_desc = prim.code, "the forged strategy"
    else:
        return "", None
    guidance = ("\n\nEXPLORE FINDINGS on THIS corpus (from a prior explore->forge run — these OVERRIDE "
                "the generic strategy above): " + "; ".join(rules)
                + (f". The held-out acceptance gate selected '{selected}' — authored alternatives "
                   f"(decomposition, multi-mode fusion) did NOT beat it and DILUTED recall." if selected else "")
                + f" The vetted strategy is {vetted_desc}: call forged(query) and RETURN it. Do NOT add "
                  "extra pools, decompose, or fuse by default — escalate with ONE targeted keyword search "
                  "on exact rare tokens ONLY if the question hinges on a constraint (a year, a name, a "
                  "part number) that a semantic search demonstrably blurs.")
    # Return the SKILL OBJECT: eval_fair's Tools.forged calls `.run(session, query, top_k)` —
    # a bare lambda broke that contract and every forged() call AttributeError'd internally,
    # so earlier "seeded" rows measured guidance-text lift only, not the forged primitive.
    sk = CodePrimitive(name=f"{stem}_gate_selected", when_to_use=vetted_desc, code=vetted_code).to_skill()
    return guidance, sk


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
    seed_guidance, forged_fn = load_explore_seed(corpus)
    only = [a for a in os.environ.get("SAC_ARMS", "").split(",") if a.strip()]
    premise = os.environ.get("SAC_JUDGE_PREMISE", "coverage")   # judge premise for product arms
    from search_as_code.harness.memory import AgentMemory as _AM
    shared_mem = _AM(path=str(HERE / f"product_memory_{corpus}.jsonl"))   # cross-query skill wins
    mem_lock = threading.Lock()
    arms = ARMS if seed_guidance else ["dense", "tool", "sac"]
    if only:
        arms = [a for a in arms if a in only or a == "dense"]
    print(f"[cost] explore seed: {'LOADED' if seed_guidance else 'none'} "
          f"(forged runnable: {bool(forged_fn)})", flush=True)

    # Crash-proof + resumable: per-query rows flush to the JSONL as they complete, and prior
    # rows are reloaded on restart (learned the hard way — a spend-limit 429 killed a run at
    # 250/300 with nothing persisted; the LEG-5 lesson applied to this script's own output).
    tag = os.environ.get("SAC_COST_TAG", "")          # e.g. "_qwen3-8b" — keeps models' rows apart
    pq_path = HERE / f"cost_{corpus}{tag}_perquery.jsonl"
    records = []
    if pq_path.exists():
        by_qid = {}
        for ln in pq_path.open():
            if ln.strip():
                r = json.loads(ln)
                by_qid[r["qid"]] = r                  # last write wins (re-runs supersede)
        records = list(by_qid.values())
        # a row is done only if it carries EVERY arm of the current run (seeded arms re-run old rows)
        have = {r["qid"] for r in records if all(a in r for a in arms)}
        records = [r for r in records if r["qid"] in have]
        rows = [r for r in rows if r["qid"] not in have]
        print(f"[cost] resuming: {len(have)} complete rows kept, {len(rows)} to run", flush=True)
    pq_file = pq_path.open("a")
    print(f"[cost] corpus={corpus} n={len(rows)} workers={workers} budget={budget}", flush=True)

    lock = threading.Lock()
    done, t0 = 0, time.time()

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

        if "tool" in arms:
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

        if "sac" in arms:
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

        # THE PRODUCT FLOW, both harnesses: gate-selected baseline at hop 0 -> deep judge ->
        # escalate ONLY on FAIL, judge between hops, cap PRODUCT_MAX_HOPS. This is the policy
        # explore learned and the gate enforced — the arm the one-shot rows can't represent.
        # (explore + oracle->judge tuning keep max_hops=10; these are SDK params — see examples.)
        if seed_guidance and forged_fn is not None:
            import numpy as np
            import search_as_code.primitives as P
            from search_as_code.harness import diagnostic_judge as djm
            from search_as_code.harness.agentic import agentic_solve
            from search_as_code.harness.diagnostic_judge import DiagnosticJudge
            from search_as_code.harness.memory import AgentMemory
            from search_as_code.harness.rag_techniques import SkillLookup
            emb = session.embedder.embed              # Session wraps the fn; .embed is the call
            skill = SkillLookup(emb)                  # "when to call what" recipes (as in the pipeline)
            # per-query memory seeded from the shared cross-query store; wins harvested back
            with mem_lock:
                wins = shared_mem.recall(q, k=3, kind="skill_win")
            per_q = AgentMemory()
            for w in wins:
                per_q.remember(w.content, kind="skill_win", **(w.meta or {}))
            n_seeded = len(wins)

            def judge_hop0(pgen, judge):
                """baseline ids + the judge's verdict on them. -> (base_ids, verdict, subs)"""
                base_ids = [str(i) for i in forged_fn.run(session, q, top_k=50)]
                try:
                    subs = [s for s in (P.decompose(q, pgen.as_generator()) or [q]) if s.strip()][:6] or [q]
                except Exception:
                    subs = [q]
                docs = {d.id: (d.text or "")[:700] for d in session.store.get(base_ids[:10])}
                texts = [docs.get(i, "") for i in base_ids[:10]]
                sub_vecs = np.asarray(emb(subs), dtype=np.float32)
                cand_vecs = (np.asarray(emb(texts), dtype=np.float32) if texts
                             else np.zeros((0, sub_vecs.shape[1]), np.float32))
                cov = djm.coverage_signals(subs, sub_vecs, texts, cand_vecs, session.reranker)
                csc = djm.candidate_scores(session.reranker, q, texts)
                cands = [{"id": i, "score": s, "snippet": t_}
                         for (i, t_), s in zip(zip(base_ids[:10], texts), csc)]
                return base_ids, judge.judge(q, subs, cands, cov), subs, csc

            def rrf_ids(lists, base_weight=2.0):
                """Weighted RRF: list 0 is the GATE-VETTED baseline and outweighs escalation
                pools — plain RRF let 5 hops of escalation noise evict a hop-0 gold at dense
                rank 8 from the fused top-10, turning a judge false-FAIL into recall 0."""
                agg2: dict = {}
                for li, lst in enumerate(lists):
                    w = base_weight if li == 0 else 1.0
                    for rank, did in enumerate(lst):
                        agg2[did] = agg2.get(did, 0.0) + w / (60 + rank + 1)
                return [d for d, _ in sorted(agg2.items(), key=lambda x: -x[1])]

            # --- sac_product: escalation = judge-guided authored code (agentic_solve) ---
            if "sac_product" in arms:
                pgen = LLM()
                judge = DiagnosticJudge(pgen, premise=premise)
                t = time.monotonic()
                base_ids, v, _, csc0 = judge_hop0(pgen, judge)
                hops, esc = 0, 0
                if v["verdict"] != "PASS":
                    esc = 1
                    res_p = agentic_solve(session, q, generator=pgen, judge=judge, judge_stop=True,
                                          reranker=session.reranker, embedder=emb,
                                          skill_lookup=skill, memory=per_q,
                                          max_hops=PRODUCT_MAX_HOPS, top_k=20)
                    hops = res_p.get("hops", 0)
                    pids = rrf_ids([base_ids, [str(i) for i in res_p.get("ids") or []]])
                    # FLOOR GUARD (issues.md PROD-1 fix b): a strong base candidate (sigmoid-CE
                    # >= 0.5, i.e. positive cross-encoder logit) cannot be evicted or demoted by
                    # escalation fusion — the vetted baseline's confident hits keep their slots.
                    strong = [x for x, s in zip(base_ids[:10], csc0) if s >= 0.5]
                    pids = strong + [d for d in pids if d not in strong]
                    with mem_lock:                        # cross-query learning: harvest new wins
                        for m in per_q.longterm[n_seeded:]:
                            shared_mem.remember(m.content, kind="skill_win", **(m.meta or {}))
                else:
                    pids = base_ids
                dt = time.monotonic() - t
                r10, a10 = recall_at(gold, pids, 10)
                res["sac_product"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                                      "searches": 1 + hops, "turns": 1 + hops, "escalated": esc,
                                      "in": pgen.usage.input_tokens, "in_cached": pgen.usage.cached_input_tokens,
                                      "out": pgen.usage.output_tokens, "in_uncached_known": True}

            # --- tool_product: SAME hop-0 + judge; escalation = tool-calling episodes,
            #     judge between episodes, same PRODUCT_MAX_HOPS cap ---
            if "tool_product" in arms:
                tgen3 = LLM()
                judge_t = DiagnosticJudge(tgen3, premise=premise)
                t = time.monotonic()
                base_ids, v, subs, csc_t = judge_hop0(tgen3, judge_t)
                pooled, hops, esc, lc_in, lc_out, tsearch, ep_steps = [base_ids], 0, 0, 0, 0, 0, 0
                if v["verdict"] != "PASS":
                    esc = 1
                    for hop in range(1, PRODUCT_MAX_HOPS + 1):
                        hint = (f"\nJudge diagnosis of current results: missing sub-fact "
                                f"{v.get('missing') or '?'} ({v.get('diagnosis') or 'weak coverage'}); "
                                f"suggested technique {v.get('technique') or 'hyde'}; "
                                f"suggested query '{v.get('next_query') or subs[0]}'")
                        tt3 = Tools(session, tgen3, budget, forged_skill=forged_fn)
                        tids3, tm3 = tool_harness(chat, tt3, q + hint, system=TOOL_SYS + seed_guidance)
                        lc_in += tm3["lc_in"]; lc_out += tm3["lc_out"]
                        tsearch += tt3.searches; ep_steps += tm3["steps"]
                        pooled.append([str(i) for i in tids3])
                        hops = hop
                        fused_now = rrf_ids(pooled)
                        docs = {d.id: (d.text or "")[:700] for d in session.store.get(fused_now[:10])}
                        texts = [docs.get(i, "") for i in fused_now[:10]]
                        sub_vecs = np.asarray(emb(subs), dtype=np.float32)
                        cand_vecs = (np.asarray(emb(texts), dtype=np.float32) if texts
                                     else np.zeros((0, sub_vecs.shape[1]), np.float32))
                        cov = djm.coverage_signals(subs, sub_vecs, texts, cand_vecs, session.reranker)
                        csc = djm.candidate_scores(session.reranker, q, texts)
                        cands = [{"id": i, "score": s, "snippet": t_}
                                 for (i, t_), s in zip(zip(fused_now[:10], texts), csc)]
                        v = judge_t.judge(q, subs, cands, cov)
                        if v["verdict"] == "PASS":
                            break
                tp_ids = rrf_ids(pooled)
                strong_t = [i for i, s in zip(base_ids[:10], csc_t) if s >= 0.5]
                tp_ids = strong_t + [d for d in tp_ids if d not in strong_t]
                dt = time.monotonic() - t
                r10, a10 = recall_at(gold, tp_ids, 10)
                res["tool_product"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                                       "searches": 1 + tsearch, "turns": 1 + ep_steps + hops,
                                       "escalated": esc,
                                       "in": lc_in + tgen3.usage.input_tokens,
                                       "in_cached": tgen3.usage.cached_input_tokens,
                                       "out": lc_out + tgen3.usage.output_tokens,
                                       "in_uncached_known": False}

        # explore-SEEDED arms: same harnesses + the corpus knowledge explore/forge produced
        if seed_guidance:
            from experiments.multi_hop_synth_queries.eval_fair import CODE_SYS
            if "tool_explored" in arms:
                tgen2 = LLM()
                tt2 = Tools(session, tgen2, budget, forged_skill=forged_fn)
                t = time.monotonic()
                tids2, tm2 = tool_harness(chat, tt2, q, system=TOOL_SYS + seed_guidance)
                dt = time.monotonic() - t
                r10, a10 = recall_at(gold, tids2, 10)
                res["tool_explored"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                                        "searches": tt2.searches, "turns": tm2["steps"],
                                        "in": tm2["lc_in"] + tgen2.usage.input_tokens,
                                        "in_cached": tgen2.usage.cached_input_tokens,
                                        "out": tm2["lc_out"] + tgen2.usage.output_tokens,
                                        "in_uncached_known": False}
            if "sac_explored" in arms:
                sgen2 = LLM()
                st2 = Tools(session, sgen2, budget, forged_skill=forged_fn)
                t = time.monotonic()
                sids2, sm2 = code_harness(sgen2, st2, q, system=CODE_SYS + seed_guidance)
                dt = time.monotonic() - t
                r10, a10 = recall_at(gold, sids2, 10)
                res["sac_explored"] = {"recall@10": r10, "all@10": a10, "latency_s": round(dt, 3),
                                       "searches": st2.searches, "turns": sm2["steps"],
                                       "in": sgen2.usage.input_tokens,
                                       "in_cached": sgen2.usage.cached_input_tokens,
                                       "out": sgen2.usage.output_tokens, "in_uncached_known": True}

        with lock:
            records.append(res)
            pq_file.write(json.dumps(res) + "\n")
            pq_file.flush()
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
    for a in arms:
        rows_a = [r for r in records if a in r]       # resumed rows may predate the seeded arms
        if not rows_a:
            continue
        agg = {m: sum(r[a][m] for r in rows_a) / len(rows_a) for m in metrics}
        lat = [r[a]["latency_s"] for r in rows_a]
        mean, lo, hi = bootstrap_ci(lat)
        out["arms"][a] = {**{m: round(agg[m], 4) for m in metrics}, "n": len(rows_a),
                          "latency_ci": [round(mean, 2), round(lo, 2), round(hi, 2)]}
        if a in ("sac_product", "tool_product"):
            out["arms"][a]["escalation_rate"] = round(
                sum(r[a].get("escalated", 0) for r in rows_a) / len(rows_a), 3)
        for tg in tags:                           # NOT `tag` — that names the output stem
            sub = [r for r in rows_a if r["tag"] == tg]
            if sub:
                out["by_tag"].setdefault(tg, {})[a] = {
                    m: round(sum(r[a][m] for r in sub) / len(sub), 4) for m in metrics}

    out["config"]["llm_model"] = common.LLM_MODEL
    stem = HERE / f"cost_{corpus}{tag}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2))
    pq_file.close()                                   # rows were flushed incrementally

    print(f"\n===== {corpus} cost (n={len(records)}, budget={budget}) =====")
    print(f"  {'arm':6s} {'r@10':>6s} {'lat_s':>7s} {'turns':>6s} {'srch':>5s} "
          f"{'in_tok':>8s} {'cached':>7s} {'out':>6s}")
    for a in arms:
        if a not in out["arms"]:
            continue
        r = out["arms"][a]
        print(f"  {a:13s} {r['recall@10']:>6.3f} {r['latency_s']:>7.2f} {r['turns']:>6.2f} "
              f"{r['searches']:>5.1f} {int(r['in']):>8d} {int(r['in_cached']):>7d} {int(r['out']):>6d}")
    print(f"\nwrote {stem}.json + perquery ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
