"""The payoff: does the DIAGNOSTIC judge break the multi-hop plateau?

Two arms on the 4-doc HotpotQA queries, both up to `max_hops`, both stopped by the gold ORACLE (so we
isolate the RETRIEVAL progress, not the judge's stop-accuracy — that is measured separately on the eval
set). Hop 1 is identical (the arsenal) for both:

  - blind       : every extra hop re-runs the arsenal on a fresh generic LLM rephrase of the whole
                  question (the "10 blind rewrites" status quo).
  - diagnostic  : every extra hop, the diagnostic judge reads the current fused set (cross-encoder /
                  lexical / score signals per sub-fact), names the WEAKEST-covered sub-fact, and
                  prescribes a TARGETED technique (hyde / fielded / rerank / decompose / prf) + a focused
                  query for THAT sub-fact. We apply it and fuse it into the pool.

We report all_golds@10 solve-rate and average hops. If the diagnostic arm solves more (or as many in
fewer hops), the plateau was a playbook gap, not a retrieval ceiling.

    python -m experiments.deep_judge.run_playbook [n=40] [max_hops=6] [hop=4]
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from search_as_code import primitives as P
from phase1 import common
from phase1.llm import LLM
from experiments.deep_judge.judge_core import INITIAL_PROMPT, parse_verdict, render_example
from experiments.deep_judge.build_evalset import _tok, _snip, _score_signals
from experiments.deep_judge.os_query import author_os_query
from experiments.deep_judge.skill_catalog import SkillLookup

CE_WEAK = 0.0   # a sub-fact whose best candidate scores below this is treated as not-yet-covered

HERE = Path(__file__).parent
DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"


def rrf_ids(lists, k=60):
    s = {}
    for lst in lists:
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(s, key=lambda i: -s[i]), s


def allocate_reserve(sf_lists, k=10):
    """Coverage-guaranteed assembly WITHOUT over-spreading: reserve each sub-fact's single best
    candidate (so every sub-fact — incl. one just fixed by a targeted hop — is represented and can't be
    evicted), then fill the remaining slots by GLOBAL RRF. Robust to over-decomposition: spurious
    sub-facts cost one slot, real golds stay locked at their sub-fact's top rank."""
    reserved, seen = [], set()
    for l in sf_lists:
        if l and l[0] not in seen:
            reserved.append(l[0]); seen.add(l[0])
    fill = [i for i in rrf_ids(sf_lists)[0] if i not in seen]
    return (reserved + fill)[:k]


def sf_arsenal(session, sub, k=30):
    """Per-sub-fact arsenal (hybrid + hyde + fielded), RRF-fused -> one ranked list for that sub-fact."""
    lists = [session.search(sub, top_k=k, mode="hybrid").ids()]
    try:
        lists.append(session.hyde_search(sub, top_k=k).ids())
    except Exception:
        pass
    f = getattr(session.store, "query_fielded", None)
    try:
        lists.append([h.id for h in f(sub, ["title", "text"], top_k=k)] if f
                     else session.search(sub, top_k=k, mode="keyword").ids())
    except Exception:
        pass
    return rrf_ids([l for l in lists if l])[0]


def arsenal_lists(session, subfacts, k=30):
    lists = []
    fielded = getattr(session.store, "query_fielded", None)
    for sub in subfacts:
        lists.append(session.search(sub, top_k=k, mode="hybrid").ids())
        try:
            lists.append(session.hyde_search(sub, top_k=k).ids())
        except Exception:
            pass
        try:
            lists.append([h.id for h in fielded(sub, ["title", "text"], top_k=k)] if fielded
                         else session.search(sub, top_k=k, mode="keyword").ids())
        except Exception:
            pass
    return [l for l in lists if l]


def apply_technique(session, rr, technique, nq, pool_ids, gen=None):
    """Targeted retrieval for the missing sub-fact -> ranked id list."""
    try:
        if technique == "os_query" and gen is not None:   # LLM authors a raw OpenSearch DSL body
            ids, _body, _ok = author_os_query(session.store, gen, nq, top_k=30)
            return ids or session.search(nq, top_k=30, mode="keyword").ids()
        if technique == "arsenal":
            return sf_arsenal(session, nq)
        if technique == "hyde":
            return session.hyde_search(nq, top_k=30).ids()
        if technique == "fielded":
            f = getattr(session.store, "query_fielded", None)
            return [h.id for h in f(nq, ["title", "text"], top_k=30)] if f else \
                session.search(nq, top_k=30, mode="keyword").ids()
        if technique == "prf":
            return session.prf_search(nq, top_k=30).ids()
        if technique == "decompose":
            subs = P.decompose(nq, session._require_generator()) or [nq]
            return rrf_ids([session.search(s, top_k=30, mode="hybrid").ids() for s in subs])[0]
        if technique == "rerank":
            docs = session.store.get(pool_ids[:40])
            texts = [d.text or "" for d in docs]
            order = np.argsort(rr(nq, texts))[::-1] if texts else []
            return [docs[i].id for i in order]
    except Exception:
        pass
    return session.search(nq, top_k=30, mode="hybrid").ids()   # default


def live_example(session, embed, rr, query, subfacts, sub_vecs, fused_ids, scoremap):
    ids = fused_ids[:10]
    docs = {d.id: d for d in session.store.get(ids)}
    texts = [_snip(docs[i].text) if i in docs else "" for i in ids]
    cand_vecs = np.asarray(embed(texts), dtype=np.float32) if texts else np.zeros((0, common.DIM), np.float32)
    ctoks = [_tok(t) for t in texts]
    scores = [scoremap.get(i, 0.0) for i in ids]
    m = max(scores) if scores else 1.0
    cov = []
    for j, sub in enumerate(subfacts):
        sims = (cand_vecs @ sub_vecs[j]) if len(cand_vecs) else np.array([0.0])
        stoks = _tok(sub)
        lex = max((len(stoks & ct) / (len(stoks) or 1) for ct in ctoks), default=0.0)
        ce = rr(sub, texts) if texts else [-10.0]
        cov.append({"subfact": sub[:90], "best_sim": round(float(sims.max()), 3),
                    "lexical_overlap": round(float(lex), 2), "ce_best": round(float(max(ce)), 2)})
    return {"query": query, "subfacts": subfacts,
            "candidates": [{"id": i, "score": round(s / (m or 1.0), 3), "snippet": t}
                           for i, s, t in zip(ids, scores, texts)],
            "coverage": cov, "score_signals": _score_signals(scores)}


def _assemble(arm, sf_lists, k=10):
    if arm == "global":                 # status quo: one global RRF over all sub-fact lists
        return rrf_ids(sf_lists)[0][:k]
    return allocate_reserve(sf_lists, k)  # widen / diagnostic: reserve best-per-sub-fact + RRF fill


def solve(session, embed, rr, gen, judge_prompt, q, gold, max_hops, arm, skill=None):
    gold = set(str(g) for g in gold)
    subfacts = [s for s in (P.decompose(q, gen.as_generator()) or [q]) if s.strip()][:6] or [q]
    sub_vecs = np.asarray(embed(subfacts), dtype=np.float32)
    sf_lists = [sf_arsenal(session, s) for s in subfacts]   # hop 1 — identical for all arms
    trace, calls, captured = [], sum(1 for _ in subfacts) * 3, []
    for hop in range(1, max_hops + 1):
        fused = _assemble(arm, sf_lists)
        got = len(gold & set(fused[:10]))
        solved = got == len(gold)
        trace.append({"hop": hop, "got": got})
        if solved or hop == max_hops:
            break
        if arm == "global":            # blind: re-run whole-query hybrid on a fresh rewrite
            rq = gen.complete(f"Rephrase this multi-hop question to retrieve better, keep all entities:\n{q}")
            sf_lists.append(session.search(rq, top_k=30, mode="hybrid").ids()); calls += 1
        elif arm == "widen":           # non-diagnostic: widen EVERY sub-fact with hyde (no targeting)
            for j, s in enumerate(subfacts):
                try:
                    sf_lists[j] = rrf_ids([sf_lists[j], session.hyde_search(s, top_k=30).ids()])[0]
                except Exception:
                    pass
            calls += len(subfacts)
        else:                          # diagnostic: judge continue/stop, then fix EACH weak sub-fact with
                                       # the technique the skill-lookup picks (incl. authored os_query)
            _, scoremap = rrf_ids(sf_lists)
            ex = live_example(session, embed, rr, q, subfacts, sub_vecs, fused, scoremap)
            v = parse_verdict(gen.complete(render_example(ex), system=judge_prompt))
            if v["verdict"] == "PASS":                 # judge says complete -> stop (autonomous mode)
                if arm == "diagnostic_judgestop":
                    break
            weak = [j for j, c in enumerate(ex["coverage"]) if c["ce_best"] < CE_WEAK]
            if not weak:
                weak = [int(np.argmin([c["ce_best"] for c in ex["coverage"]]))]
            diag = []
            for j in weak:
                # judge's own prescription for the sub-fact it flagged, else skill-lookup on the sub-fact
                if (v["missing"] or "").isdigit() and int(v["missing"]) - 1 == j and v["technique"]:
                    tech, nq = v["technique"], (v["next_query"] or subfacts[j])
                else:
                    tech = skill.suggest(subfacts[j])[0][1] if skill else "hyde"
                    nq = subfacts[j]
                before = set(sf_lists[j][:10])
                if tech == "os_query":                 # author + CAPTURE the raw DSL body for the forge
                    ids, body, ok = author_os_query(session.store, gen, nq, top_k=30)
                    new = ids or session.search(nq, top_k=30, mode="keyword").ids()
                    if body is not None:
                        captured.append({"subfact": nq, "technique": "os_query", "os_body": body})
                else:
                    new = apply_technique(session, rr, tech, nq, fused, gen=gen)
                sf_lists[j] = rrf_ids([sf_lists[j], new])[0]
                calls += 1
                gained = (gold & set(sf_lists[j][:10])) - (gold & before)  # did this technique surface a gold?
                if gained:
                    captured.append({"subfact": nq, "technique": tech, "won": sorted(gained)})
                diag.append({"sf": j + 1, "tech": tech, "won": bool(gained)})
            trace[-1]["diag"] = diag
    return {"solved": int(solved), "hops": hop, "all_recall": got / len(gold), "calls": calls,
            "trace": trace, "captured": captured}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    max_hops = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    hop = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rows = [json.loads(l) for l in (DATA / f"multihop_{hop}docs_queries.jsonl").open()][:n]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128,  # noqa: E731
                                show_progress_bar=False).tolist()
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed,
                          generator=gen.as_generator())
    bp = HERE / "best_prompt_ce_qwen.txt"
    if not bp.exists():
        bp = HERE / "best_prompt_ce_same.txt"
    judge_prompt = bp.read_text() if bp.exists() else INITIAL_PROMPT
    skill = SkillLookup(embed)
    skill.embed_one = lambda t: embed([t])[0]
    print(f"[playbook] {len(rows)} {hop}-hop queries · max_hops={max_hops} · judge_prompt={bp.name if bp.exists() else 'INITIAL'}"
          f" · skill-lookup over {len(skill.cards)} RAG techniques", flush=True)

    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    ARMS = ("global", "widen", "diagnostic")
    agg = {a: [] for a in ARMS}
    recs, captures = [], []
    lock = threading.Lock()

    def one(i, r):
        local = {}
        for arm in ARMS:
            local[arm] = solve(session, embed, rr, gen, judge_prompt, r["query"], r["gold_ids"], max_hops, arm, skill)
        with lock:
            for arm in ARMS:
                agg[arm].append(local[arm])
                recs.append({"i": i, "arm": arm, **{k: local[arm][k] for k in ("solved", "hops", "all_recall", "calls")}})
                for c in local[arm].get("captured", []):
                    captures.append({"i": i, **c})
            n = len(agg[ARMS[0]])
            if n % 5 == 0:
                print(f"  {n}/{len(rows)} · " + " | ".join(
                    f"{a}: solve={np.mean([x['solved'] for x in agg[a]]):.2f} "
                    f"rec={np.mean([x['all_recall'] for x in agg[a]]):.2f}" for a in ARMS), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, i, r) for i, r in enumerate(rows, 1)]))

    out = {}
    for arm in ARMS:
        a = agg[arm]
        out[arm] = {"solve_rate": round(float(np.mean([x["solved"] for x in a])), 3),
                    "avg_recall": round(float(np.mean([x["all_recall"] for x in a])), 3),
                    "avg_hops": round(float(np.mean([x["hops"] for x in a])), 2),
                    "avg_calls": round(float(np.mean([x["calls"] for x in a])), 1), "n": len(a)}
    tag = f"hotpot_{hop}hop"
    (HERE / f"playbook_{tag}.json").write_text(json.dumps(out, indent=2))
    with (HERE / f"playbook_{tag}_perquery.jsonl").open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    with (HERE / f"captures_{tag}.jsonl").open("w") as f:  # authored os_query bodies + winning techniques -> forge
        for c in captures:
            f.write(json.dumps(c) + "\n")
    nq = sum(1 for c in captures if c["technique"] == "os_query" and "os_body" in c)
    print(f"\n[playbook] {hop}-hop (n={out['global']['n']}):  captured {len(captures)} events "
          f"({nq} authored os_query bodies)")
    for arm in ARMS:
        print(f"   {arm:11s} solve={out[arm]['solve_rate']} recall={out[arm]['avg_recall']} "
              f"hops={out[arm]['avg_hops']} calls={out[arm]['avg_calls']}")


if __name__ == "__main__":
    main()
