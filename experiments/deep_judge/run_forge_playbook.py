"""Phase C — run the diagnostic-judge playbook through explore+forge on a corpus, then FORGE new
primitives / skills / subagents from the OpenSearch queries the loop discovered.

Per corpus (hotpot | su), for n 4-hop queries:
  1. RUN the loop twice per query: `global` (blind rewrite baseline) and `diagnostic` (LLM-as-judge picks
     the next hop; each weak sub-fact is fixed by the technique the RAG-Techniques skill-lookup returns —
     hybrid / hyde / fielded / rerank / decompose / prf / LLM-authored os_query). Record solve/recall/hops
     and CAPTURE every winning (sub-fact, technique, authored os_query body).
  2. STORE wins to cross-query AgentMemory.
  3. FORGE from the captured OpenSearch queries (the user's ask): the LLM AUTHORS free-form code
     primitives over the full SDK (session.search hybrid/dense/keyword + hyde_search + query_fielded +
     fuse/RRF), each VALIDATED on a held query with gold (forge.author_code_primitive); then a composed
     skill + a subagent + prompt rules. All persisted to forge_store_<corpus>/ and registered for reuse
     (online learning: later queries can select the forged primitive via the skill-lookup).

    python -m experiments.deep_judge.run_forge_playbook <hotpot|su> [n=30] [hop=4] [workers=6]

SU uses the user-authorized su_docs corpus. Nothing Altera-related is touched.
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
from phase1 import common
from phase1.llm import LLM
from search_as_code.harness import AgentMemory, HarnessForge, HarnessStore, SkillRegistry
from search_as_code.harness.forge import author_code_primitive
from experiments.deep_judge.judge_core import INITIAL_PROMPT
from experiments.deep_judge.skill_catalog import SkillLookup, catalog_summary
from experiments.deep_judge.run_playbook import solve

HERE = Path(__file__).parent
HOTPOT_DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
SU_DATA = Path(__file__).parents[1] / "su_multihop" / "data"
ARMS = ("global", "diagnostic")


def build(corpus, embed, gen):
    if corpus == "hotpot":
        store = sac.connect("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                            text_field="text", vector_field="vector")
        rows = [json.loads(l) for l in (HOTPOT_DATA / "multihop_4docs_queries.jsonl").open()]
    elif corpus == "su":
        import pandas as pd
        df = pd.read_csv(Path.home() / "scripts" / "data" / "su_docs_2.csv")
        docs = []
        for _, r in df.iterrows():
            c = r.get("content")
            if pd.isna(c) or not str(c).strip():
                continue
            t = "" if pd.isna(r.get("title")) else str(r.get("title"))
            docs.append({"id": str(r["id"]), "text": (t + ". " + str(c)).strip()})
        loader = sac.Session("memory", dim=common.DIM, embedder=embed, generator=gen.as_generator())
        loader.add(docs)
        store = loader.store
        rows = [json.loads(l) for l in (SU_DATA / "su_multihop_4docs.jsonl").open()]
    elif corpus == "browsecomp":
        # BrowseComp-Plus: 100K-doc FastMemoryStore with precomputed gte-base vectors; real qrels golds.
        # Memory store -> no raw _search/query_fielded (os_query/fielded degrade to keyword, as designed).
        import numpy as np
        from experiments.browsecomp import bc_common
        from search_as_code.types import Document
        vecs = np.load(bc_common.VECS_NPY)
        ids = json.loads(bc_common.IDS_JSON.read_text())
        texts = {}
        for line in bc_common.TEXTS_JSONL.open():
            row = json.loads(line)
            texts[row["id"]] = row["text"]
        store = bc_common.FastMemoryStore()
        store.upsert([Document(id=str(i), text=texts.get(str(i), ""), vector=vecs[k].tolist())
                      for k, i in enumerate(ids)])
        golds, queries = bc_common.load_golds(), bc_common.load_queries()
        rows = [{"query": queries[q], "gold_ids": golds[q]} for q in queries
                if q in golds and queries.get(q)]
    else:
        raise SystemExit(f"unknown corpus {corpus}")
    return store, rows


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "hotpot"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    hop = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128, show_progress_bar=False).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    store, all_rows = build(corpus, embed, gen)
    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)
    rows, held = all_rows[:n], all_rows[n:n + 8]     # held-out queries (with gold) to validate authored primitives

    skill = SkillLookup(embed)
    skill.embed_one = lambda t: embed([t])[0]
    memory = AgentMemory(path=str(HERE / f"forge_{corpus}_memory.jsonl"), embedder=embed)
    registry = SkillRegistry(embedder=embed)
    fstore = HarnessStore(path=str(HERE / f"forge_store_{corpus}"))
    forge = HarnessForge(fstore, registry, memory)
    bp = HERE / "best_prompt_ce_same.txt"
    judge_prompt = bp.read_text() if bp.exists() else INITIAL_PROMPT
    print(f"[forge_playbook] corpus={corpus} n={len(rows)} hop={hop} · judge={bp.name if bp.exists() else 'INITIAL'} "
          f"· skill-lookup over {len(skill.cards)} RAG techniques · store={store.__class__.__name__}", flush=True)

    agg = {a: [] for a in ARMS}
    captures = []
    lock = threading.Lock()

    def one(r):
        loc = {a: solve(session, embed, rr, gen, judge_prompt, r["query"], r["gold_ids"], hop + 2, a, skill)
               for a in ARMS}
        with lock:
            for a in ARMS:
                agg[a].append(loc[a])
            for c in loc["diagnostic"].get("captured", []):
                captures.append(c)
            if loc["diagnostic"].get("captured"):
                wins = "; ".join(f"{c['technique']}:'{c.get('subfact','')[:40]}'"
                                 for c in loc["diagnostic"]["captured"] if c.get("won"))
                if wins:
                    memory.remember(f"solved sub-facts via {wins}", kind="skill_win",
                                    winning=loc["diagnostic"]["captured"])
            n_done = len(agg["diagnostic"])
            if n_done % 5 == 0:
                print("  " + f"{n_done}/{len(rows)} · " + " | ".join(
                    f"{a}: solve={np.mean([x['solved'] for x in agg[a]]):.2f} "
                    f"rec={np.mean([x['all_recall'] for x in agg[a]]):.2f}" for a in ARMS), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, r) for r in rows]))

    results = {a: {"solve_rate": round(float(np.mean([x["solved"] for x in agg[a]])), 3),
                   "avg_recall": round(float(np.mean([x["all_recall"] for x in agg[a]])), 3),
                   "avg_hops": round(float(np.mean([x["hops"] for x in agg[a]])), 2),
                   "avg_calls": round(float(np.mean([x["calls"] for x in agg[a]])), 1),
                   "n": len(agg[a])} for a in ARMS}
    with (HERE / f"forge_captures_{corpus}.jsonl").open("w") as f:
        for c in captures:
            f.write(json.dumps(c) + "\n")
    os_bodies = [c for c in captures if c.get("technique") == "os_query" and "os_body" in c]
    print(f"\n[run] captured {len(captures)} winning events · {len(os_bodies)} authored os_query bodies", flush=True)

    # ---- FORGE: author free-form code primitives from the captured OpenSearch queries ----
    patterns = "\n".join(
        f"- {c['technique']} on '{c.get('subfact','')[:70]}'" + (f"  DSL={json.dumps(c['os_body'])[:120]}" if c.get("os_body") else "")
        for c in captures[:40]) or "- decompose + hybrid + hyde per sub-fact, RRF-fused"
    authored = []
    for i, hq in enumerate(held[:3], 1):
        name = f"{corpus}_authored_{i}"
        print(f"\n[forge] authoring primitive {name} (validate on held query, {len(hq['gold_ids'])} golds)...", flush=True)
        code, ok = author_code_primitive(gen, patterns, forge, session, hq["query"], hq["gold_ids"],
                                         name=name, tries=3)
        authored.append({"name": name, "accepted": ok, "code": code})
        if ok:
            break_note = registry.get(name)
            print(f"[forge] ACCEPTED {name}; registered={break_note is not None}", flush=True)

    # composed skill + subagent + prompt rules (from the winning technique mix)
    top_techs = winning_retrievers(captures)
    forge.create_skill(f"{corpus}_diag_arsenal", "multi-hop: fix each weak sub-fact with the winning "
                       "technique mix, RRF-fused", top_techs, combine="fuse", origin="forge_playbook")
    forge.create_subagent(f"{corpus}_subfact_agent", "solve one weak sub-fact using authored primitive "
                          "then arsenal", plan=[a["name"] for a in authored if a["accepted"]] or [f"{corpus}_diag_arsenal"])
    forge.refine_prompt(f"[{corpus}] for a generically-DESCRIBED entity use hyde; for a NAMED entity use "
                        "fielded/os_query; fuse per-sub-fact with RRF and reserve one slot per sub-fact")
    fstore.save()

    out = {"corpus": corpus, "n": len(rows), "hop": hop, "results": results,
           "captured_events": len(captures), "authored_os_queries": len(os_bodies),
           "artifacts": {
               "code_primitives": list(fstore.code_primitives.keys()),
               "skills": list(fstore.skills.keys()),
               "subagents": list(fstore.subagents.keys()),
               "learned_rules": fstore.learnings,
               "authored": authored,
           }}
    (HERE / f"forge_playbook_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] {hop}-hop (n={len(rows)}) =====")
    for a in ARMS:
        print(f"  {a:11s} solve={results[a]['solve_rate']} recall={results[a]['avg_recall']} "
              f"hops={results[a]['avg_hops']} calls={results[a]['avg_calls']}")
    print(f"  FORGED -> code_primitives={list(fstore.code_primitives.keys())} "
          f"skills={list(fstore.skills.keys())} subagents={list(fstore.subagents.keys())}")
    print(f"[forge_playbook] wrote forge_playbook_{corpus}.json + forge_store_{corpus}/")


def _counts(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return sorted(d.items(), key=lambda kv: -kv[1])


# retriever labels a composed LearnedSkill can actually execute (via forge._ALIAS); non-composable
# techniques are normalised to the nearest composable one, and empty/none are dropped.
_COMPOSABLE = {"dense", "keyword", "hybrid", "hyde", "prf", "exact", "decompose", "rerank", "mmr"}
_NORMALISE = {"os_query": "keyword", "fielded": "keyword", "arsenal": "hybrid", "hybrid": "hybrid"}


def winning_retrievers(captures, k: int = 3):
    """Top-k composable retrievers among the techniques that actually WON a gold — cleaned of the
    labelling artifacts (drop 'none'/'', map os_query/fielded->keyword since a composed skill can't run
    a raw DSL body)."""
    techs = []
    for c in captures:
        if not c.get("won"):
            continue
        t = _NORMALISE.get(c.get("technique", ""), c.get("technique", ""))
        if t and t in _COMPOSABLE:
            techs.append(t)
    top = [t for t, _ in _counts(techs)][:k]
    for d in ("decompose", "hyde", "hybrid"):     # ensure a sane, composable default set
        if d not in top:
            top.append(d)
    return top[:k]


if __name__ == "__main__":
    main()
