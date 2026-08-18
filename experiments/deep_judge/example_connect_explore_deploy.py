"""End-to-end example: connect to OpenSearch via SAC → explore+forge → deploy.

Uses only the standard package API (`search_as_code` + `search_as_code.harness`). Three stages:
  1) CONNECT   — open an OpenSearch-backed SAC Session (dense + keyword + hybrid + HyDE + fielded).
  2) EXPLORE+FORGE — on TRAIN queries (gold known), the diagnostic judge drives targeted hops; from the
     winning OpenSearch queries the LLM AUTHORS reusable code primitives (validated on gold) and the forge
     persists them + a composed skill + a subagent.
  3) DEPLOY    — a fresh process loads the persisted forge store and answers NEW queries autonomously:
     retrieval runs through the forged primitives, the diagnostic judge decides when to stop (no gold).

    python -m experiments.deep_judge.example_connect_explore_deploy
"""
from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from search_as_code.harness import (
    AgentMemory, HarnessForge, HarnessStore, SkillRegistry, SkillLookup,
    diagnostic_solve,
)
from search_as_code.harness.forge import author_code_primitive

STORE_PATH = "my_forge_store"       # where the forged primitives/skills/subagents are persisted


# ─── 1) CONNECT to OpenSearch via SAC ────────────────────────────────────────────────────────────
def connect():
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda texts: em.encode(list(texts), normalize_embeddings=True).tolist()   # noqa: E731
    gen = LLM()                                    # gpt-4.1-mini; has .complete() and .as_generator()
    reranker = sac.CrossEncoderReranker()          # cross-encoder — the judge's primary coverage signal

    store = sac.connect("opensearch", index="hotpotqa", dim=common.DIM,
                        hosts=[common.OS_HOST], text_field="text", vector_field="vector")
    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=reranker)
    return session, embed, reranker, gen


# ─── 2) EXPLORE + FORGE — discover winning OpenSearch queries, bottle them as primitives ──────────
def explore_and_forge(session, embed, reranker, gen, train):
    registry = SkillRegistry(embedder=embed)                       # holds built-in + forged skills
    fstore = HarnessStore(path=STORE_PATH)                         # persisted self-modifiable state
    forge = HarnessForge(fstore, registry, AgentMemory(path="mem.jsonl"))
    skill_lookup = SkillLookup(embed)                             # RAG_Techniques catalog (routes diagnoses)

    # Explore: the diagnostic judge + skill-lookup solve each TRAIN query with targeted hops.
    for q in train:
        res = diagnostic_solve(session, q["query"], gold=q["gold_ids"], generator=gen,
                               reranker=reranker, embedder=embed, skill_lookup=skill_lookup, max_hops=6)
        print(f"[explore] solved={res['solved']} recall={res['all_recall']:.2f} hops={res['hops']} "
              f"| {q['query'][:60]}")

    # Forge: the LLM AUTHORS a reusable retrieval primitive (composes hybrid/HyDE/fielded → RRF),
    # validated on a held query with gold; the forge persists it + a composed skill + a subagent.
    held = train[-1]
    patterns = "decompose per sub-fact; hyde for described entities; fielded for named entities; RRF-fuse"
    code, accepted = author_code_primitive(gen, patterns, forge, session,
                                           held["query"], held["gold_ids"], name="authored_multihop")
    print(f"[forge] authored primitive accepted={accepted}")
    forge.create_skill("diag_arsenal", "multi-hop: fuse the winning technique mix",
                       ["decompose", "hyde", "rerank"], combine="fuse")
    forge.create_subagent("subfact_agent", "solve a weak sub-fact via the authored primitive",
                          plan=["authored_multihop"])
    fstore.save()
    print(f"[forge] persisted -> {STORE_PATH}/ "
          f"(code_primitives={list(fstore.code_primitives)}, skills={list(fstore.skills)}, "
          f"subagents={list(fstore.subagents)})")


# ─── 3) DEPLOY — load the forge store; answer NEW queries autonomously (judge decides stop) ───────
def deploy(session, embed, reranker, gen, live_queries):
    registry = SkillRegistry(embedder=embed)
    fstore = HarnessStore(path=STORE_PATH)                        # reload persisted forge state
    HarnessForge(fstore, registry, AgentMemory())                # re-registers the forged primitives
    forged = [registry.get(n).run for n in fstore.code_primitives if registry.get(n)]

    for query in live_queries:
        res = diagnostic_solve(session, query, generator=gen, reranker=reranker, embedder=embed,
                               forged=forged, judge_stop=True, max_hops=6)   # gold=None -> judge stops
        print(f"[deploy] '{query[:50]}' -> {len(res['ids'])} ids "
              f"(stopped_by={res['stopped_by']}, hops={res['hops']}) top: {res['ids'][:3]}")


if __name__ == "__main__":
    import json
    from pathlib import Path
    data = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data" / "multihop_4docs_queries.jsonl"
    rows = [json.loads(l) for l in data.open()][:6]

    session, embed, reranker, gen = connect()
    explore_and_forge(session, embed, reranker, gen, train=rows[:5])
    deploy(session, embed, reranker, gen, live_queries=[rows[5]["query"]])
