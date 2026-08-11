"""Can SAC-style composition (FORGED primitives / skills / subagents), driven by the AUTONOMOUS LLM
diagnostic judge (NO oracle), mimic the relevance of raw-query oracle-guided targeting?

Three arms, same decomposition + reserve-assembly, on the SAME queries:
  - raw_oracle : the diagnostic playbook with RAW retrieval techniques (hybrid/hyde/fielded/os_query/…),
                 stopped by the gold ORACLE. This is "the relevance shown with raw queries" — the target.
  - sac_oracle : retrieval action per sub-fact = a FORGED authored primitive (decompose × {hybrid,hyde,
                 fielded} → RRF); weak sub-facts fixed with the forged skill/subagent. Still ORACLE-stopped.
                 Isolates: does SAC composition match raw relevance when stopping is perfect?
  - sac_judge  : identical to sac_oracle but the LLM diagnostic judge decides CONTINUE/STOP (no oracle —
                 "without knowing the ceiling"). Isolates: can the autonomous judge steer the forged
                 primitives to the same realized relevance?

Reports all_golds@10 / recall@10 / hops, plus for sac_judge the judge-stop quality (stopped-when-complete
vs stopped-early). Loads the corpus's forged store (experiments/deep_judge/forge_store_<corpus>/).

    python -m experiments.deep_judge.run_sac_replicate <hotpot|su> [n=30] [hop=4] [workers=6]
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
from search_as_code.harness import AgentMemory, HarnessForge, HarnessStore, SkillRegistry
from experiments.deep_judge.judge_core import INITIAL_PROMPT, parse_verdict, render_example
from experiments.deep_judge.run_playbook import (rrf_ids, allocate_reserve, live_example, apply_technique,
                                                 sf_arsenal, solve, CE_WEAK)
from experiments.deep_judge.skill_catalog import SkillLookup
from experiments.deep_judge.run_forge_playbook import build

HERE = Path(__file__).parent
ARMS = ("raw_oracle", "sac_oracle", "sac_judge")


def forged_retrievers(corpus, registry):
    """The forged authored primitives (ordered like the subagent plan) as callables (subfact,k)->ids."""
    store = HarnessStore(path=str(HERE / f"forge_store_{corpus}"))
    prims = []
    for name in store.subagents.get(f"{corpus}_subfact_agent", None).plan if store.subagents else []:
        sk = registry.get(name)
        if sk is not None:
            prims.append((name, sk))
    if not prims:  # fall back to any registered code primitive
        prims = [(n, registry.get(n)) for n in store.code_primitives if registry.get(n)]
    skill = registry.get(f"{corpus}_diag_arsenal")
    return prims, skill


def sac_solve(session, embed, rr, gen, judge_prompt, prims, diag_skill, q, gold, max_hops, judge_stop):
    """SAC arm: per-sub-fact retrieval via a forged authored primitive; weak sub-facts fixed with the
    forged skill/next primitive; stop by ORACLE (judge_stop=False) or the LLM judge (judge_stop=True)."""
    gold = set(str(g) for g in gold)
    subfacts = [s for s in (P.decompose(q, gen.as_generator()) or [q]) if s.strip()][:6] or [q]
    sub_vecs = np.asarray(embed(subfacts), dtype=np.float32)
    base = prims[0][1] if prims else None

    def retrieve(name_skill, text):
        try:
            out = name_skill.run(session, text, top_k=30)
            return out if isinstance(out, list) else list(out.ids())
        except Exception:
            return sf_arsenal(session, text)

    sf_lists = [retrieve(base, s) if base else sf_arsenal(session, s) for s in subfacts]
    stopped_by, got = None, 0
    for hop in range(1, max_hops + 1):
        fused = allocate_reserve(sf_lists)
        got = len(gold & set(fused[:10]))
        oracle_complete = got == len(gold)
        if hop == max_hops:
            stopped_by = "maxhops"
            break
        if judge_stop:                          # autonomous: the judge decides, no gold
            _, scoremap = rrf_ids(sf_lists)
            ex = live_example(session, embed, rr, q, subfacts, sub_vecs, fused, scoremap)
            v = parse_verdict(gen.complete(render_example(ex), system=judge_prompt))
            if v["verdict"] == "PASS":
                stopped_by = "judge_pass"
                break
            weak = [j for j, c in enumerate(ex["coverage"]) if c["ce_best"] < CE_WEAK] or \
                   [int(np.argmin([c["ce_best"] for c in ex["coverage"]]))]
        else:                                   # oracle stop
            if oracle_complete:
                stopped_by = "oracle"
                break
            _, scoremap = rrf_ids(sf_lists)
            ex = live_example(session, embed, rr, q, subfacts, sub_vecs, fused, scoremap)
            weak = [j for j, c in enumerate(ex["coverage"]) if c["ce_best"] < CE_WEAK] or \
                   [int(np.argmin([c["ce_best"] for c in ex["coverage"]]))]
        # FIX each weak sub-fact with the forged skill (composition), then the next authored primitive
        for j in weak:
            fix = retrieve(diag_skill, subfacts[j]) if diag_skill else []
            if prims:
                fix = rrf_ids([fix, retrieve(prims[min(hop, len(prims) - 1)][1], subfacts[j])])[0]
            sf_lists[j] = rrf_ids([sf_lists[j], fix])[0]
    return {"all_recall": got / len(gold), "solved": int(got == len(gold)), "hops": hop,
            "stopped_by": stopped_by, "stop_correct": int((stopped_by in ("judge_pass", "oracle")) == (got == len(gold)))}


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
    rows = all_rows[:n]

    registry = SkillRegistry(embedder=embed)
    fstore = HarnessStore(path=str(HERE / f"forge_store_{corpus}"))
    HarnessForge(fstore, registry, AgentMemory())     # registers forged primitives + skills into registry
    prims, diag_skill = forged_retrievers(corpus, registry)
    skill = SkillLookup(embed); skill.embed_one = lambda t: embed([t])[0]
    bp = HERE / "best_prompt_ce_same.txt"
    judge_prompt = bp.read_text() if bp.exists() else INITIAL_PROMPT
    print(f"[sac_replicate] corpus={corpus} n={len(rows)} hop={hop} · forged primitives={[p[0] for p in prims]} "
          f"· skill={diag_skill.name if diag_skill else None}", flush=True)

    agg = {a: [] for a in ARMS}
    lock = threading.Lock()

    def one(r):
        loc = {}
        loc["raw_oracle"] = solve(session, embed, rr, gen, judge_prompt, r["query"], r["gold_ids"], hop + 2, "diagnostic", skill)
        loc["sac_oracle"] = sac_solve(session, embed, rr, gen, judge_prompt, prims, diag_skill, r["query"], r["gold_ids"], hop + 2, False)
        loc["sac_judge"] = sac_solve(session, embed, rr, gen, judge_prompt, prims, diag_skill, r["query"], r["gold_ids"], hop + 2, True)
        with lock:
            for a in ARMS:
                agg[a].append(loc[a])
            nd = len(agg["sac_judge"])
            if nd % 5 == 0:
                print("  " + f"{nd}/{len(rows)} · " + " | ".join(
                    f"{a}: rec={np.mean([x['all_recall'] for x in agg[a]]):.2f} "
                    f"solve={np.mean([x['solved'] for x in agg[a]]):.2f}" for a in ARMS), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, r) for r in rows]))

    out = {"corpus": corpus, "n": len(rows), "hop": hop, "arms": {}}
    for a in ARMS:
        A = agg[a]
        out["arms"][a] = {"all_golds@10": round(float(np.mean([x["solved"] for x in A])), 3),
                          "recall@10": round(float(np.mean([x["all_recall"] for x in A])), 3),
                          "avg_hops": round(float(np.mean([x["hops"] for x in A])), 2)}
        if a == "sac_judge":
            out["arms"][a]["stop_correct"] = round(float(np.mean([x["stop_correct"] for x in A])), 3)
    (HERE / f"sac_replicate_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] SAC-replicate (n={len(rows)}, {hop}-hop) =====")
    for a in ARMS:
        e = out["arms"][a]
        extra = f" stop_correct={e.get('stop_correct')}" if a == "sac_judge" else ""
        print(f"  {a:11s} all_golds@10={e['all_golds@10']} recall@10={e['recall@10']} hops={e['avg_hops']}{extra}")
    print(f"[sac_replicate] wrote sac_replicate_{corpus}.json")


if __name__ == "__main__":
    main()
