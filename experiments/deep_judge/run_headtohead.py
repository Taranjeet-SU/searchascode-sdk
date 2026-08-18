"""Apples-to-apples: are the NEW pieces (diagnostic judge + forged primitives) an improvement over PAST
experiments, on the SAME queries?

For n identical 4-hop queries per corpus, every arm runs on the same fresh session:
  PAST baselines:
    - dense        : one dense search (the cheap floor).
    - oneshot_sac  : phase1.agents.run_sac(deep=False, max_retries=0) — the single-program code-mode agent.
    - deep_sac     : phase1.agents.run_sac(deep=True, monotone=True) — the ensemble+consensus deep agent.
  NEW (this exercise):
    - diagnostic   : harness.diagnostic_solve (raw arsenal), the LLM judge decides stop (autonomous).
    - forged_sac   : harness.diagnostic_solve(forged=<forged primitives>), judge decides stop.
  Reference:
    - diagnostic_oracle : diagnostic_solve, oracle-stop (retrieval ceiling; not autonomous).

All autonomous arms are judge/agent-driven (no gold), so deep_sac vs diagnostic vs forged_sac is a fair
"new vs past" head-to-head. Reports all_golds@10 / recall@10 / hops on identical queries.

    python -m experiments.deep_judge.run_headtohead <hotpot|su> [n=30] [workers=4]
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
from phase1 import agents, common
from phase1.llm import LLM
from search_as_code.harness import diagnostic_solve
from experiments.deep_judge.run_forge_playbook import build
from experiments.deep_judge.run_sac_replicate import forged_retrievers

HERE = Path(__file__).parent
# BrowseComp is a ~6%-recall needle-in-100K-haystack benchmark: all_golds@10 ~ 0, so report recall@20
# too and retrieve a deeper pool; deep_sac is dropped there (memory-store ensemble is intractable at scale).
ARMS_DEFAULT = ("dense", "oneshot_sac", "deep_sac", "diagnostic", "forged_sac", "diagnostic_oracle")
ARMS_BC = ("dense", "oneshot_sac", "diagnostic", "diagnostic_oracle")


def _recalls(gold, ids):
    g = set(map(str, gold))
    r10 = len(g & set(map(str, ids[:10]))) / len(g)
    r20 = len(g & set(map(str, ids[:20]))) / len(g)
    return r10, r20, int(g <= set(map(str, ids[:10])))


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "hotpot"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128, show_progress_bar=False).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    store, all_rows = build(corpus, embed, gen)
    rows = all_rows[:n]
    from search_as_code.harness import SkillRegistry, HarnessStore, HarnessForge, AgentMemory
    reg = SkillRegistry(embedder=embed)
    HarnessForge(HarnessStore(path=str(HERE / f"forge_store_{corpus}")), reg, AgentMemory())
    prims, _skill = forged_retrievers(corpus, reg)
    forged = [p[1].run for p in prims]
    chat = agents.lc_chat()
    bc = corpus == "browsecomp"
    ARMS = ARMS_BC if bc else ARMS_DEFAULT       # drop deep_sac/forged on the intractable memory corpus
    K = 20                                        # retrieve a deep pool so we can report recall@10 AND @20
    print(f"[h2h] corpus={corpus} n={len(rows)} · arms={ARMS} · forged={[p[0] for p in prims]}", flush=True)

    agg = {a: {"r10": [], "r20": [], "all": [], "hops": []} for a in ARMS}
    lock = threading.Lock()

    def one(r):
        q, gold = r["query"], r["gold_ids"]
        res = {}
        session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)
        if "dense" in ARMS:
            try:
                res["dense"] = (session.search(q, top_k=K, mode="dense").ids(), 1)
            except Exception:
                res["dense"] = ([], 1)
        for arm, kw in (("oneshot_sac", dict(deep=False, max_retries=0)),
                        ("deep_sac", dict(deep=True, max_retries=3, monotone=True))):
            if arm not in ARMS:
                continue
            try:
                o = agents.run_sac(session, q, chat=chat, judge_chat=chat, k=K, **kw)
                res[arm] = (o["ids"], o["hops"])
            except Exception as e:
                res[arm] = ([], 0); print(f"  [ERR {arm}] {e}", flush=True)
        for arm, kw in (("diagnostic", dict(judge_stop=True)),
                        ("forged_sac", dict(judge_stop=True, forged=forged)),
                        ("diagnostic_oracle", dict(judge_stop=False))):
            if arm not in ARMS:
                continue
            try:
                o = diagnostic_solve(session, q, gold=gold, generator=gen, reranker=rr, embedder=embed,
                                     max_hops=6, top_k=K, **kw)
                res[arm] = (o["ids"], o["hops"])
            except Exception as e:
                res[arm] = ([], 0); print(f"  [ERR {arm}] {e}", flush=True)
        with lock:
            for a in ARMS:
                r10, r20, al = _recalls(gold, res.get(a, ([], 0))[0])
                agg[a]["r10"].append(r10); agg[a]["r20"].append(r20)
                agg[a]["all"].append(al); agg[a]["hops"].append(res.get(a, ([], 0))[1])
            nd = len(agg[ARMS[0]]["r10"])
            if nd % 5 == 0:
                print("  " + f"{nd}/{len(rows)} · " + " | ".join(
                    f"{a}:r10={np.mean(agg[a]['r10']):.3f}" for a in ARMS), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, r) for r in rows]))

    out = {"corpus": corpus, "n": len(rows), "arms": {a: {
        "recall@10": round(float(np.mean(agg[a]["r10"])), 3),
        "recall@20": round(float(np.mean(agg[a]["r20"])), 3),
        "all_golds@10": round(float(np.mean(agg[a]["all"])), 3),
        "avg_hops": round(float(np.mean(agg[a]["hops"])), 2)} for a in ARMS}}
    (HERE / f"headtohead_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] head-to-head (n={len(rows)}, SAME queries) =====")
    for a in ARMS:
        e = out["arms"][a]
        print(f"  {a:18s} recall@10={e['recall@10']}  recall@20={e['recall@20']}  "
              f"all_golds@10={e['all_golds@10']}  hops={e['avg_hops']}")


if __name__ == "__main__":
    main()
