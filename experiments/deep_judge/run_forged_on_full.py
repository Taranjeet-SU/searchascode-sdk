"""Stage 7 at scale — run the FORGED explored primitive on the FULL corpus, vs dense baseline.

Loads the primitive forged by run_explore_pipeline (forge_store_<corpus>_explored/), runs it on ALL gold
queries, and reports recall@10/@20/all-golds@10 next to a plain dense baseline on the same queries.

    python -m experiments.deep_judge.run_forged_on_full <corpus> [workers=8]
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

HERE = Path(__file__).parent


def _recall(gold, ids):
    g = set(map(str, gold))
    r10 = len(g & set(map(str, ids[:10]))) / len(g)
    r20 = len(g & set(map(str, ids[:20]))) / len(g)
    return r10, r20, int(g <= set(map(str, ids[:10])))


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "browsecomp"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    if corpus == "browsecomp":
        store = sac.connect("opensearch", index="browsecomp", dim=common.DIM, hosts=[common.OS_HOST],
                            text_field="text", vector_field="vector")
        from experiments.browsecomp import bc_common
        g, q = bc_common.load_golds(), bc_common.load_queries()
        rows = [{"query": q[k], "gold_ids": g[k]} for k in q if k in g]
    else:
        from experiments.deep_judge.run_forge_playbook import build
        store, rows = build(corpus, embed, gen)
    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)

    reg = SkillRegistry(embedder=embed)
    fstore = HarnessStore(path=str(HERE / f"forge_store_{corpus}_explored"))
    HarnessForge(fstore, reg, AgentMemory())
    name = f"{corpus}_explored_primitive"
    prim = reg.get(name)
    print(f"[full] corpus={corpus} · {len(rows)} gold queries · forged primitive={name} "
          f"present={prim is not None} · structure_rules={fstore.learnings[-1:] if fstore.learnings else '?'}", flush=True)

    agg = {"forged": {"r10": [], "r20": [], "all": []}, "dense": {"r10": [], "r20": [], "all": []}}
    lock = threading.Lock()

    def one(r):
        d = session.search(r["query"], top_k=20, mode="dense").ids()
        f = []
        if prim is not None:
            try:
                f = prim.run(session, r["query"], top_k=20)
            except Exception:
                f = []
        with lock:
            for arm, ids in (("forged", f), ("dense", d)):
                a, b, c = _recall(r["gold_ids"], ids)
                agg[arm]["r10"].append(a); agg[arm]["r20"].append(b); agg[arm]["all"].append(c)
            n = len(agg["dense"]["r10"])
            if n % 50 == 0:
                print(f"  {n}/{len(rows)} · forged r10={np.mean(agg['forged']['r10']):.3f} "
                      f"dense r10={np.mean(agg['dense']['r10']):.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, r) for r in rows]))

    out = {"corpus": corpus, "n_full": len(rows), "forged_primitive": name, "present": prim is not None,
           "arms": {arm: {"recall@10": round(float(np.mean(v["r10"])), 4),
                          "recall@20": round(float(np.mean(v["r20"])), 4),
                          "all_golds@10": round(float(np.mean(v["all"])), 4)} for arm, v in agg.items()}}
    (HERE / f"explore_full_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] FULL-DATA (n={len(rows)}) with forged primitive =====")
    for arm in ("dense", "forged"):
        e = out["arms"][arm]
        print(f"  {arm:7s} recall@10={e['recall@10']} recall@20={e['recall@20']} all_golds@10={e['all_golds@10']}")


if __name__ == "__main__":
    main()
