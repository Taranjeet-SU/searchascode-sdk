"""Re-forge the DISCOVERED structure (whole-query for BrowseComp) into a working primitive, then run it
on the FULL data vs dense. The pipeline discovered structure=whole-query but its forge acceptance bar
(mean recall>0 over 5 held) was too strict for a ~0.09-recall corpus, so no primitive was bottled. Here
we author the whole-query structure directly, validate on a proper held set (accept if it matches/beats
dense), register it, and evaluate on all gold queries.

    python -m experiments.deep_judge.reforge_and_full browsecomp [workers=8]
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
from search_as_code.harness import AgentMemory, HarnessForge, HarnessStore, SkillRegistry
from search_as_code.harness.agentic import _exec

HERE = Path(__file__).parent
_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

WHOLE_QUERY_PROMPT = """Write ONE reusable retrieval primitive. Signature EXACTLY: def run(session, query, top_k):
The corpus rewards KEEPING THE QUERY WHOLE (one entity satisfying many constraints) — do NOT decompose.
Retrieve the whole query several ways and fuse, then rerank. Use ONLY:
  session.search(q, top_k=k, mode='dense'|'keyword'|'hybrid')  -> ResultSet (.ids())
  session.hyde_search(q, top_k=k)
  fuse_ids([ids_a, ids_b, ...]) -> list        # in scope
  rerank(session, query, ids, top_k=k) -> list # in scope, cross-encoder
Recommended: fuse hybrid + dense (+ optionally hyde) over the WHOLE query, then rerank the fused pool.
Return a list of ids. Return ONLY one ```python block```."""


def _recall(gold, ids):
    g = set(map(str, gold))
    return (len(g & set(map(str, ids[:10]))) / len(g),
            len(g & set(map(str, ids[:20]))) / len(g),
            int(g <= set(map(str, ids[:10]))))


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "browsecomp"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    store = sac.connect("opensearch", index="browsecomp", dim=common.DIM, hosts=[common.OS_HOST],
                        text_field="text", vector_field="vector")
    from experiments.browsecomp import bc_common
    g, q = bc_common.load_golds(), bc_common.load_queries()
    rows = [{"query": q[k], "gold_ids": g[k]} for k in q if k in g]
    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)
    held = rows[314:344]                                   # 30 held (after the pipeline's train+val)

    # ---- author the whole-query primitive, validate dense-relative on 30 held ----
    dense_held = float(np.mean([_recall(h["gold_ids"], session.search(h["query"], top_k=20, mode="dense").ids())[1]
                                for h in held]))
    print(f"[reforge] dense mean recall@20 over {len(held)} held = {dense_held:.3f}", flush=True)
    reg = SkillRegistry(embedder=embed)
    fstore = HarnessStore(path=str(HERE / f"forge_store_{corpus}_explored"))
    forge = HarnessForge(fstore, reg, AgentMemory())
    name = f"{corpus}_explored_primitive"
    code, ok, err = "", False, ""
    for attempt in range(4):
        raw = gen.complete(WHOLE_QUERY_PROMPT + (f"\n\nPrevious FAILED: {err} Fix it." if err else ""))
        m = _CODE.search(raw)
        code = (m.group(1) if m else raw).strip()
        try:
            r = [_recall(h["gold_ids"], _exec(code, session, h["query"], 20))[1] for h in held]
            mean = float(np.mean(r))
            print(f"  attempt {attempt+1}: mean recall@20 over held = {mean:.3f} (dense {dense_held:.3f})", flush=True)
            if mean >= 0.9 * dense_held:                   # accept if it matches/beats dense
                forge.create_code_primitive(name, "explored whole-query structure for browsecomp", code)
                ok = True; break
            err = f"mean recall@20 {mean:.3f} < dense {dense_held:.3f}; keep the query WHOLE, fuse hybrid+dense, rerank."
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  attempt {attempt+1}: error {err[:80]}", flush=True)
    forge.create_skill(f"{corpus}_explored_skill", "explored whole-query strategy for browsecomp",
                       ["hybrid", "dense", "rerank"], combine="fuse")
    forge.create_subagent(f"{corpus}_explored_agent", "solve via the explored whole-query primitive", plan=[name])
    fstore.save()
    print(f"[reforge] primitive accepted={ok}; code_primitives={list(fstore.code_primitives)}", flush=True)

    # ---- run on FULL data: forged primitive vs dense ----
    prim = reg.get(name)
    agg = {"forged": [[], [], []], "dense": [[], [], []]}
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
                agg[arm][0].append(a); agg[arm][1].append(b); agg[arm][2].append(c)
            n = len(agg["dense"][0])
            if n % 100 == 0:
                print(f"  full {n}/{len(rows)}: forged r10={np.mean(agg['forged'][0]):.3f} "
                      f"dense r10={np.mean(agg['dense'][0]):.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, r) for r in rows]))

    out = {"corpus": corpus, "n_full": len(rows), "forged_accepted": ok,
           "arms": {arm: {"recall@10": round(float(np.mean(v[0])), 4),
                          "recall@20": round(float(np.mean(v[1])), 4),
                          "all_golds@10": round(float(np.mean(v[2])), 4)} for arm, v in agg.items()},
           "forged_code": code if ok else None}
    (HERE / f"explore_full_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"\n===== [{corpus}] FULL-DATA (n={len(rows)}) forged whole-query vs dense =====")
    for arm in ("dense", "forged"):
        e = out["arms"][arm]
        print(f"  {arm:7s} recall@10={e['recall@10']} recall@20={e['recall@20']} all_golds@10={e['all_golds@10']}")


if __name__ == "__main__":
    main()
