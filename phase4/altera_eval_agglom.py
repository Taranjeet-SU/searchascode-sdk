"""Compare closed_book vs sac_agent (critic loop) vs sac_agglom (agglomerative
multi-evidence: wide atomic+rephrased queries -> per-sub partial answers -> agglomerate).

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_eval_agglom --n 195 --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_agent import agentic_sac
from phase4.altera_agglom import agglomerative_sac
from phase4.altera_eval import ANS_SYS, ctx_text, judge, judge_cite, SHEET


def process(r, gen, reranker, rr_lock, k, max_hops):
    q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
    o = {"ans": {}, "cite": {}, "natomic": 0}
    if r.get("Vendor Answer"):
        o["vendor"] = int(judge(gen, q, exp, r["Vendor Answer"]))
        o["vendor_sheet"] = int(r.get("Verdict", "").upper() == "PASS")
    o["ans"]["closed_book"] = int(judge(gen, q, exp,
                                        gen.complete(f"Question: {q}\n\nAnswer:", system=ANS_SYS).strip()))
    dg, final, natom = agglomerative_sac(gen, reranker, rr_lock, q, k=4)
    o["ans"]["sac_agglom"] = int(judge(gen, q, exp, final))
    o["cite"]["sac_agglom"] = judge_cite(gen, q, cites, dg, k)
    o["natomic"] = natom
    return o


def main(n=195, k=6, workers=16, max_hops=3):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")][:n]
    gen = LLM(); reranker = sac.QwenReranker(); reranker("warm", ["a", "b"]); altera.embedder()
    rr_lock = threading.Lock()
    arms = ["closed_book", "sac_agglom"]
    ans = {a: 0 for a in arms}; cite = {a: 0 for a in arms}; cite_n = {a: 0 for a in arms}
    natom = []; vp = vs = done = 0
    print(f"[agglom] {len(rows)} questions, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, r, gen, reranker, rr_lock, k, max_hops) for r in rows]
        for f in as_completed(futs):
            o = f.result(); done += 1
            vp += o.get("vendor", 0); vs += o.get("vendor_sheet", 0); natom.append(o["natomic"])
            for a in arms:
                ans[a] += o["ans"][a]
                if a in o["cite"] and o["cite"][a] is not None:
                    cite_n[a] += 1; cite[a] += o["cite"][a]
            if done % 5 == 0:
                la = " ".join(f"{a}={ans[a]/done:.2f}" for a in arms)
                print(f"[agglom] {done}/{len(rows)}  ANS[{la}]  vendor={vp/done:.2f}  "
                      f"CITE[agglom={cite['sac_agglom']/max(1,cite_n['sac_agglom']):.2f}]  "
                      f"avg_atomic={np.mean(natom):.1f}", flush=True)
    N = len(rows)
    print(f"\n===== Altera AGGLOMERATIVE eval (n={N}) =====")
    print(f"  vendor: our_judge={vp/N:.3f}  sheet={vs/N:.3f}  avg_atomic_subqs={np.mean(natom):.2f}")
    for a in arms:
        cp = f"{cite[a]/cite_n[a]:.3f}" if cite_n[a] else "n/a"
        print(f"  {a:11s} answer={ans[a]/N:.3f}  citation={cp}")
    print(f"  llm cost ${gen.usage.cost_usd:.4f}")
    (Path(common.REPO) / "phase4" / "runs" / "altera_agglom.json").write_text(json.dumps(
        {"n": N, "avg_atomic": float(np.mean(natom)), "vendor_our": vp / N, "vendor_sheet": vs / N,
         "answer": {a: ans[a] / N for a in arms},
         "citation": {a: (cite[a] / cite_n[a] if cite_n[a] else None) for a in arms}}, indent=2))
    print("[agglom] saved runs/altera_agglom.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=195); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--max-hops", type=int, default=3)
    a = ap.parse_args(); main(a.n, a.k, a.workers, a.max_hops)
