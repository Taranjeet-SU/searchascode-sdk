"""Agentic SAC eval: closed_book vs sac_base (multi-hop, no critic) vs sac_agent
(critic-in-the-loop, gold-free feedback -> new queries). Reports answer/citation PASS
+ the AVERAGE number of hops the agent used.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_eval_agent --n 195 --workers 8
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
from phase4.altera_eval import ANS_SYS, ctx_text, judge, judge_cite, SHEET
from phase4.altera_eval_tuned import ANS_SYS_TUNED, retrieve_base


def answer(gen, q, ctx, sys):
    prompt = ("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx else f"Question: {q}\n\nAnswer:"
    return gen.complete(prompt, system=sys).strip()


def process(r, gen, reranker, rr_lock, k, max_hops):
    q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
    o = {"ans": {}, "cite": {}, "hops": 0}
    if r.get("Vendor Answer"):
        o["vendor"] = int(judge(gen, q, exp, r["Vendor Answer"]))
        o["vendor_sheet"] = int(r.get("Verdict", "").upper() == "PASS")
    o["ans"]["closed_book"] = int(judge(gen, q, exp, answer(gen, q, [], ANS_SYS)))
    # base: fixed multi-hop decompose, no critic, base prompt
    db = retrieve_base(q, gen, reranker, rr_lock)
    o["ans"]["sac_base"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(db, k), ANS_SYS)))
    o["cite"]["sac_base"] = judge_cite(gen, q, cites, db, k)
    # agent: critic-in-the-loop, tuned prompt
    da, draft, hops = agentic_sac(gen, reranker, rr_lock, q, k=k, max_hops=max_hops)
    o["ans"]["sac_agent"] = int(judge(gen, q, exp, draft))
    o["cite"]["sac_agent"] = judge_cite(gen, q, cites, da, k)
    o["hops"] = hops
    return o


def main(n=195, k=6, workers=8, max_hops=3):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")][:n]
    gen = LLM(); reranker = sac.QwenReranker(); reranker("warm", ["a", "b"]); altera.embedder()
    rr_lock = threading.Lock()
    arms = ["closed_book", "sac_base", "sac_agent"]
    ans = {a: 0 for a in arms}; cite = {a: 0 for a in arms}; cite_n = {a: 0 for a in arms}
    hops = []; vp = vs = done = 0
    print(f"[agent] {len(rows)} questions, {workers} workers, max_hops={max_hops}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, r, gen, reranker, rr_lock, k, max_hops) for r in rows]
        for f in as_completed(futs):
            o = f.result(); done += 1
            vp += o.get("vendor", 0); vs += o.get("vendor_sheet", 0); hops.append(o["hops"])
            for a in arms:
                ans[a] += o["ans"][a]
                if a in o["cite"] and o["cite"][a] is not None:
                    cite_n[a] += 1; cite[a] += o["cite"][a]
            if done % 5 == 0:
                la = " ".join(f"{a}={ans[a]/done:.2f}" for a in arms)
                print(f"[agent] {done}/{len(rows)}  ANS[{la}]  vendor={vp/done:.2f}  "
                      f"avg_hops={np.mean(hops):.2f}  "
                      f"CITE[base={cite['sac_base']/max(1,cite_n['sac_base']):.2f} "
                      f"agent={cite['sac_agent']/max(1,cite_n['sac_agent']):.2f}]", flush=True)
    N = len(rows)
    print(f"\n===== Altera AGENTIC eval (n={N}, max_hops={max_hops}) =====")
    print(f"  vendor: our_judge={vp/N:.3f}  sheet={vs/N:.3f}")
    print(f"  agent avg hops = {np.mean(hops):.2f}  (dist: {dict(zip(*np.unique(hops, return_counts=True)))})")
    for a in arms:
        cp = f"{cite[a]/cite_n[a]:.3f}" if cite_n[a] else "n/a"
        print(f"  {a:11s} answer={ans[a]/N:.3f}  citation={cp}")
    print(f"  llm cost ${gen.usage.cost_usd:.4f}")
    (Path(common.REPO) / "phase4" / "runs" / "altera_agent.json").write_text(json.dumps(
        {"n": N, "max_hops": max_hops, "avg_hops": float(np.mean(hops)),
         "vendor_our": vp / N, "vendor_sheet": vs / N,
         "answer": {a: ans[a] / N for a in arms},
         "citation": {a: (cite[a] / cite_n[a] if cite_n[a] else None) for a in arms}}, indent=2))
    print("[agent] saved runs/altera_agent.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=195); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-hops", type=int, default=3)
    a = ap.parse_args(); main(a.n, a.k, a.workers, a.max_hops)
