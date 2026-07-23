"""Altera FINETUNED eval: base SAC vs tuned SAC on the sheet.

Tuning is derived from the KB-only learned profile (phase4/runs/learned_altera.json):
  - glossary query-expansion (expand FPGA acronyms found in the question)
  - KG-first retrieval (weight the curated altera_kg cards in fusion) + decompose fan-out + rerank
  - optimized answer prompt: combine context + expert knowledge, and VERIFY specific
    specs/part-numbers/values against the retrieved KG cards (MCP/altera-kg alignment)

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_eval_tuned --n 195 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import (ANS_SYS, ctx_text, decompose, judge, judge_cite,
                                rrf, SHEET)

PROF = json.loads((Path(common.REPO) / "phase4" / "runs" / "learned_altera.json").read_text())
GLOSSARY = PROF["glossary"]; SYNONYMS = PROF["synonyms"]

# glossary block injected into the tuned answer prompt (domain grounding)
_GLOSS_LINES = "\n".join(f"  {k} = {v}" for k, v in list(GLOSSARY.items())[:40])
ANS_SYS_TUNED = (
    "You are an expert Altera/Intel FPGA support engineer. Answer the question completely and "
    "accurately, combining the provided context WITH your own FPGA expertise. Ground every specific "
    "value (part numbers, speeds, capacities, register/signal names, version numbers) in the context — "
    "verify each such value against the retrieved knowledge cards before stating it, and do not invent "
    "specifics not supported by context or well-known fact. Prefer the curated knowledge-card facts when "
    "they conflict with prose. Be precise and concise.\n\nDomain glossary:\n" + _GLOSS_LINES)


def expand_query(q):
    """Append expansions for any glossary acronym / synonym present (KB-learned)."""
    extra = []
    up = q.upper()
    for ac, exp in GLOSSARY.items():
        if re.search(rf"\b{re.escape(ac)}\b", up) and exp.lower() not in q.lower():
            extra.append(exp)
    low = q.lower()
    for term, variants in SYNONYMS.items():
        if term in low:
            extra += [v for v in variants if v.lower() not in low][:2]
    return q + (" " + " ".join(extra[:6]) if extra else "")


def retrieve_tuned(q, gen, reranker, rr_lock):
    """Multi-hop AND domain-tuned: decompose -> per sub-query KG-first + glossary fan-out
    -> fuse -> rerank. KG cards (altera_kg) weighted 2x (MCP/altera-kg-aligned)."""
    subs = decompose(gen, q)                      # multi-hop (~3 sub-queries)
    pools = []
    for sq in subs:
        sqe = expand_query(sq)                    # glossary expansion per hop
        kg = altera.bm25_kg(sqe, 10)
        pools += [kg, kg, altera.dense(sqe, 10), altera.bm25_doc(sq, 10)]
    pools.append(altera.bm25_kg(expand_query(q), 10))
    fused = rrf(pools)[:30]
    texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in fused]
    if texts:
        with rr_lock:
            scores = reranker(q, texts)
        fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: -x[0])]
    return fused


def retrieve_base(q, gen, reranker, rr_lock):
    subs = decompose(gen, q)
    pools = []
    for sq in subs:
        pools.append(altera.dense(sq, 10)); pools.append(altera.bm25_doc(sq, 10))
    pools.append(altera.bm25_kg(q, 10))
    fused = rrf(pools)[:30]
    texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in fused]
    if texts:
        with rr_lock:
            scores = reranker(q, texts)
        fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: -x[0])]
    return fused


def answer(gen, q, ctx, sys):
    prompt = ("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx else f"Question: {q}\n\nAnswer:"
    return gen.complete(prompt, system=sys).strip()


def process(r, gen, reranker, rr_lock, k):
    q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
    out = {"ans": {}, "cite": {}}
    if r.get("Vendor Answer"):
        out["vendor"] = int(judge(gen, q, exp, r["Vendor Answer"]))
        out["vendor_sheet"] = int(r.get("Verdict", "").upper() == "PASS")
    # closed book (context+knowledge prompt too, for fair contamination control)
    out["ans"]["closed_book"] = int(judge(gen, q, exp, answer(gen, q, [], ANS_SYS)))
    # base SAC (fixed recipe + "use only context" prompt)
    db = retrieve_base(q, gen, reranker, rr_lock)
    out["ans"]["sac_base"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(db, k), ANS_SYS)))
    out["cite"]["sac_base"] = judge_cite(gen, q, cites, db, k)
    # tuned SAC (multi-hop decompose + glossary-expand + KG-first + verify prompt)
    dt = retrieve_tuned(q, gen, reranker, rr_lock)
    out["ans"]["sac_tuned"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(dt, k), ANS_SYS_TUNED)))
    out["cite"]["sac_tuned"] = judge_cite(gen, q, cites, dt, k)
    return out


def main(n=195, k=6, workers=8):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")][:n]
    gen = LLM(); reranker = sac.QwenReranker(); reranker("warm", ["a", "b"]); altera.embedder()
    rr_lock = threading.Lock()
    arms = ["closed_book", "sac_base", "sac_tuned"]
    ans = {a: 0 for a in arms}; cite = {a: 0 for a in arms}; cite_n = {a: 0 for a in arms}
    vp = vs = done = 0
    print(f"[tuned] {len(rows)} questions, {workers} workers  (glossary={len(GLOSSARY)})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, r, gen, reranker, rr_lock, k) for r in rows]
        for f in as_completed(futs):
            o = f.result(); done += 1
            vp += o.get("vendor", 0); vs += o.get("vendor_sheet", 0)
            for a in arms:
                ans[a] += o["ans"][a]
                if a in o["cite"] and o["cite"][a] is not None:
                    cite_n[a] += 1; cite[a] += o["cite"][a]
            if done % 5 == 0:
                la = " ".join(f"{a}={ans[a]/done:.2f}" for a in arms)
                print(f"[tuned] {done}/{len(rows)}  ANS[{la}]  vendor={vp/done:.2f}  "
                      f"CITE[base={cite['sac_base']/max(1,cite_n['sac_base']):.2f} "
                      f"tuned={cite['sac_tuned']/max(1,cite_n['sac_tuned']):.2f}]", flush=True)
    N = len(rows)
    print(f"\n===== Altera FINETUNED eval (n={N}) =====")
    print(f"  vendor: our_judge={vp/N:.3f}  sheet={vs/N:.3f}")
    for a in arms:
        cp = f"{cite[a]/cite_n[a]:.3f}" if cite_n[a] else "n/a"
        print(f"  {a:11s} answer={ans[a]/N:.3f}  citation={cp}")
    print(f"  llm cost ${gen.usage.cost_usd:.4f}")
    (Path(common.REPO) / "phase4" / "runs" / "altera_tuned.json").write_text(json.dumps(
        {"n": N, "vendor_our": vp / N, "vendor_sheet": vs / N,
         "answer": {a: ans[a] / N for a in arms},
         "citation": {a: (cite[a] / cite_n[a] if cite_n[a] else None) for a in arms}}, indent=2))
    print("[tuned] saved runs/altera_tuned.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=195); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); main(a.n, a.k, a.workers)
