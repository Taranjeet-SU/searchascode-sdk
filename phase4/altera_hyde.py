"""HyDE test: since closed-book (parametric knowledge) is strong on FPGA, a hypothetical
answer the LLM writes should look like the real docs -> retrieving with IT should beat the
raw query. Compare recipe vs claude_code vs HyDE, on OBJECTIVE citation-hit + 4.1-mini answer.

  hyde        : 4.1-mini writes a hypothetical answer passage -> dense(embed it) + bm25_kg(its terms) + kb(q)
  hyde_claude : HyDE fused with Claude's authored retrieval (best of both)

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_hyde --n 195 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_claude_code import (ANS_SYS, cite_hit, claude_retrieve, ctx_text,
                                       gold_docids, recipe_retrieve)
from phase4.altera_eval import judge, rrf, SHEET

HYDE_SYS = ("Write a short factual passage (3-5 sentences), as if from an Altera/Intel FPGA datasheet or "
            "knowledge base, that would directly ANSWER the question. State specific device names, specs, "
            "and values as best you know. This passage is used to retrieve real documents, so be concrete.")


def hyde_passage(gen, q):
    return gen.complete(f"Question: {q}\n\nPassage:", system=HYDE_SYS).strip()


def hyde_retrieve(gen, q, k=8):
    hyp = hyde_passage(gen, q)
    pools = [altera.dense(hyp, 12), altera.bm25_kg(hyp, 12), altera.bm25_kg(q, 8), altera.bm25_doc(hyp, 8)]
    return rrf(pools)[:k], hyp


def answer(gen, q, ctx):
    p = ("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx else f"Question: {q}\n\nAnswer:"
    return gen.complete(p, system=ANS_SYS).strip()


def process(r, gen, k):
    q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
    golds = gold_docids(cites)
    o = {"ans": {}, "cite": {}}
    rec = recipe_retrieve(gen, q, k)
    o["ans"]["recipe"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(rec, k)))); o["cite"]["recipe"] = cite_hit(rec, golds, k)
    cla = claude_retrieve(q, k)
    o["ans"]["claude_code"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(cla, k)))); o["cite"]["claude_code"] = cite_hit(cla, golds, k)
    hy, _ = hyde_retrieve(gen, q, k)
    o["ans"]["hyde"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(hy, k)))); o["cite"]["hyde"] = cite_hit(hy, golds, k)
    hyc = rrf([hy, cla])[:k]
    o["ans"]["hyde_claude"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(hyc, k)))); o["cite"]["hyde_claude"] = cite_hit(hyc, golds, k)
    return o


def main(n=195, k=8, workers=3):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")][:n]
    gen = LLM(); altera.embedder()
    arms = ["recipe", "claude_code", "hyde", "hyde_claude"]
    ans = {a: 0 for a in arms}; cite = {a: 0 for a in arms}; cn = {a: 0 for a in arms}; done = 0
    print(f"[hyde] {len(rows)} questions, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, r, gen, k) for r in rows]
        for f in as_completed(futs):
            try:
                o = f.result()
            except Exception:
                done += 1; continue
            done += 1
            for a in arms:
                ans[a] += o["ans"][a]
                if o["cite"].get(a) is not None:
                    cn[a] += 1; cite[a] += o["cite"][a]
            if done % 10 == 0:
                print(f"[hyde] {done}/{len(rows)} " +
                      " ".join(f"{a}[a={ans[a]/done:.2f},c={cite[a]/max(1,cn[a]):.2f}]" for a in arms), flush=True)
    N = len(rows)
    print(f"\n===== HyDE vs recipe vs claude_code (n={N}, answer=4.1-mini, citation=objective) =====")
    for a in arms:
        print(f"  {a:12s} answer={ans[a]/N:.3f}  citation={cite[a]/cn[a]:.3f}")
    (Path(common.REPO)/"phase4"/"runs"/"hyde.json").write_text(json.dumps(
        {"n": N, "answer": {a: ans[a]/N for a in arms}, "citation": {a: cite[a]/cn[a] for a in arms}}, indent=2))
    print("[hyde] saved runs/hyde.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=195); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    main(a.n, a.k, a.workers)
