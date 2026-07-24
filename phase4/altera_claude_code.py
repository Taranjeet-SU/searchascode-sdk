"""Division of labor test: CLAUDE authors the retrieval code/rephrasing; 4.1-mini generates
the answer. Compare Claude-authored retrieval vs the fixed recipe, both feeding 4.1-mini.

Claude's retrieval strategy (encoded from driving real queries):
  - expand FPGA acronyms (KB-learned glossary)
  - extract exact OPN/part-number tokens; query them in KG + docs
  - DECOMPOSE a part number to its device root (AGFC019R25A3E3E -> AGFC019) when the full
    OPN misses (a real failure I hit)
  - KG-first fusion
Metrics: answer PASS (4.1-mini gen, judged) + OBJECTIVE citation hit (gold docid overlap,
no LLM). Arms: closed_book | recipe | claude_code.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_claude_code --n 195 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import ANS_SYS, ctx_text, decompose, judge, rrf, SHEET
from phase4.altera_eval_tuned import expand_query


def exact_tokens(q):
    return [t for t in re.findall(r"\b[A-Z0-9][A-Z0-9\-]{5,}\b", q) if any(c.isdigit() for c in t)][:2]


def device_root(t):
    m = re.match(r"([A-Za-z]{2,}\d{2,4})", t)
    return m.group(1) if m else t


def claude_retrieve(q, k=8):
    """Claude-authored retrieval code."""
    qe = expand_query(q)
    pools = [altera.bm25_kg(qe, 12), altera.bm25_kg(q, 10), altera.bm25_doc(qe, 8)]
    for t in exact_tokens(q):
        pools += [altera.bm25_kg(t, 6), altera.bm25_doc(t, 6)]
        root = device_root(t)
        if root != t:
            pools.append(altera.bm25_kg(root, 8))
    return rrf(pools)[:k]


def recipe_retrieve(gen, q, k=8):
    """Fixed recipe (decompose fan-out, KG-first) for comparison."""
    subs = decompose(gen, q)[:2]
    pools = [altera.bm25_kg(q, 12), altera.dense(q, 10), altera.bm25_doc(q, 10)]
    for sq in subs:
        pools += [altera.bm25_kg(sq, 8), altera.dense(sq, 8)]
    return rrf(pools)[:k]


def gold_docids(cites):
    return set(re.findall(r"/docs/(\d+)", cites)) | set(re.findall(r"/(\d{5,7})/", cites))


def cite_hit(docs, golds, k=8):
    if not golds:
        return None
    got = set()
    for d in docs[:k]:
        blob = f"{d.get('url','')} {d.get('id','')} {d.get('docid','')}"
        got |= set(re.findall(r"(\d{5,7})", blob))
    return 1 if (golds & got) else 0


def answer(gen, q, ctx):
    p = ("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx else f"Question: {q}\n\nAnswer:"
    return gen.complete(p, system=ANS_SYS).strip()


def process(r, gen, k):
    q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
    golds = gold_docids(cites)
    o = {"ans": {}, "cite": {}}
    o["ans"]["closed_book"] = int(judge(gen, q, exp, answer(gen, q, [])))
    rec = recipe_retrieve(gen, q, k)
    o["ans"]["recipe"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(rec, k))))
    o["cite"]["recipe"] = cite_hit(rec, golds, k)
    cla = claude_retrieve(q, k)
    o["ans"]["claude_code"] = int(judge(gen, q, exp, answer(gen, q, ctx_text(cla, k))))
    o["cite"]["claude_code"] = cite_hit(cla, golds, k)
    return o


def main(n=195, k=8, workers=4):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question") and r.get("Expected Answer")][:n]
    gen = LLM(); altera.embedder()
    arms = ["closed_book", "recipe", "claude_code"]
    ans = {a: 0 for a in arms}; cite = {a: 0 for a in arms}; cn = {a: 0 for a in arms}; done = 0
    print(f"[claude-code] {len(rows)} questions, {workers} workers", flush=True)
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
                if a in o["cite"] and o["cite"][a] is not None:
                    cn[a] += 1; cite[a] += o["cite"][a]
            if done % 10 == 0:
                la = " ".join(f"{a}={ans[a]/done:.2f}" for a in arms)
                print(f"[claude-code] {done}/{len(rows)} ANS[{la}] "
                      f"CITE[recipe={cite['recipe']/max(1,cn['recipe']):.2f} "
                      f"claude={cite['claude_code']/max(1,cn['claude_code']):.2f}]", flush=True)
    N = len(rows)
    print(f"\n===== Claude-code retrieval vs recipe (n={N}, answer=4.1-mini, citation=objective docid) =====")
    for a in arms:
        cp = f"{cite[a]/cn[a]:.3f}" if cn[a] else "n/a"
        print(f"  {a:12s} answer={ans[a]/N:.3f}  citation={cp}")
    (Path(common.REPO) / "phase4" / "runs" / "claude_code.json").write_text(json.dumps(
        {"n": N, "answer": {a: ans[a]/N for a in arms},
         "citation": {a: (cite[a]/cn[a] if cn[a] else None) for a in arms}}, indent=2))
    print("[claude-code] saved runs/claude_code.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=195); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    main(a.n, a.k, a.workers)
