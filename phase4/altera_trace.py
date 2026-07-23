"""Trace the SAC arm on a few Altera questions — full transparency: the decomposed
sub-queries, the retrieved/reranked source docs, the generated answer, the gold
answer + gold citations, and both judge verdicts.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_trace --n 3 --start 10
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import answer, ctx_text, decompose, judge, judge_cite, rrf

SHEET = Path(common.REPO) / "SearchUnify_Evaluation_Package_2026_06_27 (1).xlsx - 5. All Questions.csv"
W = 100


def main(n=3, start=0, k=6):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question")]
    rows = rows[start:start + n]
    gen = LLM()
    reranker = sac.QwenReranker(); reranker("warm", ["a", "b"])
    altera.embedder()

    for i, r in enumerate(rows):
        q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
        print("\n" + "=" * W)
        print(f"Q{start+i+1} [{r.get('Device','?')}/{r.get('Topic','?')}, {r.get('Complexity','?')}]: {q[:260]}")
        print("-" * W)

        # SAC: decompose
        subs = decompose(gen, q)
        print("SAC sub-queries:")
        for s in subs:
            print(f"   • {s[:110]}")

        # SAC: fan-out + fuse
        pools = []
        for sq in subs:
            pools.append(altera.dense(sq, 10)); pools.append(altera.bm25_doc(sq, 10))
        pools.append(altera.bm25_kg(q, 10))
        fused = rrf(pools)[:30]
        # rerank
        texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in fused]
        scores = reranker(q, texts) if texts else []
        if texts:
            fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: -x[0])]
        print(f"\nSAC retrieved (top {k} after rerank, from {len(fused)} fused candidates):")
        for d in fused[:k]:
            print(f"   [{d.get('url') or d.get('id')}] {str(d.get('title'))[:80]}")
            print(f"        {str(d.get('text'))[:130].strip()}".replace("\n", " "))

        # SAC: answer
        ctx = ctx_text(fused, k)
        ans = answer(gen, q, ctx)
        print(f"\nSAC ANSWER:\n   {ans[:600]}")

        # gold + judges
        print(f"\nGOLD answer:\n   {exp[:400]}")
        print(f"GOLD citations: {(cites[:200] or '(none)')}")
        ap = judge(gen, q, exp, ans)
        cp = judge_cite(gen, q, cites, fused, k)
        print(f"\nVERDICT:  answer={'PASS' if ap else 'FAIL'}   "
              f"citation={'PASS' if cp else ('FAIL' if cp is not None else 'n/a')}")
        print(f"(sheet verdict for this Q: {r.get('Verdict','?')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args(); main(a.n, a.start, a.k)
