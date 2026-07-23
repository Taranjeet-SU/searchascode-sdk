"""TRUE code-mode SAC against the Altera KB: for each question the LLM WRITES a
Python program using retrieval primitives; we execute it and show the generated code.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_codesac --n 3 --start 12
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from phase4 import altera
from phase4.altera_eval import ANS_SYS, ctx_text, judge, judge_cite, rrf

SHEET = Path(common.REPO) / "SearchUnify_Evaluation_Package_2026_06_27 (1).xlsx - 5. All Questions.csv"

SAC_CODE_SYS = """You are a search-as-code retrieval agent for an Altera/Intel FPGA knowledge base.
Write a SHORT Python program that finds the best supporting documents for the variable `question`.
Set a variable `results` to a ranked Python list of doc dicts (each dict has keys 'title','text','url').

Primitives already defined — just call them (do NOT import anything):
  dense(query, k=10)       -> list[doc]   # semantic vector search (fine-tuned gte) over FPGA docs
  keyword(query, k=10)     -> list[doc]   # BM25 over FPGA document text (use for exact part numbers/IDs)
  kb(query, k=10)          -> list[doc]   # BM25 over CURATED knowledge-graph answer cards (often best)
  subqueries(question)     -> list[str]   # decompose a multi-part question into focused sub-queries
  fuse(list_of_lists)      -> list[doc]   # reciprocal-rank fusion of several result lists
  rerank(query, docs, k=10)-> list[doc]   # neural cross-encoder rerank (run on a WIDE fused pool)

Decide by the question shape:
- multi-part / needs several facts -> subqueries(), retrieve each, then fuse().
- exact tokens (part numbers, versions) -> include keyword().
- always consider kb() (curated answers).
Fuse multiple strategies into a wide pool, rerank against the ORIGINAL question, then set `results`.
Output ONLY one Python code block (```python ... ```), no prose."""


def build_primitives(gen, reranker):
    def dense(q, k=10): return altera.dense(q, k)
    def keyword(q, k=10): return altera.bm25_doc(q, k)
    def kb(q, k=10): return altera.bm25_kg(q, k)
    def subqueries(q):
        r = gen.complete(f"Decompose into 1-3 focused search queries (one per line):\n{q}",
                         system="You expand questions into retrieval sub-queries.")
        return [s.strip("-• ").strip() for s in r.splitlines() if s.strip()][:3] or [q]
    def fuse(lists): return rrf(lists)
    def rerank(q, docs, k=10):
        docs = list(docs)
        if not docs:
            return docs
        texts = [(d.get("title", "") + ". " + (d.get("text") or ""))[:800] for d in docs]
        scores = reranker(q, texts)
        return [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])][:k]
    return {"dense": dense, "keyword": keyword, "kb": kb, "subqueries": subqueries,
            "fuse": fuse, "rerank": rerank}


_SAFE = {b: __builtins__[b] if isinstance(__builtins__, dict) else getattr(__builtins__, b)
         for b in ["len", "range", "list", "dict", "sorted", "min", "max", "sum", "enumerate",
                   "zip", "str", "int", "float", "set", "bool", "print", "map", "filter", "abs"]}


def generate_and_run(gen, prims, question, rr_lock=None):
    raw = gen.complete(f"question = {question!r}\n\nWrite the search-as-code program.", system=SAC_CODE_SYS)
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = (m.group(1) if m else raw).strip()
    ns = {"__builtins__": _SAFE, "question": question, **prims}
    err = None
    try:
        exec(compile(code, "<sac-code>", "exec"), ns)  # noqa: S102
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    results = ns.get("results") or []
    if not isinstance(results, list):
        results = list(results) if results else []
    return code, results, err


def main(n=3, start=0, k=6):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question")][start:start + n]
    gen = LLM()
    reranker = sac.QwenReranker(); reranker("warm", ["a", "b"])
    altera.embedder()
    prims = build_primitives(gen, reranker)

    for i, r in enumerate(rows):
        q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
        print("\n" + "=" * 100)
        print(f"Q{start+i+1} [{r.get('Device','?')}/{r.get('Complexity','?')}]: {q[:240]}")
        print("-" * 100)
        code, results, err = generate_and_run(gen, prims, q)
        print("GENERATED SAC CODE:\n" + "\n".join("    " + ln for ln in code.splitlines()))
        if err:
            print(f"\n[exec error -> fell back to empty] {err}")
        print(f"\nRETRIEVED {len(results)} docs; top {k}:")
        for d in results[:k]:
            print(f"   [{d.get('url') or d.get('id')}] {str(d.get('title'))[:76]}")
        ctx = ctx_text(results, k)
        prompt = ("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx else f"Question: {q}\n\nAnswer:"
        ans = gen.complete(prompt, system=ANS_SYS).strip()
        print(f"\nSAC ANSWER:\n   {ans[:500]}")
        ap = judge(gen, q, exp, ans)
        cp = judge_cite(gen, q, cites, results, k)
        print(f"\nVERDICT: answer={'PASS' if ap else 'FAIL'}  "
              f"citation={'PASS' if cp else ('FAIL' if cp is not None else 'n/a')}  "
              f"(sheet: {r.get('Verdict','?')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args(); main(a.n, a.start, a.k)
