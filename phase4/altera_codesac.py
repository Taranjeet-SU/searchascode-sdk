"""TRUE code-mode SAC over the Altera KB with the FULL primitive surface.

The LLM writes a Python retrieval program using the standard SDK SAC surface (sac.search /
hyde_search / prf_search / expand_search / decompose_search / rerank / mmr / semantic_dedup /
compress / retrieve_rerank / ...) PLUS `kb(query, k)` for the curated altera_kg cards and the
fusion/quota/confidence helpers. Executed in a restricted namespace; shows the generated code.

    ALTERA_OS=... HF_TOKEN=... python -m phase4.altera_codesac --n 3 --start 12
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from phase1 import common
from phase1.llm import LLM
from phase1.sac_surface import SAC_SYSTEM
from phase4.altera_eval import ANS_SYS, judge, judge_cite
from phase4.altera_session import agent_namespace

SHEET = Path(common.REPO) / "SearchUnify_Evaluation_Package_2026_06_27 (1).xlsx - 5. All Questions.csv"

ALTERA_NOTE = (
    "\n\n## This knowledge base (Altera/Intel FPGA)\n"
    "- `sac.*` operates over the FPGA DOCUMENT pages (dense + all primitives).\n"
    "- `kb(query, k)` -> curated altera knowledge-graph CARDS as a ResultSet — usually the best source "
    "for specs/part-numbers/definitions. Compose with `fuse([...])`.\n"
    "- Exact part numbers/OPNs often miss as a full string; consider the device root too.\n"
    "- Set `results` to the final ResultSet (a ranked list of hits)."
)

_SAFE = {b: (__builtins__[b] if isinstance(__builtins__, dict) else getattr(__builtins__, b))
         for b in ["len", "range", "list", "dict", "sorted", "min", "max", "sum", "enumerate",
                   "zip", "str", "int", "float", "set", "bool", "print", "map", "filter", "abs"]}


def to_docs(results, k):
    out = []
    for h in list(results)[:k]:
        doc = getattr(h, "document", None)
        meta = (doc.metadata if doc else {}) or {}
        out.append({"title": meta.get("title", ""), "text": (doc.text if doc else "") or "",
                    "url": meta.get("url") or getattr(h, "id", "")})
    return out


def run_program(gen, question):
    raw = gen.complete(f"question = {question!r}\n\nWrite the search-as-code retrieval program; set `results`.",
                       system=SAC_SYSTEM + ALTERA_NOTE)
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = (m.group(1) if m else raw).strip()
    ns = agent_namespace(question); ns["__builtins__"] = _SAFE
    err = None
    try:
        exec(compile(code, "<sac>", "exec"), ns)  # noqa: S102
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    results = ns.get("results")
    return code, (list(results) if results else []), err


def ctx_text(docs, k):
    return [f"[{d['title'][:70]}] {d['text'][:600]}" for d in docs[:k] if d.get("text")]


def main(n=3, start=0, k=6):
    rows = [r for r in csv.DictReader(open(SHEET)) if r.get("Question")][start:start + n]
    gen = LLM()
    for i, r in enumerate(rows):
        q, exp, cites = r["Question"], r["Expected Answer"], r.get("Citations", "")
        print("\n" + "=" * 100)
        print(f"Q{start+i+1} [{r.get('Device')}/{r.get('Complexity')}]: {q[:220]}")
        code, results, err = run_program(gen, q)
        print("GENERATED SAC PROGRAM (full primitive surface):\n" + "\n".join("    " + ln for ln in code.splitlines()))
        if err:
            print(f"[exec error] {err}")
        docs = to_docs(results, k)
        print(f"\nRETRIEVED {len(results)} hits; top {k}:")
        for d in docs:
            print(f"   [{d['url']}] {d['title'][:70]}")
        ctx = ctx_text(docs, k)
        ans = gen.complete(("Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:") if ctx
                           else f"Question: {q}\n\nAnswer:", system=ANS_SYS).strip()
        print(f"\nANSWER:\n   {ans[:450]}")
        golds_docs = [{"url": d["url"], "id": d["url"]} for d in docs]
        ap = judge(gen, q, exp, ans)
        cp = judge_cite(gen, q, cites, golds_docs, k)
        print(f"\nVERDICT: answer={'PASS' if ap else 'FAIL'} citation={'PASS' if cp else ('FAIL' if cp is not None else 'n/a')} (sheet: {r.get('Verdict')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--start", type=int, default=12)
    ap.add_argument("--k", type=int, default=6); a = ap.parse_args()
    main(a.n, a.start, a.k)
