"""Judge Claude's (the session LLM's) hand-written answers vs gold, using the validated
4.1-mini judge, and track the running Claude-as-agent PASS rate vs the vendor.

Loop per batch: I read agent_context.jsonl rows, write my answers to a batch file
{idx: answer}, then run this -> appends {idx, verdict} to agent_answers.jsonl + prints score.

    python -m phase4.altera_judge_mine phase4/runs/mybatch.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from phase1 import common
from phase1.llm import LLM
from phase4.altera_eval import judge

CTX = Path(common.REPO) / "phase4" / "runs" / "agent_context.jsonl"
OUT = Path(common.REPO) / "phase4" / "runs" / "agent_answers.jsonl"


def load_ctx():
    d = {}
    for line in CTX.open():
        try:
            r = json.loads(line); d[r["idx"]] = r
        except Exception:
            pass
    return d


def main(batch_file):
    ctx = load_ctx()
    mine = json.loads(Path(batch_file).read_text())          # {idx(str): my_answer}
    gen = LLM()
    already = set()
    if OUT.exists():
        already = {json.loads(l)["idx"] for l in OUT.open() if l.strip()}
    n_new = 0
    with OUT.open("a") as fh:
        for k, ans in mine.items():
            idx = int(k)
            if idx in already or idx not in ctx:
                continue
            r = ctx[idx]
            v = int(judge(gen, r["question"], r["gold"], ans))
            fh.write(json.dumps({"idx": idx, "verdict": v,
                                 "complexity": r.get("complexity"),
                                 "sheet_verdict": r.get("verdict"),
                                 "answer": ans[:500]}) + "\n")
            n_new += 1
    # running tally
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    claude = sum(r["verdict"] for r in rows) / len(rows) if rows else 0
    vend = sum(1 for r in rows if str(r.get("sheet_verdict", "")).upper() == "PASS") / len(rows) if rows else 0
    print(f"[judge-mine] +{n_new} judged, total {len(rows)}/195")
    print(f"  Claude-as-agent PASS = {claude:.3f}   vs   vendor(sheet) PASS = {vend:.3f}  (same questions)")
    for c in ["easy", "medium", "high"]:
        sub = [r for r in rows if r.get("complexity") == c]
        if sub:
            print(f"    {c:7s} (n={len(sub)}): Claude={sum(r['verdict'] for r in sub)/len(sub):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
