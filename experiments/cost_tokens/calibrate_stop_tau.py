"""Calibrate the hop-0 STOP threshold per corpus — the signals-first StopGate (fable.md §2b).

Two LLM-side stop mechanisms failed on BrowseComp-qwen8b: both judge premises escalate 100%
(the field's ~0.65 judge-AUROC ceiling), and a universal sigmoid-CE >= 0.5 floor guard misses
this corpus's weak-CE golds (kept only 6/10 solved queries). The fix the literature converged
on (TASR/QPP): a cheap signal calibrated once per (model, corpus).

Feature: max whole-query sigmoid-CE over the vetted baseline's top-10.
Label:   oracle hop-0 PASS = all golds already in the baseline's top-10.
tau:     maximizes balanced accuracy on a calibration slice DISJOINT from every eval slice
         (eval/control use rows[:100]; calibration uses rows[100:100+n]).

    python -m experiments.cost_tokens.calibrate_stop_tau browsecomp_qwen8b [n=60]
Writes stop_tau_<corpus>.json; run_cost auto-loads it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase1.llm import LLM
from search_as_code.harness import diagnostic_judge as djm

from experiments.cost_tokens.run_cost import HERE, load_corpus, load_explore_seed


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "browsecomp_qwen8b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    gen = LLM()
    session, rows = load_corpus(corpus, gen)
    _, forged = load_explore_seed(corpus)
    assert forged is not None, "needs the gate-selected primitive"
    cal = rows[100:100 + n]                      # disjoint from eval/control rows[:100]
    print(f"[tau] corpus={corpus} calibration n={len(cal)} (rows[100:{100+n}])", flush=True)

    feats, labels = [], []
    for i, r in enumerate(cal):
        base = [str(x) for x in forged.run(session, r["query"], top_k=50)]
        docs = {d.id: (d.text or "")[:700] for d in session.store.get(base[:10])}
        texts = [docs.get(x, "") for x in base[:10]]
        csc = djm.candidate_scores(session.reranker, r["query"], texts)
        feats.append(max(csc) if csc else 0.0)
        labels.append(int(set(map(str, r["gold_ids"])) <= set(base[:10])))
        if (i + 1) % 20 == 0:
            print(f"[tau] {i+1}/{len(cal)}", flush=True)

    def bal(tau):
        tp = sum(1 for f, y in zip(feats, labels) if f >= tau and y)
        tn = sum(1 for f, y in zip(feats, labels) if f < tau and not y)
        p, ng = sum(labels), len(labels) - sum(labels)
        return ((tp / p if p else 0.0) + (tn / ng if ng else 0.0)) / 2

    cands = sorted(set(round(f, 4) for f in feats))
    best = max(cands, key=bal) if cands else 0.5
    out = {"tau": best, "balanced_acc": round(bal(best), 4), "n": len(cal),
           "pass_rate_at_tau": round(sum(1 for f in feats if f >= best) / len(feats), 3),
           "base_solved_rate": round(sum(labels) / len(labels), 3),
           "feature": "max sigmoid-CE over vetted-baseline top-10", "slice": f"rows[100:{100+n}]"}
    (HERE / f"stop_tau_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
