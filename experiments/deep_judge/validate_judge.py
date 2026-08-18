#!/usr/bin/env python3
"""Validate the SHIPPED judge against the gold oracle — a fresh measurement, not log re-analysis.

`reselect_judge.py` re-derived the published headline from the tuning logs (DJ-1/2/3). This does
the complementary thing: it runs `search_as_code.harness.DiagnosticJudge` — the prompt actually
shipped in the SDK — over the frozen oracle-labelled eval set and reports agreement with the
oracle, with bootstrap CIs, plus the two rates that matter operationally:

  * **false-accept** — the judge says PASS while golds are still missing. This is the expensive
    one: the agent stops early and the query is lost.
  * **false-reject** — the judge says FAIL on a complete candidate set. Costs another hop.

It also reports two reference points the original write-up lacked:
  * a **majority-class baseline** (always-PASS), which any judge must beat to be worth running;
  * a **signal-only logistic bound** on the same features, i.e. how much of the judge's
    performance is available without the LLM at all.

    python3 -m experiments.deep_judge.validate_judge [--split test|tune|all] [--n N]
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from search_as_code.harness.diagnostic_judge import DIAGNOSTIC_PROMPT, parse_verdict
from search_as_code.metrics import bootstrap_ci, format_ci

HERE = Path(__file__).parent


def load_split(seed=0, tune_frac=0.5):
    path = HERE / "evalset_ce.jsonl"
    ex = [json.loads(ln) for ln in path.open()]
    random.Random(seed).shuffle(ex)
    pos = [e for e in ex if e["oracle_pass"]]
    neg = [e for e in ex if not e["oracle_pass"]]

    def split(xs):
        c = int(len(xs) * tune_frac)
        return xs[:c], xs[c:]
    ptu, pte = split(pos)
    ntu, nte = split(neg)
    return ptu + ntu, pte + nte


def render(e) -> str:
    cov = "\n".join(
        f"  sf{i+1}: \"{c['subfact']}\" sim={c.get('best_sim')} lex={c.get('lexical_overlap')} "
        f"ce={c.get('ce_best')}" for i, c in enumerate(e["coverage"]))
    cands = "\n".join(f"  [{c['id']}] {c['snippet'][:180]}" for c in e["candidates"][:10])
    return (f"QUERY: {e['query']}\n\nSUB-FACTS:\n" +
            "\n".join(f"  {i+1}. {s}" for i, s in enumerate(e["subfacts"])) +
            f"\n\nPER-SUB-FACT COVERAGE:\n{cov}\n\nSCORE SIGNALS: {e['score_signals']}\n\n"
            f"CANDIDATES:\n{cands}\n")


def run_one(llm, e):
    try:
        out = llm.complete(render(e), system=DIAGNOSTIC_PROMPT)
        return parse_verdict(out)["pred_pass"]
    except Exception:
        return 1                      # a crashed judge defaults to PASS (the risky direction)


def confusion(preds, golds):
    tp = sum(1 for p, g in zip(preds, golds) if p and g)
    tn = sum(1 for p, g in zip(preds, golds) if not p and not g)
    fp = sum(1 for p, g in zip(preds, golds) if p and not g)
    fn = sum(1 for p, g in zip(preds, golds) if not p and g)
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"n": len(preds), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(preds) if preds else 0.0,
            "balanced_acc": (sens + spec) / 2,
            "sensitivity": sens, "specificity": spec,
            "false_accept_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_reject_rate": fn / (fn + tp) if fn + tp else 0.0}


def balanced_ci(preds, golds, n_boot=4000, seed=0):
    pos = [1.0 if p else 0.0 for p, g in zip(preds, golds) if g]
    neg = [0.0 if p else 1.0 for p, g in zip(preds, golds) if not g]
    if not pos or not neg:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        a = sum(pos[rng.randrange(len(pos))] for _ in range(len(pos))) / len(pos)
        b = sum(neg[rng.randrange(len(neg))] for _ in range(len(neg))) / len(neg)
        vals.append((a + b) / 2)
    vals.sort()
    mean = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2
    return mean, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def signal_only_bound(examples):
    """How far do the raw signals get you WITHOUT the LLM? (5-fold LogReg, balanced acc.)"""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except Exception:
        return None
    X, y = [], []
    for e in examples:
        ces = [c.get("ce_best") or 0.0 for c in e["coverage"]]
        sims = [c.get("best_sim") or 0.0 for c in e["coverage"]]
        lex = [c.get("lexical_overlap") or 0.0 for c in e["coverage"]]
        sg = e.get("score_signals") or {}
        X.append([min(ces, default=0), sum(ces) / max(1, len(ces)), min(sims, default=0),
                  sum(sims) / max(1, len(sims)), min(lex, default=0), len(ces),
                  sg.get("top3_ratio", 0), sg.get("min_ratio", 0), sg.get("cliff", 0)])
        y.append(int(e["oracle_pass"]))
    sc = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                         np.array(X), np.array(y), cv=5, scoring="balanced_accuracy")
    return float(sc.mean()), float(sc.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "tune", "all"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from phase1.llm import LLM
    tune, test = load_split()
    examples = {"tune": tune, "test": test, "all": tune + test}[a.split]
    if a.n:
        examples = examples[:a.n]

    llm = LLM()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        preds = list(ex.map(lambda e: run_one(llm, e), examples))
    golds = [e["oracle_pass"] for e in examples]

    cm = confusion(preds, golds)
    mean, lo, hi = balanced_ci(preds, golds)
    always_pass = confusion([1] * len(golds), golds)
    bound = signal_only_bound(tune + test)

    print(f"\n=== SHIPPED DiagnosticJudge vs gold oracle — split={a.split}, n={cm['n']} ===")
    print(f"  balanced accuracy   {format_ci(mean, lo, hi)}")
    print(f"  accuracy            {cm['accuracy']:.3f}")
    print(f"  sensitivity (PASS)  {cm['sensitivity']:.3f}     specificity (FAIL) {cm['specificity']:.3f}")
    print(f"  FALSE-ACCEPT rate   {cm['false_accept_rate']:.3f}  <- stops early, loses the query")
    print(f"  false-reject rate   {cm['false_reject_rate']:.3f}  <- costs one more hop")
    print(f"  confusion           tp={cm['tp']} tn={cm['tn']} fp={cm['fp']} fn={cm['fn']}")
    print("\n  reference points")
    print(f"    always-PASS baseline      balanced acc {always_pass['balanced_acc']:.3f} "
          f"(accuracy {always_pass['accuracy']:.3f})")
    if bound:
        print(f"    signal-only LogReg (5cv)  balanced acc {bound[0]:.3f} +/- {bound[1]:.3f} "
              f"<- available WITHOUT the LLM")
    verdict = ("beats" if mean > always_pass["balanced_acc"] + 0.02 else "does NOT clearly beat")
    print(f"\n  => the judge {verdict} the always-PASS baseline.")
    if bound and mean <= bound[0] + 0.02:
        print("  => and it does not clearly beat a plain logistic model on the same signals,")
        print("     i.e. the LLM is not adding much over the features it is shown.")

    out = {"split": a.split, "confusion": cm,
           "balanced_acc_ci": [round(mean, 4), round(lo, 4), round(hi, 4)],
           "always_pass_baseline": always_pass,
           "signal_only_logreg": bound}
    (HERE / f"judge_validation_{a.split}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote judge_validation_{a.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
