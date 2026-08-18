#!/usr/bin/env python3
"""Validate the SHIPPED judge against the gold oracle — a fresh measurement, not log re-analysis.

`reselect_judge.py` re-derived the published headline from the tuning logs (DJ-1/2/3). This does
the complementary thing: it runs `search_as_code.harness.DiagnosticJudge` — the prompt actually
shipped in the SDK — over the frozen oracle-labelled eval set and reports agreement with the
oracle, with bootstrap CIs, plus the two rates that matter operationally:

  * **false-accept** — the judge says PASS while golds are still missing. This is the expensive
    one: the agent stops early and the query is lost.
  * **false-reject** — the judge says FAIL on a complete candidate set. Costs another hop.

2026-08-18 rework (issues.md §17):
  * DJ-6  — the split is now GROUPED BY QUERY: each query contributes a shallow and a deep
    example, and the old label-stratified shuffle put 52 of ~76 test queries in tune too. The
    bootstrap now resamples query groups, not rows.
  * DJ-8  — the LogReg bound uses GroupKFold by query (the old unshuffled 5-fold on a
    split+label-blocked row order was a row-ordering artifact: 0.722 in one order, 0.705 in
    another), and is ALSO reported train-on-tune/test-on-test so it is measured on exactly the
    rows the judge is.
  * DJ-9  — a tuned min-CE threshold baseline (fit on tune only) is reported: the judge's own
    prompt is a verbalized threshold on min_i ce_i, so this is the floor it must beat.
  * DJ-14 — examples are rendered with the SHIPPED renderer (diagnostic_judge.render), not a
    third format private to this script.

    python3 -m experiments.deep_judge.validate_judge [--split test|tune|all] [--n N]
                                                     [--legacy-split]  # the old leaky split
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from search_as_code.harness import diagnostic_judge as dj
from search_as_code.harness.diagnostic_judge import (
    DIAGNOSTIC_PROMPT,
    SUFFICIENCY_PROMPT,
    parse_verdict,
)
from search_as_code.metrics import format_ci

HERE = Path(__file__).parent


def _load_examples():
    return [json.loads(ln) for ln in (HERE / "evalset_ce.jsonl").open()]


def _qkey(e) -> str:
    return e.get("qid") or e["query"]


def load_split(seed=0, tune_frac=0.5, legacy=False):
    """Split GROUPED BY QUERY (DJ-6). ``legacy=True`` reproduces the old leaky
    label-stratified row shuffle, kept only for comparison with published numbers."""
    ex = _load_examples()
    if legacy:
        random.Random(seed).shuffle(ex)
        pos = [e for e in ex if e["oracle_pass"]]
        neg = [e for e in ex if not e["oracle_pass"]]

        def split(xs):
            c = int(len(xs) * tune_frac)
            return xs[:c], xs[c:]
        ptu, pte = split(pos)
        ntu, nte = split(neg)
        return ptu + ntu, pte + nte

    groups: dict[str, list] = {}
    for e in ex:
        groups.setdefault(_qkey(e), []).append(e)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = tune_frac * len(ex)
    tune, test, count = [], [], 0
    for k in keys:
        if count < target:
            tune.extend(groups[k])
            count += len(groups[k])
        else:
            test.extend(groups[k])
    overlap = {_qkey(e) for e in tune} & {_qkey(e) for e in test}
    assert not overlap, f"query leak across split: {len(overlap)}"
    return tune, test


def render(e) -> str:
    """The SHIPPED renderer (DJ-14) — tuning, validation and production see one format."""
    sig = e.get("score_signals") or dj.score_signals([c.get("score", 0.0) for c in e["candidates"]])
    return dj.render(e["query"], e["subfacts"], e["candidates"][:10], e["coverage"], sig)


def run_one(llm, e, system=DIAGNOSTIC_PROMPT):
    try:
        out = llm.complete(render(e), system=system)
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


def balanced_ci_grouped(preds, examples, n_boot=4000, seed=0):
    """Bootstrap balanced accuracy resampling QUERY GROUPS (DJ-6): a query's shallow and
    deep rows are correlated, so i.i.d. row resampling understated the interval."""
    by_group: dict[str, list[tuple[int, int]]] = {}
    for p, e in zip(preds, examples):
        by_group.setdefault(_qkey(e), []).append((int(bool(p)), int(bool(e["oracle_pass"]))))
    gkeys = list(by_group)
    rng = random.Random(seed)

    def bal(pairs):
        pos = [p for p, g in pairs if g]
        neg = [1 - p for p, g in pairs if not g]
        if not pos or not neg:
            return None
        return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2

    point = bal([pg for pairs in by_group.values() for pg in pairs]) or 0.0
    vals = []
    for _ in range(n_boot):
        sample = [pg for _ in gkeys for pg in by_group[gkeys[rng.randrange(len(gkeys))]]]
        b = bal(sample)
        if b is not None:
            vals.append(b)
    vals.sort()
    if not vals:
        return point, 0.0, 0.0
    return point, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def _features(e):
    ces = [c.get("ce_best") or 0.0 for c in e["coverage"]]
    sims = [c.get("best_sim") or 0.0 for c in e["coverage"]]
    lex = [c.get("lexical_overlap") or 0.0 for c in e["coverage"]]
    sg = e.get("score_signals") or {}
    return [min(ces, default=0), sum(ces) / max(1, len(ces)), min(sims, default=0),
            sum(sims) / max(1, len(sims)), min(lex, default=0), len(ces),
            sg.get("top3_ratio", 0), sg.get("min_ratio", 0), sg.get("cliff", 0)]


def signal_only_bound(tune, test):
    """The no-LLM references the judge must beat (DJ-8/DJ-9), leak-free:
    * LogReg GroupKFold(5) by query over all examples (no shallow/deep pairing leak);
    * LogReg fit on tune, scored on test — the same rows the judge is scored on;
    * best min-CE threshold fit on TUNE only, scored on test."""
    out = {}
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import GroupKFold, cross_val_score
        allex = tune + test
        X = np.array([_features(e) for e in allex])
        y = np.array([int(e["oracle_pass"]) for e in allex])
        groups = np.array([_qkey(e) for e in allex])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        sc = cross_val_score(lr, X, y, cv=GroupKFold(n_splits=5), groups=groups,
                             scoring="balanced_accuracy")
        out["logreg_groupkfold"] = [round(float(sc.mean()), 4), round(float(sc.std()), 4)]
        Xtu = np.array([_features(e) for e in tune]); ytu = np.array([int(e["oracle_pass"]) for e in tune])
        Xte = np.array([_features(e) for e in test]); yte = np.array([int(e["oracle_pass"]) for e in test])
        lr.fit(Xtu, ytu)
        out["logreg_tune_to_test"] = round(float(balanced_accuracy_score(yte, lr.predict(Xte))), 4)
    except Exception as exc:
        out["logreg_error"] = str(exc)[:120]

    # min-CE threshold: fit on tune, score on test (the judge's prompt is this rule verbalized)
    def min_ce(e):
        return min((c.get("ce_best") or 0.0 for c in e["coverage"]), default=0.0)

    def bal_at(th, exs):
        preds = [1 if min_ce(e) > th else 0 for e in exs]
        return confusion(preds, [e["oracle_pass"] for e in exs])["balanced_acc"]

    ths = sorted({round(min_ce(e), 2) for e in tune})
    if ths:
        best_th = max(ths, key=lambda t: bal_at(t, tune))
        out["min_ce_threshold"] = {"threshold_fit_on_tune": best_th,
                                   "tune_balanced_acc": round(bal_at(best_th, tune), 4),
                                   "test_balanced_acc": round(bal_at(best_th, test), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "tune", "all"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--legacy-split", action="store_true",
                    help="the pre-DJ-6 leaky split, for comparison with published numbers")
    ap.add_argument("--premise", default="coverage", choices=["coverage", "sufficiency"],
                    help="which stop question the judge asks (fable.md §2b action 1)")
    a = ap.parse_args()

    from phase1.llm import LLM
    tune, test = load_split(legacy=a.legacy_split)
    tq, teq = {_qkey(e) for e in tune}, {_qkey(e) for e in test}
    print(f"split: {len(tune)} tune / {len(test)} test examples; "
          f"{len(tq)}/{len(teq)} distinct queries; overlap={len(tq & teq)} "
          f"({'LEGACY leaky split' if a.legacy_split else 'grouped by query, DJ-6 fixed'})")
    examples = {"tune": tune, "test": test, "all": tune + test}[a.split]
    if a.n:
        examples = examples[:a.n]

    llm = LLM()
    system = SUFFICIENCY_PROMPT if a.premise == "sufficiency" else DIAGNOSTIC_PROMPT
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        preds = list(ex.map(lambda e: run_one(llm, e, system), examples))
    golds = [e["oracle_pass"] for e in examples]

    cm = confusion(preds, golds)
    mean, lo, hi = balanced_ci_grouped(preds, examples)
    always_pass = confusion([1] * len(golds), golds)
    bound = signal_only_bound(tune, test)

    print(f"\n=== SHIPPED DiagnosticJudge vs gold oracle — split={a.split}, n={cm['n']} ===")
    print(f"  balanced accuracy   {format_ci(mean, lo, hi)}   (grouped bootstrap)")
    print(f"  accuracy            {cm['accuracy']:.3f}")
    print(f"  sensitivity (PASS)  {cm['sensitivity']:.3f}     specificity (FAIL) {cm['specificity']:.3f}")
    print(f"  FALSE-ACCEPT rate   {cm['false_accept_rate']:.3f}  <- stops early, loses the query")
    print(f"  false-reject rate   {cm['false_reject_rate']:.3f}  <- costs one more hop")
    print(f"  confusion           tp={cm['tp']} tn={cm['tn']} fp={cm['fp']} fn={cm['fn']}")
    print("\n  no-LLM references (leak-free)")
    print(f"    always-PASS baseline      balanced acc {always_pass['balanced_acc']:.3f}")
    if "logreg_groupkfold" in bound:
        m, s = bound["logreg_groupkfold"]
        print(f"    LogReg GroupKFold(5)      balanced acc {m:.3f} +/- {s:.3f}")
        print(f"    LogReg tune->test         balanced acc {bound['logreg_tune_to_test']:.3f}")
    if "min_ce_threshold" in bound:
        t = bound["min_ce_threshold"]
        print(f"    min-CE > {t['threshold_fit_on_tune']:+.2f} (fit on tune) "
              f"balanced acc {t['test_balanced_acc']:.3f}  <- the floor the LLM must beat (DJ-9)")

    out = {"split": a.split, "legacy_split": a.legacy_split, "premise": a.premise, "confusion": cm,
           "balanced_acc_ci_grouped": [round(mean, 4), round(lo, 4), round(hi, 4)],
           "always_pass_baseline": always_pass,
           "no_llm_references": bound,
           "n_queries": {"tune": len(tq), "test": len(teq), "overlap": len(tq & teq)}}
    suffix = ("_legacy" if a.legacy_split else "") + ("_sufficiency" if a.premise == "sufficiency" else "")
    (HERE / f"judge_validation_{a.split}{suffix}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote judge_validation_{a.split}{suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
