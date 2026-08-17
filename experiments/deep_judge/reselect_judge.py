#!/usr/bin/env python3
"""Re-derive the judge headline honestly — DJ-1, DJ-2, DJ-3.

The published claim is "the LLM judge reaches 0.721 held-out balanced accuracy, and 0.72 IS the
signal ceiling". Three defects in how that number was produced (issues.md §10):

  DJ-1  `tune_judge.py:146-150` selected the best round on TUNE but **tie-broke on TEST**, so the
        reported number is not a clean held-out estimate.
  DJ-2  The adopted gain is +0.011 balanced accuracy on TUNE — a **single example** flipping
        (tn 36->37, fp 11->10) — against a 95% interval of roughly ±0.09 at n=100. No interval
        was reported anywhere.
  DJ-3  The "independent Qwen-32B critic | 0.70" row is `## Best (round 0)`, i.e. the **untuned**
        INITIAL_PROMPT. The critic never produced an adopted revision.

This script re-reads the tuning logs, re-selects on TUNE only (ties broken by the earliest
round, never by TEST), and attaches bootstrap CIs via the promoted
`search_as_code.metrics.bootstrap_ci`. It needs no API calls — it is pure re-analysis of
artifacts already in the repo.

    python3 -m experiments.deep_judge.reselect_judge
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from search_as_code.metrics import bootstrap_ci, compare, format_ci

HERE = Path(__file__).parent
LOGS = {
    "bi-encoder cosine (v0)": "tuning_log_same.md",
    "+ cross-encoder, same-model critic (v1)": "tuning_log_ce_same.md",
    "+ cross-encoder, independent Qwen-32B critic": "tuning_log_ce_qwen.md",
}
ROUND_RE = re.compile(r"^## Round (\d+)", re.M)
BEST_RE = re.compile(r"^## Best \(round (\d+)\)", re.M)
CM_RE = re.compile(r"^(TUNE|TEST): (\{.*\})", re.M)


def parse(path: Path) -> tuple[list[dict], int | None]:
    """[{round, tune, test}], plus the round the original run adopted."""
    text = path.read_text()
    rounds: list[dict] = []
    starts = [(int(m.group(1)), m.start()) for m in ROUND_RE.finditer(text)]
    for i, (rnd, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        block = text[start:end]
        cms = {kind: ast.literal_eval(body) for kind, body in CM_RE.findall(block)}
        if "TUNE" in cms and "TEST" in cms:
            rounds.append({"round": rnd, "tune": cms["TUNE"], "test": cms["TEST"]})
    m = BEST_RE.search(text)
    return rounds, (int(m.group(1)) if m else None)


def outcomes(cm: dict) -> list[float]:
    """Per-example correctness (1/0) reconstructed from the confusion matrix."""
    return [1.0] * (cm["tp"] + cm["tn"]) + [0.0] * (cm["fp"] + cm["fn"])


def balanced_outcomes(cm: dict) -> tuple[list[float], list[float]]:
    """Per-example correctness split by true class, for a balanced-accuracy bootstrap."""
    pos = [1.0] * cm["tp"] + [0.0] * cm["fn"]      # truly PASS
    neg = [1.0] * cm["tn"] + [0.0] * cm["fp"]      # truly FAIL
    return pos, neg


def balanced_ci(cm: dict, n_boot: int = 4000) -> tuple[float, float, float]:
    """Bootstrap the balanced accuracy = mean(sensitivity, specificity)."""
    import random
    pos, neg = balanced_outcomes(cm)
    if not pos or not neg:
        return 0.0, 0.0, 0.0
    rng = random.Random(0)
    vals = []
    for _ in range(n_boot):
        s = sum(pos[rng.randrange(len(pos))] for _ in range(len(pos))) / len(pos)
        t = sum(neg[rng.randrange(len(neg))] for _ in range(len(neg))) / len(neg)
        vals.append((s + t) / 2)
    vals.sort()
    mean = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2
    return mean, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def honest_select(rounds: list[dict], margin: float = 0.01) -> dict:
    """Select on TUNE only; ties broken by the EARLIEST round (never by TEST)."""
    best = rounds[0]
    for r in rounds[1:]:
        if r["tune"]["balanced_acc"] > best["tune"]["balanced_acc"] + margin:
            best = r
    return best


def main() -> int:
    report: dict = {"note": "DJ-1/2/3 re-analysis: selection on TUNE only, CIs attached.",
                    "variants": {}}
    print("Judge re-analysis — selection on TUNE only, bootstrap 95% CIs (DJ-1/2/3)\n")
    for label, fname in LOGS.items():
        path = HERE / fname
        if not path.exists():
            print(f"  (skipped {fname} — not present)")
            continue
        rounds, adopted = parse(path)
        if not rounds:
            print(f"  (skipped {fname} — no parseable rounds)")
            continue
        picked = honest_select(rounds)
        r0 = rounds[0]

        tune_m, tune_lo, tune_hi = balanced_ci(picked["tune"])
        test_m, test_lo, test_hi = balanced_ci(picked["test"])
        base_test = balanced_ci(r0["test"])

        gain = compare(outcomes(picked["test"]), outcomes(r0["test"]), paired=False)

        print(f"── {label}")
        print(f"   rounds parsed        : {len(rounds)}  (original log adopted round {adopted})")
        print(f"   honest pick (TUNE)   : round {picked['round']}")
        print(f"   TUNE balanced-acc    : {format_ci(tune_m, tune_lo, tune_hi)}")
        print(f"   TEST balanced-acc    : {format_ci(test_m, test_lo, test_hi)}")
        print(f"   round-0 TEST (untuned): {format_ci(*base_test)}")
        print(f"   tuning gain on TEST  : {gain['delta']:+.3f} "
              f"[{gain['lo']:+.3f}, {gain['hi']:+.3f}]  "
              f"{'SIGNIFICANT' if gain['significant'] else 'NOT distinguishable from noise'}")
        if adopted is not None and adopted != picked["round"]:
            print(f"   ** DJ-1 CONFIRMED: the log adopted round {adopted}, honest selection on "
                  f"TUNE alone gives round {picked['round']}")
        if picked["round"] == 0:
            print("   ** DJ-3 CONFIRMED: the adopted round IS round 0 — the untuned INITIAL_PROMPT. "
                  "No critic revision was ever adopted for this variant.")
        print()

        report["variants"][label] = {
            "log": fname, "rounds": len(rounds), "originally_adopted_round": adopted,
            "honest_round": picked["round"],
            "tune_balanced_acc": [round(tune_m, 4), round(tune_lo, 4), round(tune_hi, 4)],
            "test_balanced_acc": [round(test_m, 4), round(test_lo, 4), round(test_hi, 4)],
            "round0_test_balanced_acc": [round(x, 4) for x in base_test],
            "tuning_gain_on_test": gain,
            "is_untuned_round_zero": picked["round"] == 0,
        }

    out = HERE / "judge_reanalysis.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
