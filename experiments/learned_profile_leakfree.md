# The learned profile, measured without the leak (P2-1)

**Result: the learned-profile lift does not exist.** On both datasets, with the rules mined on a
train split and evaluated on a disjoint test split, every delta is zero or within noise. The
CHANGELOG's "**+2.7 pts all_found from learned synonyms**" (HotpotQA) is **not reproduced** —
and it was not only a leakage artifact, because the lift is absent *in-sample* too.

## The defect

`learn_rules.mine()` iterated `[x for x in qr if any(v>0 ...)][:120]` and `impact_eval.main()`
iterated `[x for x in qr if any(v>0 ...)][:150]` — the **same dict in the same insertion order**.
The first 120 of the 150 evaluation queries were therefore exactly the mining set: **80%
contamination**. `run_learn_sweep.sh` ran that pair for four datasets, so every
learning-pipeline number in `MULTI_DATASET_REPORT.md` inherited it. (`issues.md` P2-1.)

Fixed by [`internal/legacy/phase2/splits.py`](../internal/legacy/phase2/splits.py): one seeded, sorted, order-independent split
used by both passes. Sorting before shuffling matters — the two passes agreed silently *because*
they both trusted dict insertion order.

## Method

```bash
python -m internal.legacy.phase2.learn_rules  --dataset <ds> --n 120 --max-cases 40 --split train
python -m internal.legacy.phase2.impact_eval  --dataset <ds> --n 150 --split train   # in-sample
python -m internal.legacy.phase2.impact_eval  --dataset <ds> --n 150 --split test    # held-out
```

Deltas carry paired bootstrap 95% CIs (`search_as_code.metrics.compare`). Logs:
`experiments/p2_1_leakfree.log`, `experiments/p2_1_leakfree_hotpot.log`.

## Results

### FiQA

| arm | split | recall@10 | all_found@10 | Δ recall vs dense [95% CI] |
|---|---|---|---|---|
| dense (raw) | train | 0.4523 | 0.2467 | — |
| learned-normalized | train | 0.4523 | 0.2467 | +0.0000 [+0.0000, +0.0000] ns |
| synonym-expand+fuse | train | 0.4460 | 0.2467 | −0.0063 [−0.0227, +0.0100] ns |
| dense (raw) | **test** | 0.4480 | 0.2667 | — |
| learned-normalized | **test** | 0.4447 | 0.2667 | −0.0033 [−0.0133, +0.0000] ns |
| synonym-expand+fuse | **test** | 0.4486 | 0.2800 | +0.0006 [−0.0157, +0.0192] ns |

### HotpotQA — the dataset the "+2.7 pts" claim came from

| arm | split | recall@10 | all_found@10 | Δ recall vs dense [95% CI] |
|---|---|---|---|---|
| dense (raw) | train | 0.7833 | 0.6200 | — |
| learned-normalized | train | 0.7867 | 0.6200 | +0.0033 [+0.0000, +0.0100] ns |
| synonym-expand+fuse | train | 0.7833 | 0.6133 | +0.0000 [−0.0100, +0.0100] ns |
| dense (raw) | **test** | 0.7967 | 0.6333 | — |
| learned-normalized | **test** | 0.7967 | 0.6333 | +0.0000 [+0.0000, +0.0000] ns |
| synonym-expand+fuse | **test** | 0.7967 | 0.6333 | +0.0000 [−0.0200, +0.0200] ns |

## Reading it honestly

1. **No lift, held-out or in-sample.** The strongest reading available is that the mined profile
   is inert on these datasets with this evaluation (dense-only arms, recall@10 / all_found@10).
2. **So the leak is not the whole story.** P2-1 predicted an inflated number; what we find is
   no number at all. Two candidate explanations, neither yet tested: the original figure came
   from a profile mined with different settings, or it came from a different arm combination
   than `impact_eval`'s three.
3. **The mechanism barely fires.** On the HotpotQA test split the normalizer changed **0** of
   150 queries and synonym expansion touched 12. A rule set that rarely applies cannot move an
   aggregate, which is consistent with a near-zero delta being the *correct* answer rather than
   a measurement failure.
4. **What we did NOT re-run:** `align_prompts.calibrate_judge` has the same shape (it tunes the
   judge threshold on `qr[:n]`, and that judge is then used at eval time). It is still open.

**Action taken:** both passes are now split-aware and default to mining on `train` / evaluating
on `test`; `impact_eval` prints deltas with intervals and writes
`internal/legacy/phase2/runs/impact_<dataset>_<split>.json`. The claim should be removed from `CHANGELOG.md`
and `MULTI_DATASET_REPORT.md` unless someone can reproduce it under a stated protocol.
