"""One deterministic train/test split, shared by the mining and evaluation passes.

`issues.md` P2-1: `learn_rules.mine()` iterated `[x for x in qr if ...][:120]` and
`impact_eval.main()` iterated `[x for x in qr if ...][:150]` — the SAME dict in the SAME
insertion order — so **the first 120 of the 150 evaluation queries were exactly the mining
set**: 80% contamination. `run_learn_sweep.sh` ran precisely that pair for four datasets, so
the CHANGELOG's "+2.7 pts all_found from learned synonyms" and every learning-pipeline number
in `MULTI_DATASET_REPORT.md` inherit the leak. `align_prompts.calibrate_judge` has the same
shape (it tunes the judge threshold on `qr[:n]`, then that judge is used at eval time).

Both passes now call :func:`split_qids` with the same seed and take disjoint halves, so a
"learned" lift is measured on queries the rules have never seen.
"""
from __future__ import annotations

import random

DEFAULT_SEED = 0


def labelled_qids(qr: dict) -> list[str]:
    """Query ids that actually carry a positive judgement, in a stable order."""
    return sorted(x for x in qr if any(v > 0 for v in qr[x].values()))


def split_qids(qr: dict, n: int | None = None, seed: int = DEFAULT_SEED,
               train_frac: float = 0.5) -> tuple[list[str], list[str]]:
    """Return ``(train_qids, test_qids)`` — disjoint, seeded, order-independent.

    Sorted before shuffling so the split does not depend on dict insertion order (which is what
    made the two passes silently agree in the first place). ``n`` caps the TOTAL considered.
    """
    qids = labelled_qids(qr)
    rng = random.Random(seed)
    rng.shuffle(qids)
    if n is not None:
        qids = qids[:n]
    cut = int(len(qids) * train_frac)
    return qids[:cut], qids[cut:]


def pick(qr: dict, split: str, n: int | None = None, seed: int = DEFAULT_SEED) -> list[str]:
    """``split`` in {"train", "test", "all"} -> the query ids for that split."""
    train, test = split_qids(qr, n=n, seed=seed)
    if split == "train":
        return train
    if split == "test":
        return test
    if split == "all":
        return train + test
    raise ValueError(f"unknown split {split!r} (train|test|all)")
