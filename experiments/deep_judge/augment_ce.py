"""Augment the frozen eval set with a CROSS-ENCODER coverage signal.

The bi-encoder cosine (gte-base) is saturated high (PASS min_sim 0.86 vs FAIL 0.81 — a 0.05 gap), so
the judge can't tell a covered sub-fact from a missing one. A cross-encoder scores (sub-fact, candidate)
relevance on a wide, calibrated scale (positive = relevant, strongly negative = irrelevant): PASS vs FAIL
separate by ~5.5 points. We add, per sub-fact, the best cross-encoder score over the candidates — the
signal the diagnostic judge actually needs to flag "sub-fact 3's best candidate scores -5 -> missing".

    python -m experiments.deep_judge.augment_ce   # evalset.jsonl -> evalset_ce.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import search_as_code as sac

HERE = Path(__file__).parent


def main():
    ex = [json.loads(l) for l in (HERE / "evalset.jsonl").open()]
    rr = sac.CrossEncoderReranker()
    for i, e in enumerate(ex):
        texts = [c["snippet"] for c in e["candidates"]] or [""]
        for j, sf in enumerate(e["subfacts"]):
            scores = rr(sf, texts)
            best = max(range(len(scores)), key=lambda k: scores[k]) if scores else -1
            e["coverage"][j]["ce_best"] = round(float(scores[best]), 2) if scores else -10.0
            e["coverage"][j]["ce_best_id"] = e["candidates"][best]["id"] if best >= 0 and e["candidates"] else ""
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ex)}", flush=True)
    out = HERE / "evalset_ce.jsonl"
    with out.open("w") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")
    # quick separability sanity
    import numpy as np
    P = [min(c["ce_best"] for c in e["coverage"]) for e in ex if e["oracle_pass"]]
    F = [min(c["ce_best"] for c in e["coverage"]) for e in ex if not e["oracle_pass"]]
    print(f"[augment_ce] wrote {out} · min-CE PASS={np.mean(P):.2f} FAIL={np.mean(F):.2f} gap={np.mean(P)-np.mean(F):.2f}")


if __name__ == "__main__":
    main()
