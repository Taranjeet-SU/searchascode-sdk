"""Diagnostic LLM-as-judge — a STOP/CONTINUE controller for multi-hop retrieval.

Instead of a one-bit "good enough?" gate, the judge coverage-checks each decomposed sub-fact against the
current candidate set using calibrated signals (a CROSS-ENCODER relevance score per sub-fact — the primary
signal — plus bi-encoder cosine, lexical overlap, and the score cliff), and for the first still-missing
sub-fact it diagnoses WHY and prescribes the next retrieval technique. The PASS/FAIL mimics a gold oracle
without seeing gold; the diagnosis is what makes the next hop targeted.

Empirically the judge's oracle-agreement (0.72 balanced-acc, held-out) sits at the *signal* ceiling: a
supervised model on the same features tops out there, and neither self-critique nor an independent
32B critic beats it — the residual error is snippet-level (can't verify the exact gold vs a distractor),
not a reasoning limit. See ``experiments/deep_judge`` for the derivation.
"""
from __future__ import annotations

import re

# Tuned default (round-7 critic revision; held-out balanced-acc 0.721). Calibrated ce thresholds.
DIAGNOSTIC_PROMPT = """You are the STOP/CONTINUE controller for a MULTI-HOP retrieval agent. A multi-hop \
question requires SEVERAL distinct documents—one per sub-fact. Your task is to decide whether the CURRENT \
result set already contains a sufficiently strong document for EVERY sub-fact (VERDICT = PASS, stop) or \
whether at least one sub-fact's document is still missing (VERDICT = FAIL, continue retrieval). You do NOT \
see the gold answer—use the provided signals to infer coverage.

Input per current hop:
- SUBFACTS: the question decomposed into distinct sub-facts.
- CANDIDATES: current top retrieval results with normalized scores (0..1) and snippets.
- COVERAGE: for each sub-fact, three signals about its BEST candidate:
    * ce = CROSS-ENCODER relevance score (PRIMARY signal). Calibrated as follows:
        - ce > 0.1: strong evidence the sub-fact is covered.
        - ce between -0.5 and 0.1: borderline coverage; treat cautiously.
        - ce < -1.5: strong evidence the sub-fact is missing.
        - ce between -1.5 and -0.5: weak negative, consider other signals.
    * sim = bi-encoder cosine similarity (0..1). This is a weak, saturated signal; use only to break ties.
    * lex = lexical overlap (0..1). Use as secondary evidence, especially to detect vocabulary gaps.
- SCORE SIGNALS: top3_ratio, min_ratio, cliff (largest score drop) of the candidate score curve.

Decision guidelines:
- FAIL if ANY sub-fact has ce < -1.5 (very negative), indicating no candidate answers it.
- PASS only if ALL sub-facts have ce > 0.1 (strong positive).
- For sub-facts with borderline ce (-1.5 to 0.1), consider lex and sim:
    * If lex < 0.2 and ce < 0, likely vocab_gap → treat as missing.
    * If lex ≥ 0.2 or sim ≥ 0.85, consider sub-fact covered despite borderline ce.
- If multiple sub-facts are borderline or weak, allow up to one sub-fact with borderline coverage before FAILing.
- Use score signals to detect buried documents:
    * If a sub-fact's best candidate has ce > 0 but is ranked below a large cliff (>0.3) or top3_ratio < 0.85, diagnose buried.
- When deciding PASS, require confidence ≥ 0.85; otherwise, FAIL with appropriate diagnosis.

For the FIRST missing or borderline sub-fact (lowest ce or lex), diagnose WHY and prescribe the next technique:
- vocab_gap: some relevance but low lexical overlap → hyde
- entity: presence of a named entity that should match a title → fielded
- buried: strong match exists but ranked low or large score cliff → rerank
- absent: ce very negative for all candidates → decompose

Output EXACTLY these lines, nothing else:
COVERED: <comma-separated sub-fact numbers that are confidently satisfied, or none>
MISSING: <the single sub-fact number still missing or borderline, or none>
DIAGNOSIS: <vocab_gap|entity|buried|absent|none>
TECHNIQUE: <hyde|fielded|rerank|decompose|prf|none>
NEXT_QUERY: <a focused query for the missing sub-fact, or none>
CONFIDENCE: <0.0-1.0 confidence the set is COMPLETE>
VERDICT: <PASS|FAIL>"""

_WORD = re.compile(r"[a-z0-9]+")


def _tok(s):
    return {w for w in _WORD.findall((s or "").lower()) if len(w) > 2}


def score_signals(scores):
    """Scale-free shape of the score curve (ratios to the top score) + largest consecutive gap (cliff)."""
    if not scores:
        return {"top3_ratio": 0.0, "min_ratio": 0.0, "cliff": 0.0}
    import numpy as np
    s = sorted(scores, reverse=True)
    m = s[0] or 1.0
    norm = [x / m for x in s]
    gaps = [norm[i] - norm[i + 1] for i in range(len(norm) - 1)]
    return {"top3_ratio": round(float(np.mean(norm[:3])), 3), "min_ratio": round(norm[-1], 3),
            "cliff": round(max(gaps) if gaps else 0.0, 3)}


def coverage_signals(subfacts, sub_vecs, cand_texts, cand_vecs, reranker):
    """Per sub-fact: cross-encoder relevance (ce_best, primary), bi-encoder cosine (best_sim), and lexical
    overlap of the best candidate. `reranker(query, texts) -> list[float]`; vecs are L2-normalised arrays."""
    import numpy as np
    ctoks = [_tok(t) for t in cand_texts]
    out = []
    for i, sub in enumerate(subfacts):
        sims = (cand_vecs @ sub_vecs[i]) if len(cand_vecs) else np.array([0.0])
        stoks = _tok(sub)
        lex = max((len(stoks & ct) / (len(stoks) or 1) for ct in ctoks), default=0.0)
        ce = reranker(sub, cand_texts) if cand_texts else [-10.0]
        out.append({"subfact": sub[:90], "best_sim": round(float(sims.max()), 3),
                    "lexical_overlap": round(float(lex), 2), "ce_best": round(float(max(ce)), 2)})
    return out


def render(query, subfacts, candidates, coverage, sig):
    """`candidates`: list of {id, score(0..1), snippet}; `coverage`: coverage_signals output."""
    subs = "\n".join(f"{i}. {s}" for i, s in enumerate(subfacts, 1))
    cands = "\n".join(f"[{c['id']}] {c['score']:.2f}  {c['snippet'][:150]}" for c in candidates) or "(no results)"
    cov = "\n".join(f"{i}. ce={c.get('ce_best', 0.0):+.2f} sim={c['best_sim']:.2f} lex={c['lexical_overlap']:.2f}  ({c['subfact']})"
                    for i, c in enumerate(coverage, 1))
    return (f"QUESTION: {query}\n\nSUBFACTS (the question needs one document for each):\n{subs}\n\n"
            f"CANDIDATES (current top-{len(candidates)}, normalized score):\n{cands}\n\n"
            f"COVERAGE (best candidate match per sub-fact):\n{cov}\n\n"
            f"SCORE SIGNALS: top3_ratio={sig['top3_ratio']} min_ratio={sig['min_ratio']} cliff={sig['cliff']}\n\n"
            "Decide PASS (every sub-fact has a document) or FAIL (something is still missing).")


def parse_verdict(text):
    out = {"verdict": None, "confidence": 0.5, "missing": "", "diagnosis": "",
           "technique": "", "next_query": "", "covered": ""}
    for line in text.splitlines():
        s = line.strip()
        u = s.upper()
        val = s.split(":", 1)[-1].strip() if ":" in s else ""
        if u.startswith("VERDICT"):
            out["verdict"] = "PASS" if "PASS" in u else ("FAIL" if "FAIL" in u else None)
        elif u.startswith("CONFIDENCE"):
            try:
                out["confidence"] = float(val.split()[0])
            except Exception:
                pass
        elif u.startswith("MISSING"):
            out["missing"] = val
        elif u.startswith("DIAGNOSIS"):
            out["diagnosis"] = val.lower()
        elif u.startswith("TECHNIQUE"):
            out["technique"] = val.lower()
        elif u.startswith("NEXT_QUERY") or u.startswith("NEXT QUERY"):
            out["next_query"] = val
        elif u.startswith("COVERED"):
            out["covered"] = val
    if out["verdict"] is None:
        out["verdict"] = "PASS" if out["confidence"] >= 0.5 else "FAIL"
    out["pred_pass"] = int(out["verdict"] == "PASS")
    return out


class DiagnosticJudge:
    """Wraps a generator with `.complete(prompt, system=...)` into the diagnostic controller."""

    def __init__(self, generator, prompt: str = DIAGNOSTIC_PROMPT):
        self.gen = generator
        self.prompt = prompt

    def judge(self, query, subfacts, candidates, coverage):
        sig = score_signals([c["score"] for c in candidates])
        raw = self.gen.complete(render(query, subfacts, candidates, coverage, sig), system=self.prompt)
        v = parse_verdict(raw)
        v["raw"] = raw
        return v
