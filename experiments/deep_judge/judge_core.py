"""The DIAGNOSTIC judge — a stop/continue controller that mimics the gold oracle without seeing gold.

Instead of a one-bit "looks good enough?" the judge (1) coverage-checks each decomposed sub-fact
against the candidate set using score + semantic + lexical signals, (2) for the first still-missing
sub-fact diagnoses WHY (vocab gap / named entity / buried / absent) and prescribes the retrieval
technique to fix it, and (3) emits VERDICT = PASS (all sub-facts covered -> stop) or FAIL (something
missing -> hop again). The PASS/FAIL is what we align to the oracle; the diagnosis is what makes the
next hop targeted instead of a blind rewrite.

`INITIAL_PROMPT` is v0 — `tune_judge.py` rewrites it round by round via a critic LLM.
"""
from __future__ import annotations

INITIAL_PROMPT = """You are the STOP/CONTINUE controller for a MULTI-HOP retrieval agent. A multi-hop \
question needs SEVERAL different documents — one per sub-fact. Decide whether the CURRENT result set \
already contains a strong document for EVERY sub-fact (VERDICT = PASS, stop) or whether at least one \
sub-fact's document is still missing (VERDICT = FAIL, do another retrieval hop). You do NOT see the \
gold answer — infer coverage from the signals.

You are given, for the current hop:
- SUBFACTS: the question split into the distinct documents it needs.
- CANDIDATES: the current top results (normalized score 0..1 + snippet).
- COVERAGE: per sub-fact, three signals about the BEST candidate for that sub-fact:
    * ce = a CROSS-ENCODER relevance score (the PRIMARY signal). It is calibrated: ce clearly POSITIVE \
(> ~0) means a candidate genuinely answers this sub-fact; ce strongly NEGATIVE (< ~ -3) means NO candidate \
does — that document is MISSING. ce near 0 / mildly negative is borderline.
    * sim = bi-encoder cosine (0..1) — WEAK/saturated here (even missing sub-facts sit ~0.8), so use it \
only to break ties, never as the main evidence.
    * lex = lexical term overlap (0..1).
- SCORE SIGNALS: top3_ratio / min_ratio / cliff (largest drop) of the score curve.

Decision rule of thumb: FAIL if ANY sub-fact's ce is clearly negative (no candidate answers it); PASS only \
when every sub-fact has a candidate with non-negative ce. Do not be fooled by one strongly-covered \
sub-fact — a multi-hop set is complete only if EVERY sub-fact is covered.

For the FIRST still-missing sub-fact (lowest ce), diagnose WHY and prescribe the next technique:
- vocab_gap  (only DESCRIBED generically — some relevance but low lexical overlap) -> hyde
- entity     (a NAMED entity that should match a title) -> fielded
- buried     (a strong match exists but is ranked low / there is a big cliff above it) -> rerank
- absent     (ce very negative for all — needs a different split or the doc is elsewhere) -> decompose

Reply on EXACTLY these lines, nothing else:
COVERED: <comma-separated sub-fact numbers that ARE satisfied, or none>
MISSING: <the single sub-fact number still missing, or none>
DIAGNOSIS: <vocab_gap|entity|buried|absent|none>
TECHNIQUE: <hyde|fielded|rerank|decompose|prf|none>
NEXT_QUERY: <a focused query for the missing sub-fact, or none>
CONFIDENCE: <0.0-1.0 that the set is COMPLETE>
VERDICT: <PASS|FAIL>"""


def render_example(e: dict) -> str:
    """The judge's user message — signals only, NEVER the gold."""
    subs = "\n".join(f"{i}. {s}" for i, s in enumerate(e["subfacts"], 1))
    cands = "\n".join(f"[{c['id']}] {c['score']:.2f}  {c['snippet'][:150]}" for c in e["candidates"]) \
        or "(no results)"
    cov = "\n".join(f"{i}. ce={c.get('ce_best', 0.0):+.2f} sim={c['best_sim']:.2f} lex={c['lexical_overlap']:.2f}  ({c['subfact']})"
                    for i, c in enumerate(e["coverage"], 1))
    sg = e["score_signals"]
    return (f"QUESTION: {e['query']}\n\n"
            f"SUBFACTS (the question needs one document for each):\n{subs}\n\n"
            f"CANDIDATES (current top-{len(e['candidates'])}, normalized score):\n{cands}\n\n"
            f"COVERAGE (best candidate match per sub-fact):\n{cov}\n\n"
            f"SCORE SIGNALS: top3_ratio={sg['top3_ratio']} min_ratio={sg['min_ratio']} cliff={sg['cliff']}\n\n"
            "Decide PASS (every sub-fact has a document) or FAIL (something is still missing).")


def parse_verdict(text: str) -> dict:
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
    if out["verdict"] is None:  # fall back on confidence if the line was malformed
        out["verdict"] = "PASS" if out["confidence"] >= 0.5 else "FAIL"
    return out


def run_judge(gen, prompt: str, e: dict) -> dict:
    """gen: phase1.llm.LLM. Returns parsed verdict dict + the raw text."""
    raw = gen.complete(render_example(e), system=prompt)
    v = parse_verdict(raw)
    v["raw"] = raw
    v["pred_pass"] = int(v["verdict"] == "PASS")
    return v


# ------------------------------------------------------------------ metrics
def confusion(preds: list[int], golds: list[int]) -> dict:
    tp = sum(p and g for p, g in zip(preds, golds))       # PASS & oracle-complete  (correct stop)
    tn = sum((not p) and (not g) for p, g in zip(preds, golds))  # FAIL & oracle-incomplete (correct hop)
    fp = sum(p and (not g) for p, g in zip(preds, golds))  # PASS but incomplete  (FALSE ACCEPT — plateau)
    fn = sum((not p) and g for p, g in zip(preds, golds))  # FAIL but complete    (false reject — wasted hop)
    n = len(preds) or 1
    pos = tp + fn or 1   # oracle-complete
    neg = tn + fp or 1   # oracle-incomplete
    tpr, tnr = tp / pos, tn / neg
    return {"n": len(preds), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round((tp + tn) / n, 3),
            "balanced_acc": round((tpr + tnr) / 2, 3),
            "false_accept_rate": round(fp / neg, 3),   # of the sets that were NOT complete, how many judge wrongly PASSed
            "false_reject_rate": round(fn / pos, 3)}
