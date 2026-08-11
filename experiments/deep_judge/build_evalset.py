"""Build a FROZEN, oracle-labeled eval set for the diagnostic judge.

The judge's job is to mimic the gold ORACLE ("are all the documents this multi-hop question needs
already in the result set?") WITHOUT seeing the gold. To compare judge-prompt variants fairly we
freeze the candidate result sets once here, so every variant is scored on identical inputs.

For each of N multi-hop HotpotQA synth queries (gold known), we build TWO candidate states that give
a natural spread of oracle labels:
  - shallow : one plain hybrid search over the whole question (usually under-retrieves multi-hop).
  - deep    : the arsenal — decompose × {hybrid, HyDE, fielded}, RRF-fused (reaches described entities).

Each (query, state) becomes ONE example with the signals a real diagnostic judge would read:
  - candidates: [id] normalized_score snippet   (scores are score/max, so 0..1 within the set)
  - score signals: max/top3/min + the largest consecutive gap (score cliff)
  - decomposition: the LLM's sub-facts (no gold used)
  - coverage matrix: per sub-fact, the best candidate semantic sim + lexical overlap  (the signal that
    lets the judge say "sub-fact 3 has no match above 0.4 and zero lexical overlap -> missing, vocab gap")
  - oracle_pass: all gold ids in top-10   (SCORING ONLY — never shown to the judge)

    python -m experiments.deep_judge.build_evalset [n=100] [workers=8]
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from search_as_code import primitives as P
from phase1 import common
from phase1.llm import LLM

HERE = Path(__file__).parent
DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
# Weight toward the multi-doc regime where a stop-signal actually matters (2-hop comparison queries
# name every entity in the question -> trivially retrieved -> nothing for a judge to decide).
HOP_WEIGHTS = {2: 0.2, 3: 0.4, 4: 0.4}
K_DEEP = 10        # strong second hop returns a full window
K_SHALLOW = 5      # a cheap first hop returns few candidates -> genuinely-partial coverage
_WORD = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> set[str]:
    return {w for w in _WORD.findall((s or "").lower()) if len(w) > 2}


def _snip(text: str, n: int = 220) -> str:
    return " ".join((text or "").split())[:n]


def _score_signals(scores: list[float]) -> dict:
    """Scale-free shape of the score curve (ratios to the top score), so shallow (hybrid) and deep
    (RRF) states are comparable: top3_ratio/min_ratio in 0..1, and the largest consecutive gap (cliff)."""
    if not scores:
        return {"top3_ratio": 0.0, "min_ratio": 0.0, "cliff": 0.0}
    s = sorted(scores, reverse=True)
    m = s[0] or 1.0
    norm = [x / m for x in s]
    gaps = [norm[i] - norm[i + 1] for i in range(len(norm) - 1)]
    return {"top3_ratio": round(float(np.mean(norm[:3])), 3), "min_ratio": round(norm[-1], 3),
            "cliff": round(max(gaps) if gaps else 0.0, 3)}


def arsenal(session, subfacts: list[str], k: int = 30) -> P.ResultSet:
    """decompose × {hybrid, hyde, fielded} RRF-fused — scored ResultSet (fusion preserves coverage)."""
    sets = []
    fielded = getattr(session.store, "query_fielded", None)
    for sub in subfacts:
        sets.append(session.search(sub, top_k=k, mode="hybrid"))
        try:
            sets.append(session.hyde_search(sub, top_k=k))
        except Exception:
            pass
        try:
            sets.append(P.ResultSet(fielded(sub, ["title", "text"], top_k=k)) if fielded
                        else session.search(sub, top_k=k, mode="keyword"))
        except Exception:
            pass
    sets = [s for s in sets if s is not None and len(s)]
    return P.rrf(sets) if sets else session.search(" ".join(subfacts), top_k=k, mode="hybrid")


def coverage_matrix(subfacts, sub_vecs, cand_texts, cand_vecs):
    """Per sub-fact: best semantic sim to any candidate + best lexical (Jaccard-ish) overlap."""
    out = []
    ctoks = [_tok(t) for t in cand_texts]
    for i, sub in enumerate(subfacts):
        sims = (cand_vecs @ sub_vecs[i]) if len(cand_vecs) else np.array([0.0])
        stoks = _tok(sub)
        lex = max((len(stoks & ct) / (len(stoks) or 1) for ct in ctoks), default=0.0)
        out.append({"subfact": sub[:90], "best_sim": round(float(sims.max()), 3),
                    "lexical_overlap": round(float(lex), 2)})
    return out


def make_example(session, embed, gen, q, gold, titles, state, subfacts, sub_vecs):
    if state == "shallow":
        rs = session.search(q, top_k=K_SHALLOW, mode="hybrid")
        win = K_SHALLOW
    else:
        rs = arsenal(session, subfacts, k=30)
        win = K_DEEP
    hits = list(rs)[:win]
    ids = [str(h.id) for h in hits]
    texts = [_snip(h.text) for h in hits]
    scores = [float(h.score) for h in hits]
    cand_vecs = np.asarray(embed(texts), dtype=np.float32) if texts else np.zeros((0, common.DIM), np.float32)
    m = max(scores) if scores else 1.0
    cands = [{"id": i, "score": round(s / (m or 1.0), 3), "snippet": t}
             for i, s, t in zip(ids, scores, texts)]
    gold = [str(g) for g in gold]
    return {
        "query": q, "state": state, "n_docs": len(gold),
        "gold_ids": gold, "titles": titles,
        "candidates": cands,
        "score_signals": _score_signals(scores),
        "subfacts": subfacts,
        "coverage": coverage_matrix(subfacts, sub_vecs, texts, cand_vecs),
        "oracle_pass": int(set(gold) <= set(ids)),
        "oracle_missing": [g for g in gold if g not in ids],
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = []
    for hop, w in HOP_WEIGHTS.items():
        cnt = round(n * w)
        rs = [json.loads(l) for l in (DATA / f"multihop_{hop}docs_queries.jsonl").open()][:cnt]
        rows += rs
    print(f"[evalset] {len(rows)} queries (weights {HOP_WEIGHTS}) · shallow top-{K_SHALLOW} / deep top-{K_DEEP}",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128,  # noqa: E731
                                show_progress_bar=False).tolist()
    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed,
                          generator=gen.as_generator())

    lock = threading.Lock()
    examples, done = [], {"n": 0}

    def one(r):
        q, gold = r["query"], r["gold_ids"]
        titles = r.get("titles", [""] * len(gold))
        subs = P.decompose(q, gen.as_generator()) or [q]
        subs = [s for s in subs if s.strip()][:6] or [q]
        sub_vecs = np.asarray(embed(subs), dtype=np.float32)
        exs = [make_example(session, embed, gen, q, gold, titles, st, subs, sub_vecs)
               for st in ("shallow", "deep")]
        with lock:
            examples.extend(exs)
            done["n"] += 1
            if done["n"] % 10 == 0:
                pas = sum(e["oracle_pass"] for e in examples)
                print(f"  {done['n']}/{len(rows)} queries · {len(examples)} examples · "
                      f"oracle_pass={pas}/{len(examples)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, r) for r in rows]))

    out = HERE / "evalset.jsonl"
    with out.open("w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    pas = sum(e["oracle_pass"] for e in examples)
    by = {}
    for e in examples:
        k = (e["state"], e["n_docs"])
        by.setdefault(k, [0, 0])
        by[k][0] += e["oracle_pass"]
        by[k][1] += 1
    print(f"\n[evalset] wrote {out} · {len(examples)} examples · oracle_pass={pas} "
          f"({pas / len(examples):.0%})")
    for k in sorted(by):
        print(f"   {k[0]:8s} {k[1]}-hop: pass {by[k][0]}/{by[k][1]}")


if __name__ == "__main__":
    main()
