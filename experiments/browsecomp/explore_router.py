"""sac.explore template-router on BrowseComp — filling the gap (HotpotQA/SU got this, BrowseComp didn't).

Labels a sample of BrowseComp queries against the 16 templates and trains the router. Uses the
recall@10 (ANY-gold) gate (all_golds=False) because all_golds@10 over a 100k corpus with ~3 golds is
hopeless. Reports oracle coverage, label distribution, CV accuracy, and the failure taxonomy — the
honest question is "is there ANY routing headroom on this hard corpus?"

    python -m experiments.browsecomp.explore_router [n=150] [workers=4]
"""
from __future__ import annotations

import json
import random
import sys

import search_as_code as sac
from phase1.llm import LLM
from experiments.browsecomp import bc_common as B


class TruncRR:
    """Truncate candidate text before the cross-encoder (BrowseComp docs are ~33KB; CE sees 512 tok)."""
    def __init__(self, rr, n=2000):
        self.rr, self.n = rr, n

    def __call__(self, query, texts):
        return self.rr(query, [(t or "")[:self.n] for t in texts])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    golds = B.load_golds(); queries = B.load_queries()
    corpus_ids = set(json.loads(B.IDS_JSON.read_text()))
    eligible = [q for q, gs in golds.items()
                if q in queries and queries[q] and any(g in corpus_ids for g in gs)]
    random.seed(0); random.shuffle(eligible); sample = eligible[:n]
    labeled = [{"query": queries[q], "gold_ids": [g for g in golds[q] if g in corpus_ids]} for q in sample]
    print(f"[bc-explore] labeling {len(labeled)} of {len(eligible)} eligible queries", flush=True)

    gen = LLM(); session = B.load_session(generator=gen.as_generator())
    session.store.build_kw()
    session.reranker = TruncRR(sac.CrossEncoderReranker())

    pack_dir = str(B.HERE / "pack_browsecomp")
    ex = sac.explore(session, out=pack_dir)                       # init pack (sample/profile)
    ex.dataset(queries=labeled, all_golds=False, label_llm=True, label_rerank=True,
               workers=workers, batch_size=64, progress_every=10)
    ex.set_model("hist_gb", learning_rate=0.1, max_depth=8, max_iter=400)
    m = ex.train(cv=3)

    from search_as_code.explore import (load_dataset, best_from_hits, analyze_failures,
                                         write_dataset_csv, unsolved)
    from collections import Counter
    ds = load_dataset(ex.pack)
    winners = Counter(best_from_hits(json.loads(l).get("hits") or {})
                      for f in sorted((ex.pack.root / "dataset" / "shards").glob("lab_*.jsonl"))
                      for l in open(f))
    fails = analyze_failures(session, unsolved(ex.pack))   # (session, items) — classify the unsolved
    write_dataset_csv(ex.pack)

    out = {"n": len(labeled), "gate": "recall@10 (any-gold)",
           "oracle_coverage": ds.meta.get("oracle_coverage"),
           "cv_accuracy": m.get("cv_accuracy"), "cv_std": m.get("cv_std"),
           "label_distribution": dict(winners.most_common()),
           "failure_taxonomy": fails}
    (B.HERE / "bc_explore.json").write_text(json.dumps(out, indent=2))
    print("\n===== BrowseComp explore router =====")
    print(f"  oracle (any template solves@10, any-gold): {out['oracle_coverage']}")
    print(f"  router CV accuracy: {out['cv_accuracy']}")
    print(f"  winners: {dict(list(out['label_distribution'].items())[:6])}")
    print(f"  failure taxonomy: {fails}")
    print("saved bc_explore.json")


if __name__ == "__main__":
    main()
