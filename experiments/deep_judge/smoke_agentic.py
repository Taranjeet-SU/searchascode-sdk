"""Does agentic_solve DISCOVER the right structure per corpus (whole-query for BrowseComp, decompose for
HotpotQA) instead of hardcoding it? Prints the authored strategy + whether it decomposed, per query."""
from __future__ import annotations

import re
import sys

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from search_as_code.harness import agentic_solve


def _decomposed(code: str) -> bool:
    c = code.lower()
    return bool(re.search(r"split|\bfor .* in .*(parts|subq|sub_)|decompose|re\.split", c))


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "browsecomp"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()

    if corpus == "browsecomp":
        from experiments.browsecomp import bc_common
        store = sac.connect("opensearch", index="browsecomp", dim=common.DIM, hosts=[common.OS_HOST],
                            text_field="text", vector_field="vector")
        g, q = bc_common.load_golds(), bc_common.load_queries()
        rows = [{"query": q[k], "gold_ids": g[k]} for k in q if k in g][:n]
    else:
        import json
        from pathlib import Path
        store = sac.connect("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                            text_field="text", vector_field="vector")
        data = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data" / "multihop_4docs_queries.jsonl"
        rows = [json.loads(l) for l in data.open()][:n]

    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)
    dec = 0
    for r in rows:
        res = agentic_solve(session, r["query"], gold=r["gold_ids"], generator=gen, reranker=rr,
                            embedder=embed, judge_stop=False, max_hops=3)   # oracle-stop for scoring
        hop1 = res["codes"][0] if res["codes"] else ""
        d = _decomposed(hop1)
        dec += int(d)
        print(f"[{corpus}] recall={res['all_recall']:.2f} hops={res['hops']} decomposed={d} "
              f"| {r['query'][:55]}", flush=True)
    print(f"\n[{corpus}] n={len(rows)}: LLM DECOMPOSED in {dec}/{len(rows)} (expect LOW for browsecomp, HIGH for hotpot)")


if __name__ == "__main__":
    main()
