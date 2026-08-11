"""Trace the diagnostic loop on a few 4-hop queries: per hop show fused coverage, what the judge
diagnoses (missing sub-fact + technique + next query), whether the applied technique adds the missing
gold, and the running all-golds count. Reveals whether targeting helps or the fusion dilutes hop-1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from search_as_code import primitives as P
from phase1 import common
from phase1.llm import LLM
from experiments.deep_judge.judge_core import INITIAL_PROMPT, parse_verdict, render_example
from experiments.deep_judge.run_playbook import arsenal_lists, apply_technique, live_example, rrf_ids

DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"
HERE = Path(__file__).parent


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_hops = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rows = [json.loads(l) for l in (DATA / "multihop_4docs_queries.jsonl").open()][:n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    em = SentenceTransformer(common.EMB_MODEL, device=dev)
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128, show_progress_bar=False).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    session = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                          text_field="text", vector_field="vector", embedder=embed, generator=gen.as_generator())
    bp = HERE / "best_prompt_ce_same.txt"
    judge_prompt = bp.read_text() if bp.exists() else INITIAL_PROMPT

    for r in rows:
        q, gold = r["query"], set(str(g) for g in r["gold_ids"])
        gmap = {str(g): t for g, t in zip(r["gold_ids"], r.get("titles", [""] * len(r["gold_ids"])))}
        print("\n" + "=" * 100)
        print("Q:", q[:150])
        subfacts = [s for s in (P.decompose(q, gen.as_generator()) or [q]) if s.strip()][:6] or [q]
        print("SUBFACTS:", [s[:55] for s in subfacts])
        sub_vecs = np.asarray(embed(subfacts), dtype=np.float32)
        lists = arsenal_lists(session, subfacts)
        fused, scoremap = rrf_ids(lists)
        for hop in range(1, max_hops + 1):
            got = gold & set(fused[:10])
            missing_gold = {g: gmap[g] for g in gold - got}
            print(f"\n--HOP {hop}-- all-golds {len(got)}/{len(gold)} in top-10 | missing: "
                  f"{[t[:30] for t in missing_gold.values()]}")
            if len(got) == len(gold) or hop == max_hops:
                print("  STOP" if len(got) == len(gold) else "  (max hops)")
                break
            ex = live_example(session, embed, rr, q, subfacts, sub_vecs, fused, scoremap)
            print("  coverage(ce):", [f"{c['ce_best']:+.1f}" for c in ex["coverage"]])
            v = parse_verdict(gen.complete(render_example(ex), system=judge_prompt))
            print(f"  JUDGE verdict={v['verdict']} missing_sf={v['missing']} tech={v['technique']} "
                  f"nq='{(v['next_query'] or '')[:55]}'")
            nq = v["next_query"] or (subfacts[int(v["missing"]) - 1] if (v["missing"] or "").isdigit()
                                     and int(v["missing"]) <= len(subfacts) else q)
            tech = v["technique"] or "hyde"
            new = apply_technique(session, rr, tech, nq or q, fused)
            new_gold = (gold - got) & set(new[:10])
            print(f"  applied {tech} -> {len(new)} ids; added missing gold in top-10: "
                  f"{[gmap[g] for g in new_gold]}")
            lists.append(new)
            fused, scoremap = rrf_ids(lists)


if __name__ == "__main__":
    main()
