"""TRUE LLM primitive authoring (not composition) — with validate-and-retry.

The LLM WRITES a new retrieval primitive as Python code (`def run(session, query, top_k)`), which is
compiled, executed in a restricted sandbox (whitelisted builtins + imports), validated on a held
query (real errors fed back for a retry), and — only if it actually retrieves — registered + persisted
as a skill. This is the "true primitive creation" beyond composing existing retrievers.

NOTE: this executes model-generated code, so run it deliberately (a harness safety classifier may gate
ad-hoc execution). Sandbox = search_as_code/harness/forge.py:_safe_globals.

    python -m experiments.explore_forge.run_authored_primitive
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import search_as_code as sac
from phase1 import common
from phase1.llm import LLM
from search_as_code.harness import AgentMemory, HarnessForge, HarnessStore, SkillRegistry
from search_as_code.harness.forge import author_code_primitive

HERE = Path(__file__).parent
DATA = Path(__file__).parents[1] / "multi_hop_synth_queries" / "data"


def main():
    gen = LLM()
    mem = AgentMemory(path=str(HERE / "transparent_memory.jsonl"))
    patterns = "\n".join(m.content for m in mem.longterm if m.kind == "skill_win")
    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128).tolist()  # noqa: E731
    s = sac.Session("opensearch", index="hotpotqa", dim=common.DIM, hosts=[common.OS_HOST],
                    text_field="text", vector_field="vector", embedder=embed, generator=gen.as_generator())
    reg = SkillRegistry()
    store = HarnessStore(path=str(HERE / "transparent_store"))
    forge = HarnessForge(store, reg, mem)

    q = json.loads(open(DATA / "multihop_4docs_queries.jsonl").readline())
    print(f"test query: {q['query'][:80]} | golds: {len(q['gold_ids'])}")
    code, ok = author_code_primitive(gen, patterns, forge, s, q["query"], q["gold_ids"],
                                     name="llm_authored_multihop", tries=3)
    print(f"\nACCEPTED={ok}\n=== FINAL AUTHORED PRIMITIVE ===\n{code}")
    if ok:
        ids = reg.get("llm_authored_multihop").run(s, q["query"], top_k=10)
        g = set(map(str, q["gold_ids"]))
        print(f"\nre-exec: {len(ids)} ids, golds {len(g & set(map(str, ids)))}/{len(g)} ; "
              f"persisted={(HERE / 'transparent_store' / 'code_primitives.jsonl').exists()}")


if __name__ == "__main__":
    main()
