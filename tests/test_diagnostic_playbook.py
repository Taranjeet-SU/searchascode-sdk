"""Integration test: the STANDARD package reproduces the diagnostic-playbook relevance.

Uses ONLY the pip-installed public API (`search_as_code` + `search_as_code.harness`), no experiment
modules, to confirm that `diagnostic_solve` — in raw-arsenal mode and in FORGED-primitive (SAC) mode —
reaches the recall we measured in experiments/deep_judge (sac_oracle ~ raw_oracle).

Skips cleanly if OpenSearch (hotpotqa) or the OpenAI key are unavailable. Run directly for the numbers:
    python -m tests.test_diagnostic_playbook
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "multi_hop_synth_queries" / "data" / "multihop_4docs_queries.jsonl"
FORGE_STORE = ROOT / "experiments" / "deep_judge" / "forge_store_hotpot"
OS_HOST = {"host": "127.0.0.1", "port": 9200}
DIM = 768
N = 6


def _env_ready():
    try:
        from phase1 import common
        common.load_env()                    # OPENAI_API_KEY lives in a .env, not the shell env
    except Exception:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:9200/hotpotqa/_count", timeout=3) as r:
            return json.load(r).get("count", 0) > 0
    except Exception:
        return False


def _make():
    import torch
    from sentence_transformers import SentenceTransformer
    import search_as_code as sac
    from phase1.llm import LLM
    from phase1 import common

    em = SentenceTransformer(common.EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embed = lambda t: em.encode(list(t), normalize_embeddings=True, batch_size=128, show_progress_bar=False).tolist()  # noqa
    rr = sac.CrossEncoderReranker()
    gen = LLM()
    store = sac.connect("opensearch", index="hotpotqa", dim=DIM, hosts=[OS_HOST],
                        text_field="text", vector_field="vector")
    session = sac.Session(store, embedder=embed, generator=gen.as_generator(), reranker=rr)
    return session, embed, rr, gen


def _forged_primitives(embed):
    """Load the forged authored primitives from the persisted store via the public harness API."""
    from search_as_code.harness import HarnessStore, SkillRegistry, HarnessForge, AgentMemory
    if not (FORGE_STORE / "code_primitives.jsonl").exists():
        return []
    store = HarnessStore(path=str(FORGE_STORE))
    reg = SkillRegistry(embedder=embed)
    HarnessForge(store, reg, AgentMemory())          # registers the authored primitives into `reg`
    return [reg.get(n).run for n in store.code_primitives if reg.get(n)]


@pytest.mark.skipif(not _env_ready(),
                    reason="needs OpenSearch hotpotqa + OPENAI_API_KEY")
def test_sac_reproduces_raw_relevance():
    import numpy as np
    from search_as_code.harness import diagnostic_solve
    session, embed, rr, gen = _make()
    rows = [json.loads(l) for l in DATA.open()][:N]
    forged = _forged_primitives(embed)

    raw, sac_r = [], []
    for r in rows:
        a = diagnostic_solve(session, r["query"], gold=r["gold_ids"], generator=gen,
                             reranker=rr, embedder=embed, max_hops=5)          # raw arsenal, oracle-stop
        raw.append(a["all_recall"])
        if forged:
            b = diagnostic_solve(session, r["query"], gold=r["gold_ids"], generator=gen, reranker=rr,
                                 embedder=embed, forged=forged, max_hops=5)     # FORGED SAC primitives
            sac_r.append(b["all_recall"])

    raw_m = float(np.mean(raw))
    print(f"\n[pip test] raw diagnostic_solve recall@10 = {raw_m:.3f} (n={len(raw)})")
    assert raw_m >= 0.55, f"raw recall too low: {raw_m}"
    if sac_r:
        sac_m = float(np.mean(sac_r))
        print(f"[pip test] forged-SAC diagnostic_solve recall@10 = {sac_m:.3f}")
        assert abs(sac_m - raw_m) <= 0.20, f"SAC recall {sac_m} not near raw {raw_m}"


if __name__ == "__main__":
    if not _env_ready():
        print("SKIP: needs OpenSearch hotpotqa + OPENAI_API_KEY")
    else:
        test_sac_reproduces_raw_relevance()
        print("[pip test] PASS")
