"""Explore-first: let the corpus tell you the retrieval strategy. Zero setup.

The README calls explore "the default workflow" but the only runnable version lived in
`experiments/deep_judge/run_explore_pipeline.py`, which is not shipped in the wheel — so a
`pip install` user could not run the documented headline workflow at all (issues.md EX-1).
This is that workflow, in the package, over the in-memory backend.

    python examples/03_explore_first.py

What it shows, in order:
  1. introspect BEFORE writing retrieval code   (session.describe)
  2. explore the corpus -> a versioned ProfilePack
  3. label queries against the strategy templates and train a router
  4. the metric that matters: REALIZED RECALL, not classification accuracy
  5. what explore hands to an agent: a per-query plan + corpus-grounded exemplars

No API key needed: a tiny scripted "generator" stands in for the LLM so the whole pipeline
runs offline. Point `generator=` at a real LLM to see the genuine version.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import search_as_code as sac

# --------------------------------------------------------------------------- #
# a small corpus with two clearly different shapes: prose pages and fact-cards  #
# --------------------------------------------------------------------------- #
DOCS = []
for i in range(12):
    DOCS.append({"id": f"guide{i}",
                 "text": f"Getting started guide {i}. This page explains how to configure the "
                         f"indexing pipeline and tune retrieval quality for tenant {i}.",
                 "metadata": {"title": f"Guide {i}", "kind": "prose"}})
for i in range(12):
    DOCS.append({"id": f"card{i}",
                 "text": f"Device=AGFC{i:03d} | Transceivers={24 + i} | PCIe=Gen5 | Power={10 + i}W",
                 "metadata": {"title": f"AGFC{i:03d}", "kind": "factcard"}})

# Queries with their gold answer. Real runs generate these with `explore.generate_multihop`
# or the synthesize stage; here they are explicit so the example is deterministic.
LABELLED = (
    [{"query": f"how do I configure indexing for tenant {i}", "gold_id": f"guide{i}"} for i in range(12)]
    + [{"query": f"transceiver count for AGFC{i:03d}", "gold_id": f"card{i}"} for i in range(12)]
)


def scripted_generator(prompt):
    """Stands in for an LLM so this example runs with no API key.

    Order matters: the synthesize prompt contains the word "paraphrased", so its branch has to
    be checked before the paraphrase branch. (A real LLM has no such problem — this is a quirk
    of the stub, kept as a reminder that prompt-sniffing stubs are order-sensitive.)
    """
    import json as _json
    import re as _re

    if "search questions whose" in prompt:                      # SynthesizeStage
        m = _re.search(r"DOCUMENT:\n(.*)", prompt, _re.DOTALL)
        body = " ".join((m.group(1) if m else "").split())[:70]
        return [_json.dumps([
            {"difficulty": "easy", "query": f"what describes {body}"},
            {"difficulty": "medium", "query": f"explain the setup in {body[:40]}"},
        ])]
    if "profiling a search corpus" in prompt:                   # describe(llm=True) / ProfileStage
        return ["(1) mixed: prose guides plus structured fact-cards",
                "(2) key entities: device part numbers (AGFCxxx), tenant ids",
                "(3) exact/keyword for part-numbers, dense for the prose guides"]
    if "sub-question" in prompt or "sub-questions" in prompt:   # decompose
        return ["what is the device", "what is the transceiver count"]
    if "different ways" in prompt:                              # rephrase / expand
        return ["rephrased variant one", "rephrased variant two"]
    return ["a short hypothetical passage answering the query"]  # HyDE


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="sac-explore-")) / "pack"

    s = sac.Session("memory", dim=64, embedder=sac.HashEmbedder(dim=64),
                    generator=scripted_generator)
    s.add(DOCS)

    # 1. INTROSPECT FIRST — schema-first agentic retrieval (docs/INTROSPECTION.md).
    print("=" * 72)
    print("1. introspect the corpus BEFORE writing retrieval code")
    profile = s.describe(n_samples=4, llm=True)
    print(f"   backend={profile.get('backend')}  content types={profile.get('content_types')}")
    print("   LLM profile:")
    for line in str(profile.get("llm", "")).splitlines():
        print(f"     {line}")

    # 2. EXPLORE — a resumable, versioned pass that writes a ProfilePack.
    print("\n" + "=" * 72)
    print("2. explore -> ProfilePack")
    explorer = sac.explore(s, out=str(out))
    for name in ("sample", "profile", "synthesize", "validate"):
        st = explorer.pack.stage(name) if hasattr(explorer.pack, "stage") else None
        print(f"   stage {name:12s} {(st or {}).get('status', '(see manifest)')}")

    # 3. LABEL + TRAIN. label_llm/label_rerank matter: without them most templates degrade
    #    into duplicates of light_dense, so labeling scores ~4 real strategies (SDK-A1).
    print("\n" + "=" * 72)
    print("3. label every query against the strategy templates, then train the router")
    metrics = explorer.fit(queries=LABELLED, rephrases=0,
                           label_llm=False, label_rerank=False, progress_every=0)
    print(f"   queries labelled : {metrics['n']}")
    print(f"   oracle coverage  : {metrics['oracle_coverage']}")
    print(f"   winners          : {metrics.get('label_distribution')}")
    print(f"   NOT evaluated    : {metrics.get('templates_not_evaluated')}")
    print("     ^ templates that would degrade to a duplicate without an LLM/reranker are")
    print("       reported as unavailable rather than scored as misses (issues.md SDK-A1).")

    # 4. THE METRIC THAT MATTERS. open_problems.md #3: CV classification accuracy is
    #    misleading for routing; realized recall is the task metric.
    print("\n" + "=" * 72)
    print("4. realized recall — routing vs the best always-one-template baseline")
    rr = metrics.get("realized_recall")
    if rr:
        print(f"   routed              {rr['routed']}")
        print(f"   best fixed template {rr['best_fixed']}  ({rr['best_fixed_template']})")
        print(f"   lift over it        {rr['lift_over_best_fixed']:+}")
        print(f"   oracle ceiling      {rr['oracle']}")
    else:
        print("   (too few solved queries/classes to train on this toy corpus)")
    print(f"   cv_accuracy is diagnostic only: {metrics.get('primary_metric')}")

    # 5. WHAT AN AGENT CONSUMES.
    print("\n" + "=" * 72)
    print("5. what explore hands to a code-mode agent")
    q = "transceiver count for AGFC003"
    try:
        print(f"   route({q!r}) -> {explorer.route(q)}")
        print(f"   plan  -> {explorer.plan_prompt(q)[:150]}...")
    except Exception as e:
        print(f"   (router not fitted on this toy corpus: {e})")
    block = explorer.fewshot_block(per_template=1, max_templates=3)
    print(f"   fewshot block ({len(block)} chars) — inject into the agent prompt:")
    for line in block.splitlines()[:4]:
        print(f"     {line}")

    print(f"\npack written to {out}")
    print("Next: examples/04_agentic_harness.py runs the agent that consumes this.")


if __name__ == "__main__":
    main()
