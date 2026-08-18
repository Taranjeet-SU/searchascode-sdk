"""The agentic harness, the diagnostic judge, and the forge — in one runnable pass. Zero setup.

`examples/` shipped only `demo.py` and `opensearch_quickstart.py`, covering the oldest API, while
the README leads with `agentic_solve`, the `DiagnosticJudge` and the forge (issues.md EX-1). A new
user had no runnable path to any of them. This is that path.

    python examples/04_harness_judge_forge.py

Covers, in order:
  1. TRIAGE     — what kind of query is this, and how much effort does it deserve?
  2. SKILLS     — the registry, progressive disclosure, and picking a skill for a query
  3. HARNESS    — Plan-Execute-Verify, and why the reward you pass in decides what is learned
  4. JUDGE      — the stop/continue controller, and reading its diagnosis honestly
  5. FORGE      — turning a verified win into a persisted skill/subagent/rule, reloaded next run

Runs on the in-memory backend with a scripted generator, so no API key is required. Every step
prints the SDK call that produced it, so you can lift the lines straight into your own code.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import search_as_code as sac

DOCS = (
    [{"id": f"p{i}", "text": f"The Agilex 7 FPGA family supports high-bandwidth designs. "
                             f"Variant {i} targets data-center acceleration.",
      "metadata": {"title": f"Agilex 7 variant {i}"}} for i in range(8)]
    + [{"id": f"c{i}", "text": f"Device=AGFC0{i} | LEs={100000+i} | Transceivers=96 | PCIe=Gen5",
        "metadata": {"title": f"AGFC0{i}"}} for i in range(8)]
    + [{"id": f"l{i}", "text": f"Install Quartus step {i}: open project, compile design.",
        "metadata": {"title": f"Quartus step {i}"}} for i in range(8)]
)


def scripted_generator(prompt):
    if "sub-question" in prompt.lower() or "sub-questions" in prompt.lower():
        return ["What are the Agilex 7 transceiver counts?",
                "What are the Quartus install steps?"]
    return ["a short hypothetical passage answering the query"]


def session():
    s = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=32),
                    generator=scripted_generator)
    s.add(DOCS)
    return s


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    store_dir = Path(tempfile.mkdtemp(prefix="sac-harness-")) / "store"

    # ---------------------------------------------------------------- 1. TRIAGE
    rule("1. TRIAGE — sac.triage(query): what is this, and how deep should we go?")
    for q in ["What is reciprocal rank fusion?",
              "Device AGFC03 error transceiver count",
              "Compare the Agilex 7 transceivers and the Quartus install steps",
              "a film released in 1994 and directed by a female director and set in Paris"]:
        it = sac.triage(q)
        print(f"   {it.kind:14s} depth={it.depth:6s} skill={it.recommended_skill:18s} "
              f"conf={it.confidence:.2f}  :: {q[:52]}")
    print("\n   Note the last one: many constraints, ONE document. It stays depth=single —")
    print("   decomposing a conjunctive-constraint query hurts (issues.md SDK-A4).")

    # ---------------------------------------------------------------- 2. SKILLS
    rule("2. SKILLS — a registry with progressive disclosure")
    reg = sac.SkillRegistry()
    print(f"   {len(reg.names())} built-in skills. The agent first sees only summaries:")
    for name in list(reg.names())[:5]:
        print(f"     - {reg.get(name).summary()}")
    q = "Compare the Agilex 7 transceivers and the Quartus install steps"
    print(f"\n   reg.find({q[:40]!r}...) ->")
    for sk in reg.find(q, k=3):
        print(f"     - {sk.name} (cost {sk.cost})")

    # ---------------------------------------------------------------- 3. HARNESS
    rule("3. HARNESS — Plan-Execute-Verify. The reward you pass decides what gets learned.")
    s = session()

    print("   (a) with the DEFAULT verify — no relevance signal available:")
    h = sac.Harness(s, store_path=str(store_dir), learn=True)
    r = h.run(q, top_k=6)
    print(f"       skill={r.skill} score={r.score} verified={r.verified} ids={r.ids[:4]}")
    print(f"       forged: {r.meta.get('forged')}   long-term memories: {h.memory.stats()['longterm']}")
    print("       ^ nothing learned, ON PURPOSE. default_verify cannot see relevance, so the")
    print("         harness refuses to treat its score as evidence (issues.md SDK-A3).")

    print("\n   (b) with a REAL reward (gold here; a teacher reranker or the judge in production):")
    # Gold for THIS query. Kept to a document the retriever genuinely surfaces, so the demo
    # shows the forge firing; with unreachable gold the harness correctly forges nothing
    # (try gold = {"zzz"} to see the refusal).
    gold = {"p1"}

    def verify(ctx, ids):
        hit = len(gold & set(ids)) / len(gold)
        return (hit >= 1.0, hit)

    h2 = sac.Harness(session(), store_path=str(store_dir), learn=True, verify=verify)
    r2 = h2.run(q, top_k=6)
    print(f"       skill={r2.skill} score={r2.score:.2f} verified={r2.verified} ids={r2.ids[:4]}")
    print(f"       forged: {r2.meta.get('forged')}")

    # ---------------------------------------------------------------- 4. JUDGE
    rule("4. JUDGE — the stop/continue controller (sac.DiagnosticJudge)")
    print("   Per hop it coverage-checks EACH sub-fact and, for the ones still missing,")
    print("   diagnoses why and prescribes the next technique — which the next hop executes.")
    print("   It emits: COVERED / MISSING / DIAGNOSIS / TECHNIQUE / NEXT_QUERY / CONFIDENCE / VERDICT")
    from search_as_code.harness.diagnostic_judge import parse_verdict
    sample = ("COVERED: sf1\nMISSING: sf2\nDIAGNOSIS: vocab_gap\nTECHNIQUE: hyde\n"
              "NEXT_QUERY: transceiver count for the data-center variant\n"
              "CONFIDENCE: 0.4\nVERDICT: FAIL")
    print(f"\n   parse_verdict(...) -> {parse_verdict(sample)}")
    print("\n   Read it honestly: re-validated LEAK-FREE (query-grouped split, shipped renderer)")
    print("   the judge scores balanced accuracy 0.771 [0.666, 0.870] — above always-PASS (0.500),")
    print("   a tuned min-CE threshold (0.738) and a logistic gate (0.749), within CI at n=100.")
    print("   Its DIAGNOSIS (missing sub-fact / why / technique / next query) is passed to the")
    print("   NEXT hop's author, so a FAIL is a set of instructions, not just a verdict.")
    print()
    print("   The depth knobs are SDK params on agentic_solve / diagnostic_solve:")
    print("     judge_stop=True             # the judge decides when to stop (product mode)")
    print("     max_hops=5                  # the PRODUCT escalation cap: baseline hop-0,")
    print("                                 #   escalate only on judge-FAIL, judge every hop")
    print("     max_hops=10                 # what EXPLORE and oracle->judge tuning use —")
    print("                                 #   deeper, gold-stopped strategy discovery")
    print("     skill_lookup=SkillLookup(embed)   # 'when to call what' recipes per weak sub-fact")
    print("     memory=AgentMemory()        # in-hop findings + cross-query skill-win recall")

    # ---------------------------------------------------------------- 5. FORGE
    rule("5. FORGE — a verified win becomes a durable skill / subagent / rule")
    print(f"   store: {store_dir}")
    print(f"   skills          : {list(h2.store.skills)}")
    print(f"   subagents       : {list(h2.store.subagents)}")
    print(f"   code primitives : {list(h2.store.code_primitives)}")
    for ln in h2.store.learnings:
        print(f"   learned rule    : {ln[:96]}")

    print("\n   A NEW session loads them and uses them online:")
    h3 = sac.Harness(session(), store_path=str(store_dir))
    learned = [n for n in h3.skills.names() if n.startswith("learned")]
    print(f"     forged skills registered: {learned}")
    if learned:
        ids = h3.skills.get(learned[0]).run(h3.session, "Agilex 7 transceivers", top_k=5)
        print(f"     running the forged skill -> {ids}")
    print(f"     learned rules injected into the prompt: {bool(h3.store.learnings_block())}")

    print("\n   The LLM can also AUTHOR a genuinely new primitive as code — validated against a")
    print("   held query before acceptance — via harness.forge.author_code_primitive(...).")
    print("   That path needs a real LLM; see experiments/deep_judge/ for a live run.")

    rule("where to go next")
    print("   examples/03_explore_first.py   explore -> ProfilePack -> router -> agent hints")
    print("   docs/HARNESS.md                the harness design")
    print("   docs/EXPLORE.md                the exploration pipeline")
    print("   issues.md                      what these components do NOT yet do")


if __name__ == "__main__":
    main()
