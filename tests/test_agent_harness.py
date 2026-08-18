"""Tests for the agentic harness (no GPU / no LLM — memory backend + dependency-free embedder)."""

from __future__ import annotations

import search_as_code as sac
from search_as_code.harness import (
    AgentMemory, Harness, HarnessForge, HarnessStore, SkillRegistry,
    triage, extract_codes, decompose_query, fuse_ids,
)


def _gold_verify(gold):
    """A REAL reward signal (gold-based). Online learning refuses to learn from
    default_verify, which cannot see relevance — see issues.md SDK-A3."""
    def verify(ctx, ids):
        hit = len(set(gold) & set(ids)) / max(1, len(gold))
        return (hit >= 1.0, hit)
    return verify


def _session():
    s = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=32))
    s.add([{"id": f"p{i}", "text": f"The Agilex 7 FPGA family supports high-bandwidth designs. "
            f"Variant {i} targets data-center acceleration."} for i in range(8)]
          + [{"id": f"c{i}", "text": f"Device=AGFC0{i} | LEs={100000+i} | Transceivers=96 | PCIe=Gen5"}
             for i in range(8)]
          + [{"id": f"l{i}", "text": f"Install Quartus step {i}: open project, compile design."}
             for i in range(8)])
    return s


# ---- triage ---------------------------------------------------------------
def test_triage_error_code():
    it = triage("Device AGFC03 shows error, what is the transceiver count")
    assert it.kind == "error_code" and it.recommended_skill == "exact_lookup"
    assert "AGFC03" in extract_codes("Device AGFC03 transceiver count")


def test_triage_definition_and_multihop():
    assert triage("What is the Agilex 7 FPGA").kind == "definition"
    it = triage("Compare the Agilex 7 transceivers and the Quartus install steps")
    assert it.kind == "multi_hop" and it.depth == "multi" and it.recommended_skill == "decompose_arsenal"
    assert triage("who is the lead architect").kind == "entity_factoid"


def test_decompose_and_fuse():
    subs = decompose_query("the Agilex 7 transceivers and the Quartus install steps")
    assert len(subs) >= 2
    assert fuse_ids([["a", "b"], ["b", "c"]])[0] == "b"   # b appears in both -> ranked first


# ---- memory ---------------------------------------------------------------
def test_memory_in_session_and_cross_session(tmp_path):
    m = AgentMemory(path=str(tmp_path / "mem.jsonl"))
    m.observe("query: transceiver count", kind="query")
    m.remember("skill 'exact_lookup' worked for AGFC codes", kind="skill_win", skill="exact_lookup")
    assert m.stats()["longterm"] == 1
    hits = m.recall("AGFC transceiver", k=3)
    assert hits and "exact_lookup" in hits[0].content
    m2 = AgentMemory(path=str(tmp_path / "mem.jsonl"))     # cross-session: reload from disk
    assert m2.stats()["longterm"] == 1 and m2.recall("AGFC", k=1)


def test_memory_flush_promotes_working():
    m = AgentMemory()
    m.observe("outcome: dense_lookup got 5 hits", kind="outcome")
    m.observe("just a query", kind="query")
    n = m.flush()
    assert n == 1 and m.stats()["longterm"] == 1


# ---- skills ---------------------------------------------------------------
def test_skill_registry_progressive_disclosure_and_find():
    reg = SkillRegistry()
    assert {"dense_lookup", "decompose_fuse", "exact_lookup", "hyde_bridge"} <= set(reg.names())
    summ = reg.summaries()
    assert "decompose_fuse" in summ and "cost" in summ            # short catalog, not full detail
    found = reg.find("multi-hop question needing several documents", k=2)
    assert any(s.name == "decompose_fuse" for s in found)


def test_skill_runs_over_session():
    s = _session()
    reg = SkillRegistry()
    ids = reg.get("dense_lookup").run(s, "Agilex 7 transceivers", top_k=5)
    assert isinstance(ids, list) and 0 < len(ids) <= 5


# ---- harness (end-to-end) -------------------------------------------------
def test_harness_single_query():
    h = Harness(_session())
    r = h.run("What is the Agilex 7 FPGA", top_k=5)
    assert r.ids and r.intent == "definition"
    assert r.skill in h.skills.names()
    assert r.dynamic_prompt and "AVAILABLE SKILLS" in r.dynamic_prompt   # dynamic prompt assembled
    assert r.steps                                                       # control-loop trace


def test_harness_multihop_spawns_subagents():
    h = Harness(_session())
    r = h.run("Compare the Agilex 7 transceivers and the Quartus install steps", top_k=6)
    assert r.intent == "multi_hop" and r.skill == "subagents"
    assert len(r.subagents) >= 2 and all("ids" in sa for sa in r.subagents)  # real subagents ran
    assert r.ids                                                             # fused result


def test_harness_writes_memory_and_recalls_next_time():
    s = _session()
    h = Harness(s, verify=_gold_verify({"c3"}))
    h.run("Device AGFC03 error transceiver count", top_k=5)     # error_code -> should log a skill_win
    assert h.memory.stats()["longterm"] >= 1
    r2 = h.run("AGFC05 error status", top_k=5)                  # similar query recalls prior win
    assert "RELEVANT MEMORY" in r2.dynamic_prompt


def test_harness_pluggable_verify_reward():
    s = _session()
    gold = {"p1"}
    def verify(ctx, ids):                       # gold-based reward instead of the self-judge
        hit = len(gold & set(ids[:10])) / len(gold)
        return (hit >= 1.0, hit)
    h = Harness(s, verify=verify, max_steps=3)
    r = h.run("Agilex 7 variant 1 data-center acceleration", top_k=5)
    assert isinstance(r.score, float) and r.steps


def test_harness_cross_hop_findings():
    """Subagents share one memory; each writes its FINDING so later hops can recall it."""
    h = Harness(_session())
    h.run("Compare the Agilex 7 transceivers and the Quartus install steps", top_k=6)
    findings = [w for w in h.memory.working if w.kind == "finding"]
    assert len(findings) >= 2                                  # every subagent wrote a finding
    assert h.memory.working_context(kinds={"finding"})         # available to the next hop's prompt


def test_harness_cross_session_memory(tmp_path):
    """Long-term memory persists across Harness instances (sessions) via memory_path."""
    path = str(tmp_path / "hmem.jsonl")
    Harness(_session(), memory_path=path, verify=_gold_verify({"c3"})).run(
        "Device AGFC03 error transceiver count", top_k=5)
    h2 = Harness(_session(), memory_path=path, verify=_gold_verify({"c7"}))   # fresh 'session', same store
    assert h2.memory.stats()["longterm"] >= 1
    r = h2.run("AGFC07 error status", top_k=5)
    assert "RELEVANT MEMORY" in r.dynamic_prompt               # recalled the persisted win


# ---- self-improvement / forge (the paper's "second iteration") ------------
def test_forge_creates_and_registers_skill():
    reg = SkillRegistry()
    store = HarnessStore()
    forge = HarnessForge(store, reg)
    name = forge.create_skill("dense_kw_fused", "corpus with vocab gaps",
                              retrievers=["dense", "keyword"], combine="fuse")
    assert name in reg.names()                                 # usable online, same run
    s = _session()
    ids = reg.get("dense_kw_fused").run(s, "Agilex 7 transceivers", top_k=5)
    assert isinstance(ids, list) and ids                       # forged skill actually runs (composes)


def test_forge_refine_prompt_and_persist(tmp_path):
    store = HarnessStore(path=str(tmp_path / "hstore"))
    forge = HarnessForge(store, SkillRegistry())
    forge.refine_prompt("For error-code queries, use exact_lookup first.")
    forge.create_skill("learned_x", "x queries", retrievers=["dense", "hyde"])
    store2 = HarnessStore(path=str(tmp_path / "hstore"))        # reload = cross-session
    assert store2.learnings and "exact_lookup" in store2.learnings[0]
    assert "learned_x" in store2.skills


def test_harness_online_learning_end_to_end(tmp_path):
    """Solve → forge a skill + learned rule → persist → NEW session loads and uses it (online)."""
    sp = str(tmp_path / "store")
    verify = _gold_verify({"p0"})
    h = Harness(_session(), store_path=sp, learn=True, verify=verify)
    r = h.run("Compare the Agilex 7 transceivers and the Quartus install steps", top_k=6)
    assert r.meta.get("forged")                                # created artifacts from the solve
    assert any(n.startswith("learned_multihop") for n in h.store.skills)
    # a fresh harness (new session) loads the forged skill + learned rule and injects the rule
    h2 = Harness(_session(), store_path=sp, learn=True, verify=verify)
    assert any(n.startswith("learned_multihop") for n in h2.skills.names())   # forged skill online
    r2 = h2.run("Compare the transceivers and install steps", top_k=6)
    assert "LEARNED RULES" in r2.dynamic_prompt                # self-modifiable prompt in effect


def test_online_learning_refuses_to_learn_without_a_real_reward(tmp_path):
    """SDK-A3: with the default verify (no relevance signal) the harness must NOT forge
    skills or write "X worked" to long-term memory. Any non-empty result used to score 1.0,
    so every run forged a skill and a subagent from no evidence at all."""
    h = Harness(_session(), store_path=str(tmp_path / "store"), learn=True)
    r = h.run("Compare the Agilex 7 transceivers and the Quartus install steps", top_k=6)
    assert r.ids                                     # it still retrieves
    assert r.verified is False                       # but nothing verified it
    assert r.meta.get("forged") == []                # so nothing was forged
    assert h.memory.stats()["longterm"] == 0         # and no "skill X worked" was recorded


def test_plan_execute_verify_actually_iterates_without_a_real_reward():
    """SDK-A3: default_verify returned 1.0 for any non-empty list, so the loop always broke
    on the first skill and max_steps never bound."""
    from search_as_code.harness.context import HarnessContext
    from search_as_code.harness.loop import default_verify, plan_execute_verify

    ctx = HarnessContext(query="q", top_k=10)
    ctx.plan = ["a", "b", "c"]
    tried = []

    def execute(name):
        tried.append(name)
        return ["d1", "d2"]

    plan_execute_verify(ctx, execute, lambda ids: default_verify(ctx, ids), max_steps=3)
    assert tried == ["a", "b", "c"], f"loop stopped early at {tried} — max_steps never bound"
