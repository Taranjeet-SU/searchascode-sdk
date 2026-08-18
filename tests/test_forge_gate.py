"""The forge acceptance gate + provenance (FRG-1/3/4; fable.md WS3).

The dense-default gate was deleted once by a merge and nothing noticed, because it lived
only in an experiment script (issues.md FRG-1). These tests pin it in the SDK:
forge→accept, forge→reject→best-baseline fallback, archive-on-overwrite, rule supersession.
"""
from __future__ import annotations

import json

import search_as_code as sac
from search_as_code.harness.forge import HarnessForge, HarnessStore
from search_as_code.harness.skills import SkillRegistry


def _session():
    s = sac.Session("memory")
    s.add([{"id": "g1", "text": "the seattle pop festival happened in 1969 at woodinville"},
           {"id": "g2", "text": "the lifelight music festival is a christian event in sioux falls"},
           {"id": "d1", "text": "an unrelated document about cooking pasta"},
           {"id": "d2", "text": "another unrelated document about gardening tools"}])
    return s


HELD = [
    {"query": "seattle pop festival 1969", "gold_ids": ["g1"]},
    {"query": "lifelight music festival sioux falls", "gold_ids": ["g2"]},
    {"query": "festival woodinville 1969", "gold_ids": ["g1"]},
]

WINNING_CODE = (
    "def run(session, query, top_k):\n"
    "    return session.search(query, top_k=top_k, mode='hybrid').ids()\n"
)
LOSING_CODE = (
    "def run(session, query, top_k):\n"
    "    return session.search('cooking pasta gardening', top_k=top_k, mode='dense').ids()\n"
)


def _forge(tmp_path):
    store = HarnessStore(path=str(tmp_path / "store"))
    return HarnessForge(store, SkillRegistry()), store


def test_gate_accepts_a_candidate_that_ties_or_beats_and_records_provenance(tmp_path):
    forge, store = _forge(tmp_path)
    prov = forge.accept_code_primitive("forged_x", "test primitive", WINNING_CODE,
                                       session=_session(), held=HELD, k=4)
    assert set(prov) >= {"held_n", "candidate_mean", "baseline_means", "gate_baseline",
                         "delta_vs_baseline", "accepted", "created"}
    cp = store.code_primitives["forged_x"]
    assert cp.provenance["held_n"] == 3
    # reload round-trip keeps provenance (FRG-4)
    store2 = HarnessStore(path=str(tmp_path / "store"))
    assert store2.code_primitives["forged_x"].provenance["accepted"] == prov["accepted"]


def test_gate_rejects_a_losing_candidate_and_emits_the_best_baseline(tmp_path):
    forge, store = _forge(tmp_path)
    prov = forge.accept_code_primitive("forged_y", "test primitive", LOSING_CODE,
                                       session=_session(), held=HELD, k=4)
    assert prov["accepted"] is False
    cp = store.code_primitives["forged_y"]
    assert cp.provenance["fallback_of"] == prov["gate_baseline"]
    assert "rejected_code" in cp.provenance            # the losing candidate is kept, inspectable
    # the emitted primitive actually runs and behaves like the baseline
    sess = _session()
    ids = cp.to_skill().run(sess, "seattle pop festival 1969", top_k=4)
    assert "g1" in ids


def test_overwrite_archives_the_old_version(tmp_path):
    forge, store = _forge(tmp_path)
    forge.create_code_primitive("p", "v1", WINNING_CODE)
    forge.create_code_primitive("p", "v2", WINNING_CODE)
    sup = (tmp_path / "store" / "superseded.jsonl").read_text().splitlines()
    assert len(sup) == 1 and json.loads(sup[0])["when_to_use"] == "v1"
    assert "supersedes" in store.code_primitives["p"].provenance


def test_refine_prompt_supersedes_contradictory_rules(tmp_path):
    forge, store = _forge(tmp_path)
    forge.refine_prompt("structure = decompose (1/2 queries)")
    forge.refine_prompt("structure = whole-query (39/274 queries)", supersedes="structure =")
    assert store.learnings == ["structure = whole-query (39/274 queries)"]
    archived = (tmp_path / "store" / "superseded.jsonl").read_text()
    assert "decompose (1/2" in archived                # retired, not lost


def test_skill_overwrite_archives_and_versions(tmp_path):
    forge, store = _forge(tmp_path)
    forge.create_skill("s", "v1", ["dense"])
    forge.create_skill("s", "v2", ["dense", "hybrid"])
    assert store.skills["s"].when_to_use == "v2"
    assert "supersedes" in store.skills["s"].provenance
