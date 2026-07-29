"""Tests for the sac.explore engine + ProfilePack."""

from __future__ import annotations

import search_as_code as sac
from search_as_code.explore import ProfilePack, corpus_fingerprint, explore


def _corpus():
    # three visibly different content shapes so clustering/profiling has signal
    prose = [{"id": f"p{i}", "text": f"The Agilex 7 FPGA family supports high-bandwidth "
              f"designs. Variant {i} targets data-center acceleration and scales well."}
             for i in range(8)]
    cards = [{"id": f"c{i}", "text": f"Device=AGFC0{i} | LEs={100000+i} | Transceivers=96 "
              f"| PCIe=Gen5 | Package=BGA"} for i in range(8)]
    lists = [{"id": f"l{i}", "text": f"- install Quartus {i}\n- open project\n- compile design"}
             for i in range(8)]
    return prose + cards + lists


def _session():
    s = sac.Session("memory", dim=32)
    s.add(_corpus())
    return s


def test_explore_runs_end_to_end(tmp_path):
    s = _session()
    pack = explore(s, out=str(tmp_path / "pack"))
    # implemented stages succeed
    assert pack.is_done("sample")
    assert pack.is_done("profile")
    # planned stub recorded as planned, pipeline not aborted
    assert pack.stage_status("ontology") == "planned"
    # synthesize needs a generator -> error here; validate then can't run -> skipped
    assert pack.stage_status("synthesize") == "error"
    assert pack.stage_status("validate") == "skipped"
    # artifacts exist
    assert pack.has("sample.jsonl")
    assert pack.has("content_profile.json")
    prof = pack.read_json("content_profile.json")
    assert prof["content_types"]  # some tags counted
    assert prof["n_sampled"] > 0
    assert "report" in pack.report().lower() or "ProfilePack" in pack.report()


def test_sample_is_stratified(tmp_path):
    s = _session()
    pack = explore(s, out=str(tmp_path / "pack"),
                   config={"pool_size": 24, "n_clusters": 3, "per_cluster": 2})
    rows = pack.read_jsonl("sample.jsonl")
    assert rows
    clusters = {r["cluster"] for r in rows}
    assert len(clusters) >= 2  # picked from multiple clusters, not one blob


def test_resumable_skips_done(tmp_path):
    s = _session()
    out = str(tmp_path / "pack")
    explore(s, out=out)
    ts1 = ProfilePack.open(out).stage("sample")["ts"]
    # second run without force must not re-run an already-done stage
    explore(s, out=out)
    ts2 = ProfilePack.open(out).stage("sample")["ts"]
    assert ts1 == ts2


def test_force_reruns(tmp_path):
    s = _session()
    out = str(tmp_path / "pack")
    explore(s, out=out)
    ts1 = ProfilePack.open(out).stage("sample")["ts"]
    explore(s, out=out, force=True)
    ts2 = ProfilePack.open(out).stage("sample")["ts"]
    assert ts2 >= ts1


def test_llm_profile_uses_generator(tmp_path):
    calls = {"n": 0}

    def gen(prompt):
        calls["n"] += 1
        return ["Curated FPGA fact-cards + prose. Entities: device, transceivers. "
                "Use keyword/regex for part-numbers, dense for prose."]

    s = sac.Session("memory", dim=32, generator=gen)
    s.add(_corpus())
    pack = explore(s, out=str(tmp_path / "pack"), config={"llm": True})
    prof = pack.read_json("content_profile.json")
    assert prof["llm_overall"]
    assert prof["llm_by_cluster"]
    assert calls["n"] > 0


def test_synthesize_and_validate(tmp_path):
    import json as _json

    def gen(prompt):
        if "JSON list" in prompt:                       # synthesize asks for queries
            return [_json.dumps([
                {"difficulty": "easy", "query": "Agilex 7 transceiver count"},
                {"difficulty": "medium", "query": "how many high-speed lanes does Agilex 7 have"},
                {"difficulty": "hard", "query": "which family supports 96 channels for data center"},
            ])]
        return ["Mixed FPGA corpus; use keyword for part numbers, dense for prose."]

    s = sac.Session("memory", dim=32, generator=gen)
    s.add(_corpus())
    pack = explore(s, out=str(tmp_path / "pack"),
                   config={"llm": True, "synth_docs": 5, "synth_per_doc": 3})

    assert pack.is_done("synthesize")
    q = pack.read_jsonl("synth_queries.jsonl")
    assert q and all("gold_id" in r and "query" in r for r in q)
    assert {r["difficulty"] for r in q} & {"easy", "medium", "hard"}

    assert pack.is_done("validate")
    v = pack.read_json("validation.json")
    assert v["best_overall"] in ("dense", "keyword", "hybrid")
    assert set(v["recall_at_k"]) == {"dense", "keyword", "hybrid"}
    assert pack.has("REPORT.md") and "recall@" in pack.path("REPORT.md").read_text()
    # router still planned (stub), codegen skipped (needs router)
    assert pack.stage_status("router") == "planned"


def test_templates_and_router_fit(tmp_path):
    from search_as_code.explore import TEMPLATE_NAMES
    from search_as_code.explore.templates import StrategyContext, run_template

    assert len(TEMPLATE_NAMES) == 16
    assert {"light_hybrid", "rephrase_rerank", "deep_hyde_decompose",
            "score_guarded", "escalating"} <= set(TEMPLATE_NAMES)

    s = sac.Session("memory", dim=32)
    s.add(_corpus())
    ex = explore(s, out=str(tmp_path / "pack"))

    # a strategy template produces ranked ids from a per-query context
    ctx = StrategyContext(s, "Agilex 7 transceivers", P_pool=10, use_llm=False, use_rerank=False)
    assert isinstance(run_template("light_dense", ctx, top_k=5), list)
    assert isinstance(run_template("score_guarded", ctx, top_k=5), list)

    # fit on explicit labeled queries (gold_id = a real doc id)
    labeled = []
    for i in range(8):
        labeled += [
            {"query": f"Agilex 7 transceiver detail {i}", "gold_id": f"p{i}"},
            {"query": f"Device AGFC0{i} logic elements", "gold_id": f"c{i}"},
            {"query": f"install Quartus step {i}", "gold_id": f"l{i}"},
        ]
    m = ex.fit(queries=labeled, rephrases=0, label_llm=False, label_rerank=False,
               progress_every=0)
    assert m["n_labeled"] == len(labeled)
    assert 0.0 <= m["oracle_coverage"] <= 1.0
    assert m["n_templates"] == 16
    assert set(m["template_hit_rate@k"]) == set(TEMPLATE_NAMES)
    assert pack_has(ex, "router_meta.json") and pack_has(ex, "router_labels.jsonl")
    if m.get("cv_accuracy") is not None:
        assert 0.0 <= m["cv_accuracy"] <= 1.0
        assert pack_has(ex, "router.pkl")
        # a fitted router predicts a known template name
        assert ex.route("part number for AGFC03") in TEMPLATE_NAMES


def pack_has(explorer, name):
    return explorer.pack.has(name)


def test_fingerprint_detects_drift(tmp_path):
    s = _session()
    fp1 = corpus_fingerprint(s.store)
    s.add([{"id": "new1", "text": "a brand new document about PCIe retimers"}])
    fp2 = corpus_fingerprint(s.store)
    assert fp1 != fp2
