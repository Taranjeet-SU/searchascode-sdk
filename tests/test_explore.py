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
    s = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=32))
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

    s = sac.Session("memory", dim=32, generator=gen, embedder=sac.HashEmbedder(dim=32))
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

    s = sac.Session("memory", dim=32, generator=gen, embedder=sac.HashEmbedder(dim=32))
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

    s = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=32))
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
    assert m["n"] == len(labeled)
    assert 0.0 <= m["oracle_coverage"] <= 1.0
    assert m["n_templates"] == 16
    # SDK-A1: with label_llm=False / label_rerank=False most templates would degrade to a
    # duplicate of light_dense / light_hybrid, so they are reported as NOT EVALUATED rather
    # than scored (which used to hand light_dense every tie on cost, by construction).
    from search_as_code.explore.templates import available_templates
    usable = set(available_templates(use_llm=False, use_rerank=False))
    unusable = set(TEMPLATE_NAMES) - usable
    assert usable < set(TEMPLATE_NAMES), "labeling without an LLM/reranker is not the full space"
    # evaluated is a subset of available (cascade stops at the first solving cost group)…
    assert set(m["template_hit_rate@k"]) <= usable
    # …and no degenerate template is ever scored, nor counted as a miss.
    assert not (set(m["template_hit_rate@k"]) & unusable)
    assert unusable <= set(m["templates_not_evaluated"])
    assert pack_has(ex, "router_meta.json")
    if m.get("cv_accuracy") is not None:
        assert 0.0 <= m["cv_accuracy"] <= 1.0
        assert pack_has(ex, "router.pkl")
        # a fitted router predicts a known template name
        assert ex.route("part number for AGFC03") in TEMPLATE_NAMES


def pack_has(explorer, name):
    return explorer.pack.has(name)


def test_training_dataset_setmodel_train(tmp_path):
    s = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=32))
    s.add(_corpus())
    ex = explore(s, out=str(tmp_path / "pack"))

    labeled = []
    for i in range(8):
        labeled += [
            {"query": f"Agilex 7 transceiver detail {i}", "gold_id": f"p{i}"},
            {"query": f"Device AGFC0{i} logic elements", "gold_id": f"c{i}"},
            {"query": f"install Quartus step {i}", "gold_id": f"l{i}"},
        ]

    # atomic sharded dataset
    ds = ex.dataset(queries=labeled, label_llm=False, label_rerank=False,
                    batch_size=8, workers=2, progress_every=0)
    assert len(ds) == len(labeled)
    ddir = ex.pack.root / "dataset"
    assert (ddir / "queries.jsonl").exists()
    assert list((ddir / "shards").glob("feat_*.npy"))            # feature shards on disk
    assert (ddir / "checkpoint.json").exists()

    # swappable model head + train
    ex.set_model("logreg", C=0.5)
    m = ex.train(cv=3)
    assert m["n"] == len(labeled)
    assert 0.0 <= m["oracle_coverage"] <= 1.0
    if m.get("cv_accuracy") is not None:
        assert m["model"] == "logreg"
        assert ex.pack.has("router.pkl")

    # resume: rebuilding loads the shards, doesn't relabel (same length)
    ds2 = ex.dataset(queries=labeled, resume=True, batch_size=8, progress_every=0)
    assert len(ds2) == len(labeled)

    # CSV export: per-query recall + per-template recall summary
    from search_as_code.explore import TEMPLATE_NAMES, write_dataset_csv
    paths = write_dataset_csv(ex.pack)
    assert paths["rows"] == len(labeled)
    head = open(paths["labels"]).readline()
    assert "winner" in head and "hit_light_dense" in head
    trec = open(paths["template_recall"]).read().splitlines()
    assert trec[0].startswith("template,tier,cost,recall@k")
    assert len(trec) == len(TEMPLATE_NAMES) + 1


def test_winner_policy_recall_and_cheapest():
    from search_as_code.explore import TEMPLATE_COST, best_from_hits

    # no template solved -> none
    assert best_from_hits({}) == "none"
    assert best_from_hits({"deep_all": 0, "light_dense": 0}) == "none"
    # among solvers, the cheapest tier wins (recall@k gate, not rank)
    assert best_from_hits({"deep_all": 1, "light_dense": 1}) == "light_dense"
    assert TEMPLATE_COST["light_dense"] < TEMPLATE_COST["deep_all"]
    # a medium beats a deep when both solve
    assert best_from_hits({"deep_hyde_decompose": 1, "hyde_rerank": 1}) == "hyde_rerank"
    # only a deep solves -> deep is the label
    assert best_from_hits({"deep_all": 1}) == "deep_all"


def test_load_query_list_preserves_gold_set():
    # qrels items carry multiple relevant docs; the normalizer must keep the full set + dataset
    from search_as_code.explore.training import _load_query_list
    r = _load_query_list([{"query": "q", "gold_ids": ["a", "b"], "dataset": "nfcorpus"}])
    assert r[0]["gold_ids"] == ["a", "b"] and r[0]["gold_id"] == "a"
    assert r[0]["dataset"] == "nfcorpus"
    # single-gold and tuple forms still work
    assert _load_query_list([{"query": "q", "gold_id": "x"}])[0]["gold_ids"] == ["x"]
    assert _load_query_list([("q", "y")])[0]["gold_ids"] == ["y"]


def test_generate_multihop(tmp_path):
    import json as _json
    from search_as_code.explore import generate_multihop

    def gen(prompt):
        # the generator must see N documents and be asked for a multi-doc question
        assert "ALL" in prompt and "DOCUMENT 1" in prompt and "DOCUMENT 2" in prompt
        return [_json.dumps({"question": "which alpha relates to which beta?",
                             "facts": ["fact a", "fact b"]})]

    s = sac.Session("memory", dim=32, generator=gen, embedder=sac.HashEmbedder(dim=32))
    # docs that share keywords so keyword-neighbors form chains
    s.add([{"id": f"d{i}", "text": f"alpha beta gamma topic {i} shared keywords device fpga"}
           for i in range(30)])
    qs = generate_multihop(s, n_docs=2, target=5, workers=2, sample_chunk=20, progress_every=0)
    assert 1 <= len(qs) <= 5
    for q in qs:
        assert q["n_docs"] == 2 and len(q["gold_ids"]) == 2 and q["query"]
    # NONE responses are skipped, not forced
    qs2 = generate_multihop(s, n_docs=2, target=3, generator=lambda p: ["NONE"],
                            workers=2, sample_chunk=20, progress_every=0)
    assert qs2 == []


def test_make_model_registry():
    from search_as_code.explore import MODEL_REGISTRY, make_model
    assert {"hist_gb", "logreg", "random_forest", "mlp"} <= set(MODEL_REGISTRY)
    assert hasattr(make_model("hist_gb"), "fit")
    assert hasattr(make_model("logreg", C=0.1), "fit")


def test_fingerprint_detects_drift(tmp_path):
    s = _session()
    fp1 = corpus_fingerprint(s.store)
    s.add([{"id": "new1", "text": "a brand new document about PCIe retimers"}])
    fp2 = corpus_fingerprint(s.store)
    assert fp1 != fp2
