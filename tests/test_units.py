"""Unit coverage for modules the end-to-end tests don't exercise directly:
filters operators, the registry, embeddings, rerankers, ResultSet helpers,
sandbox safety, capability emulation, and the freshness primitive.
"""

import pytest

import search_as_code as sac
from search_as_code import errors
from search_as_code.filters import matches, normalize, validate
from search_as_code.primitives import freshness
from search_as_code.types import Capabilities, Document, Hit, ResultSet

CORPUS = [
    {"id": "d1", "text": "the quick brown fox", "metadata": {"lang": "en", "year": 2020}},
    {"id": "d2", "text": "reciprocal rank fusion merges lists", "metadata": {"lang": "en", "year": 2024}},
    {"id": "d3", "text": "le renard brun rapide", "metadata": {"lang": "fr", "year": 2021}},
]


# ---- filters ----------------------------------------------------------------
def test_filter_operator_coverage():
    md = {"year": 2023, "tag": "cve", "note": "has a fix"}
    assert matches(md, {})  # empty filter matches everything
    assert matches(md, {"tag": {"$exists": True}})
    assert matches(md, {"absent": {"$exists": False}})
    assert matches(md, {"note": {"$contains": "fix"}})
    assert matches(md, {"year": {"$in": [2022, 2023]}})
    assert matches(md, {"year": {"$nin": [1999]}})
    assert matches(md, {"$or": [{"year": {"$lt": 2000}}, {"tag": "cve"}]})
    assert matches(md, {"$and": [{"tag": "cve"}, {"year": {"$gt": 2023}}]}) is False
    assert matches(md, {"$not": {"tag": "advisory"}})
    assert not matches(md, {"tag": {"$ne": "cve"}})


def test_normalize_expands_shorthand():
    assert normalize({"lang": "en"}) == {"lang": {"$eq": "en"}}
    assert normalize(None) == {}
    assert normalize({"$or": [{"a": 1}]}) == {"$or": [{"a": {"$eq": 1}}]}


def test_validate_rejects_bad_structure():
    with pytest.raises(errors.InvalidFilterError):
        validate({"$and": {"not": "a list"}})
    with pytest.raises(errors.InvalidFilterError):
        validate({"field": {"$nope": 1}})
    validate({"$or": [{"a": {"$gte": 1}}], "$not": {"b": 2}})  # valid: no raise


# ---- registry ---------------------------------------------------------------
def test_registry_register_and_connect():
    from search_as_code.adapters import available, connect, register
    from search_as_code.adapters.memory import MemoryStore

    assert "memory" in available()
    register("mymem", MemoryStore)
    assert "mymem" in available()
    assert connect("mymem").capabilities().dense


# ---- embeddings -------------------------------------------------------------
def test_hash_embedder_dim_and_determinism():
    e = sac.HashEmbedder(dim=32)
    v = e.embed(["alpha beta gamma"])[0]
    assert len(v) == 32
    assert e.embed(["alpha beta gamma"])[0] == v


def test_as_embedder_wraps_callable():
    emb = sac.as_embedder(lambda ts: [[0.0] * 4 for _ in ts])
    assert emb.embed(["x", "y"]) == [[0.0] * 4, [0.0] * 4]


def test_get_embedder_hash():
    assert isinstance(sac.get_embedder("hash", dim=8), sac.HashEmbedder)


# ---- rerankers --------------------------------------------------------------
def test_cross_encoder_reranker_empty_is_model_free_noop():
    # returns before lazy-loading the model, so no torch/sentence-transformers needed
    assert sac.CrossEncoderReranker()("q", []) == []


# ---- ResultSet --------------------------------------------------------------
def test_resultset_helpers():
    rs = ResultSet([
        Hit(id="a", score=0.2, document=Document(id="a", text="hello", metadata={"k": 1})),
        Hit(id="b", score=0.9, document=Document(id="b", text="world", metadata={"k": 2})),
    ])
    assert rs.top(1).ids() == ["b"]
    assert rs.ids() == ["a", "b"]
    assert set(rs.texts()) == {"hello", "world"}
    assert rs.where(lambda h: h.get("k") == 2).ids() == ["b"]
    ev = rs.top(1).to_evidence(fields=["k"], max_chars=3)
    assert ev[0]["k"] == 2 and len(ev[0]["text"]) <= 3


def test_resultset_dedup_keeps_highest_score():
    rs = ResultSet([Hit(id="a", score=0.2), Hit(id="a", score=0.8)])
    deduped = rs.dedup()
    assert len(deduped) == 1 and deduped[0].score == 0.8


# ---- sandbox safety ---------------------------------------------------------
def test_sandbox_blocks_open_and_import():
    s = sac.Session("memory")
    s.add([{"id": "1", "text": "x"}])
    box = sac.LocalExecutor(s)
    r_open = box.run("evidence = open('/etc/passwd').read()")
    assert not r_open.ok and "NameError" in (r_open.error or "")
    r_import = box.run("import os\nevidence = os.listdir('/')")
    assert not r_import.ok  # __import__ is not in the safe builtins


# ---- capability emulation ---------------------------------------------------
def test_emulates_missing_modes_on_dense_only_backend():
    from search_as_code.adapters.memory import MemoryStore

    class DenseOnly(MemoryStore):
        backend = "denseonly"

        def capabilities(self):
            return Capabilities(dense=True, keyword=False, hybrid=False, regex=False,
                                metadata_filter=True)

        def query_keyword(self, *a, **k):  # must be emulated, not called
            raise AssertionError("keyword should be emulated in-SDK")

        def query_hybrid(self, *a, **k):
            raise AssertionError("hybrid should be emulated in-SDK")

        def query_regex(self, *a, **k):
            raise AssertionError("regex should be emulated in-SDK")

    s = sac.Session(DenseOnly())
    s.add(CORPUS)
    assert len(s.search("fusion", mode="keyword", top_k=3)) >= 1
    assert len(s.search("fusion", mode="hybrid", top_k=3)) >= 1
    assert s.search("fusion", mode="regex", top_k=3) is not None


# ---- freshness primitive ----------------------------------------------------
def test_freshness_prefers_recent():
    rs = ResultSet([Hit(id="old", score=1.0), Hit(id="new", score=1.0)])
    ts = {"old": 0.0, "new": 100.0}
    out = freshness(rs, timestamp=lambda h: ts[h.id], now=100.0, half_life=10.0, weight=0.9)
    assert out.ids()[0] == "new"
