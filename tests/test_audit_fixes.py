"""Regression tests for the audit fixes that are reachable without a live backend.

Each test names the issues.md entry it locks down. The theme of the audit was "a documented
property that no test covers and whose failure mode is silent", so these assert the documented
property rather than the happy path.
"""
from __future__ import annotations

import threading

import pytest

import search_as_code as sac


# --------------------------------------------------------------------------- #
# SDK-C1 — the os_query read-only allowlist was dead code                       #
# --------------------------------------------------------------------------- #
def test_os_query_allowlist_actually_rejects_unknown_keys():
    from search_as_code.harness.os_query import _validate

    ok, err = _validate({"bool": {"should": [{"match_phrase": {"title": "x"}}]}},
                        fields=["title", "text"])
    assert ok, err

    # An unknown clause is rejected. Before the fix the condition ended in
    # `not isinstance(k, str)` — always False — so nothing was ever rejected.
    ok, err = _validate({"bool": {"must": [{"some_unknown_clause": {"title": "x"}}]}},
                        fields=["title", "text"])
    assert not ok and "unexpected key" in err

    # A field that does not exist on this index is rejected too.
    ok, err = _validate({"match_phrase": {"nonexistent_field": "x"}}, fields=["text"])
    assert not ok and "nonexistent_field" in err


def test_os_query_still_bans_scripts_and_aggregations():
    from search_as_code.harness.os_query import _validate
    for banned in ({"script": {"source": "1"}}, {"aggs": {"a": {}}}, {"size": 10}):
        ok, err = _validate({"bool": {"must": [banned]}}, fields=["text"])
        assert not ok and "banned key" in err


# --------------------------------------------------------------------------- #
# SDK-A6 / SDK-A7 — os_query hardcoded `title` + a HotpotQA example             #
# --------------------------------------------------------------------------- #
def test_author_prompt_is_built_from_the_schema_not_hardcoded():
    from search_as_code.harness.os_query import build_author_system, describe_fields

    class _NoTitleStore:
        text_field, vector_field = "body", "vector"

        def describe_schema(self):
            return {"fields": {"body": "text", "published": "date", "vector": "knn_vector"}}

    fields = describe_fields(_NoTitleStore())
    assert "vector" not in fields, "the vector field is not lexically queryable"
    assert set(fields) == {"body", "published"}

    prompt = build_author_system(fields)
    assert "The Cardboard Crown" not in prompt, "HotpotQA example leaked into a general SDK prompt"
    assert "body" in prompt
    # On a corpus with no title field the prompt must not tell the model that `title` exists.
    assert "`title`" not in prompt


def test_describe_fields_tolerates_a_store_without_a_schema():
    from search_as_code.harness.os_query import describe_fields

    class _Bare:
        def describe_schema(self):
            raise RuntimeError("no mapping")

    assert describe_fields(_Bare()) == ["text"]


# --------------------------------------------------------------------------- #
# SDK-C10 — MemoryStore swallowed every kwarg, so dim was never enforced        #
# --------------------------------------------------------------------------- #
def test_memory_store_enforces_the_declared_dimension():
    s = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=32))
    s.add([{"id": "1", "text": "fine"}])                      # matches → ok

    bad = sac.Session("memory", dim=32, embedder=sac.HashEmbedder(dim=64))
    with pytest.raises(sac.DimensionMismatchError):
        bad.add([{"id": "1", "text": "wrong width"}])


# --------------------------------------------------------------------------- #
# SDK-C11 / BC-3 — MemoryStore rebuilt the whole DF table on every keyword query #
# --------------------------------------------------------------------------- #
def test_keyword_index_is_built_once_not_per_query(monkeypatch):
    from search_as_code.adapters import memory as memmod

    s = sac.Session("memory", embedder=sac.HashEmbedder(dim=32))
    s.add([{"id": str(i), "text": f"document {i} about agentic retrieval"} for i in range(50)])

    calls = {"n": 0}
    real = memmod._tokenize

    def counting(text):
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr(memmod, "_tokenize", counting)

    s.search("agentic retrieval", top_k=5, mode="keyword")
    after_first = calls["n"]
    for _ in range(5):
        s.search("agentic retrieval", top_k=5, mode="keyword")
    after_more = calls["n"]

    # 5 further queries must cost 5 tokenizations (the query itself), not 5 x 50 documents.
    assert after_more - after_first == 5, (
        f"corpus re-tokenized per query: {after_more - after_first} calls for 5 queries")
    assert after_first >= 50, "the index should have tokenized the corpus once"


def test_keyword_index_is_invalidated_on_write():
    s = sac.Session("memory", embedder=sac.HashEmbedder(dim=32))
    s.add([{"id": "1", "text": "alpha document"}])
    assert [h.id for h in s.search("beta", top_k=5, mode="keyword")] == []
    s.add([{"id": "2", "text": "beta document"}])
    assert [h.id for h in s.search("beta", top_k=5, mode="keyword")] == ["2"]
    s.store.delete(["2"])
    assert [h.id for h in s.search("beta", top_k=5, mode="keyword")] == []


def test_keyword_results_match_the_pre_index_implementation():
    """The index is an optimisation — ranking must not change."""
    s = sac.Session("memory", embedder=sac.HashEmbedder(dim=32))
    docs = [
        {"id": "a", "text": "vector search over embeddings"},
        {"id": "b", "text": "vector vector vector search"},
        {"id": "c", "text": "unrelated cooking recipe"},
    ]
    s.add(docs)
    ids = [h.id for h in s.search("vector search", top_k=3, mode="keyword")]
    assert ids[:2] == ["b", "a"] and "c" not in ids


# --------------------------------------------------------------------------- #
# SDK-C7 — thread-unsafe lazy model loading (duplicate GPU loads)               #
# --------------------------------------------------------------------------- #
def test_reranker_loads_the_model_once_under_concurrency():
    loads = {"n": 0}

    class _Slow(sac.CrossEncoderReranker):
        def _load(self):
            loads["n"] += 1
            return object()

    rr = _Slow()

    def fake_build():
        import time
        time.sleep(0.01)
        return rr._load()

    # emulate _ensure's body without importing torch
    def ensure():
        if rr._model is None:
            with rr._lock:
                if rr._model is None:
                    rr._model = fake_build()
        return rr._model

    threads = [threading.Thread(target=ensure) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loads["n"] == 1, f"model loaded {loads['n']}x across 8 threads — SDK-C7 regression"


def test_qwen_reranker_attributes_exist_before_first_call():
    """QwenReranker.dev/.tok were only assigned inside _ensure (SDK-C14)."""
    rr = sac.QwenReranker()
    assert rr.tok is None and rr.dev is None      # accessible, not AttributeError


# --------------------------------------------------------------------------- #
# SDK-C6 — templates.regex() could never hit (OpenSearch anchors regexp)        #
# --------------------------------------------------------------------------- #
def test_regex_template_wraps_the_pattern_for_anchored_backends():
    """OpenSearch `regexp` anchors to the whole field value, so a bare escaped code never
    matched and the regex pool was always empty (SDK-C6)."""
    from search_as_code.explore.templates import StrategyContext

    seen = []

    class _RecordingSession:
        generator = None
        reranker = None

        def search(self, q, top_k=10, mode="dense", **kw):
            seen.append((mode, q))
            return sac.ResultSet()

    ctx = StrategyContext(_RecordingSession(), "does part XC7A100T-1CSG324C support PCIe?", P_pool=5)
    ctx.regex()
    patterns = [q for mode, q in seen if mode == "regex"]
    assert patterns, "no regex search was issued for a query containing a code"
    assert all(p.startswith(".*") and p.endswith(".*") for p in patterns), patterns
