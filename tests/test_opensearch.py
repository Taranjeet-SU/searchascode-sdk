"""Integration tests for the OpenSearch adapter.

Skipped automatically unless a local OpenSearch is reachable on :9200, so the
default `pytest` run (unit tests over the in-memory backend) still works with no
services. To run these:  start OpenSearch, then `pytest tests/test_opensearch.py`.
"""

import time

import pytest

import search_as_code as sac

HOSTS = [{"host": "127.0.0.1", "port": 9200}]
INDEX = "sac_pytest"


def _opensearch_up() -> bool:
    try:
        from opensearchpy import OpenSearch

        return OpenSearch(hosts=HOSTS, use_ssl=False).ping()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _opensearch_up(),
                                reason="no local OpenSearch on :9200")


@pytest.fixture()
def session():
    s = sac.Session("opensearch", index=INDEX, dim=256, hosts=HOSTS)
    s.store.client.indices.delete(index=INDEX, ignore=[404])
    s.store.ensure_index(256)
    s.add([
        {"id": "a", "text": "def search(query): return db.query(query)", "metadata": {"kind": "code", "year": 2024}},
        {"id": "b", "text": "reciprocal rank fusion merges ranked lists", "metadata": {"kind": "doc", "year": 2023}},
        {"id": "c", "text": "the QueryEngine class exposes a search method", "metadata": {"kind": "code", "year": 2024}},
    ])
    time.sleep(1)
    yield s
    s.store.client.indices.delete(index=INDEX, ignore=[404])


def test_capabilities(session):
    caps = session.store.capabilities()
    assert caps.dense and caps.keyword and caps.hybrid and caps.regex and caps.metadata_filter


def test_count_and_get(session):
    assert session.store.count() == 3
    docs = session.store.get(["a", "b"])
    assert {d.id for d in docs} == {"a", "b"}


def test_dense_search(session):
    hits = session.search("how do agents search", top_k=3)
    assert len(hits) >= 1
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # larger-is-better, sorted


def test_keyword_search(session):
    assert session.search("reciprocal rank fusion", top_k=3, mode="keyword").ids()[0] == "b"


def test_hybrid_search(session):
    ids = session.search("search method", top_k=3, mode="hybrid").ids()
    assert ids and len(ids) == len(set(ids))


def test_regex_search(session):
    assert session.search(r".*def search.*", top_k=3, mode="regex").ids() == ["a"]


def test_metadata_filter(session):
    ids = session.search("search", top_k=5, mode="keyword", filter={"kind": "code"}).ids()
    assert "b" not in ids
    ids2 = session.search("search", top_k=5, mode="keyword", filter={"year": {"$gte": 2024}}).ids()
    assert "b" not in ids2


def test_aggregation(session):
    agg = session.store.aggregate({"by_kind": {"terms": {"field": "kind.keyword"}}})
    counts = {b["key"]: b["doc_count"] for b in agg["by_kind"]["buckets"]}
    assert counts == {"code": 2, "doc": 1}


# ---- extended lexical primitives -------------------------------------------
def test_phrase_search(session):
    # ordered phrase: only b has "ranked lists"
    assert session.store.query_phrase("ranked lists", top_k=5).ids() == ["b"]
    # slop lets non-adjacent terms match; "reciprocal ... lists" are 5 apart
    assert session.store.query_phrase("reciprocal lists", top_k=5, slop=0).ids() == []
    assert "b" in session.store.query_phrase("reciprocal lists", top_k=5, slop=5).ids()


def test_fielded_search(session):
    ids = session.store.query_fielded("search", ["text"], top_k=5).ids()
    assert set(ids) <= {"a", "c"} and ids  # only code docs mention "search"


def test_prefix_search(session):
    # term-prefix "quer" matches "query"/"QueryEngine" tokens in a and c
    ids = session.store.query_prefix("quer", top_k=5).ids()
    assert "a" in ids or "c" in ids


def test_wildcard_search(session):
    ids = session.store.query_wildcard("*fusion*", top_k=5).ids()
    assert ids == ["b"]


def test_fuzzy_search(session):
    # "reciprical" (typo) should still find b via edit distance
    ids = session.store.query_fuzzy("reciprical rank", top_k=5, fuzziness=2).ids()
    assert "b" in ids


def test_more_like_this(session):
    ids = session.store.more_like_this(text="search query method", top_k=5,
                                       min_term_freq=1, min_doc_freq=1).ids()
    assert any(i in ids for i in ("a", "c"))


def test_random_sample_is_seed_reproducible(session):
    a = session.store.random_sample(size=3, seed=7).ids()
    b = session.store.random_sample(size=3, seed=7).ids()
    assert a == b and len(a) == 3


def test_browse_enumerates(session):
    ids = session.store.browse(top_k=10).ids()
    assert set(ids) == {"a", "b", "c"}


def test_facet_and_count_distinct(session):
    assert session.store.facet("kind.keyword") == {"code": 2, "doc": 1}
    assert session.store.count_distinct("kind.keyword") == 2


def test_stats(session):
    st = session.store.stats("year")
    assert st["min"] == 2023 and st["max"] == 2024 and st["count"] == 3


# --------------------------------------------------------------------------- #
# Audit fixes verified against a live cluster (issues.md SDK-C2 / C3 / C4)      #
# --------------------------------------------------------------------------- #
def test_eq_filter_on_string_metadata_matches(session):
    """SDK-C3: `term` on a dynamically-mapped string field is analyzed-vs-keyword and
    matched nothing, so $eq on string metadata failed CLOSED (zero hits, no error)."""
    s = session
    # A multi-word, capitalised value: the standard analyzer lowercases and splits it, so a
    # `term` query on the bare (text) field can never match it. A single lowercase token like
    # "hardware" round-trips unchanged and would pass even against the unfixed adapter.
    s.add([{"id": "sc1", "text": "alpha doc", "metadata": {"category": "Network Hardware"}},
           {"id": "sc2", "text": "beta doc", "metadata": {"category": "System Software"}}])
    s.store.client.indices.refresh(index=INDEX)
    hits = s.search("doc", top_k=10, mode="keyword", filter={"category": "Network Hardware"})
    ids = [h.id for h in hits]
    assert "sc1" in ids, "string $eq filter returned nothing (SDK-C3 regression)"
    assert "sc2" not in ids, "filter did not exclude the non-matching document"


def test_or_filter_is_translated_not_silently_dropped(session):
    """SDK-C2: $and/$or/$not were skipped, so the search ran UNFILTERED and returned more
    results than requested, while filters.validate() accepted the operator."""
    s = session
    s.add([{"id": "or1", "text": "gamma doc", "metadata": {"category": "hardware"}},
           {"id": "or2", "text": "gamma doc", "metadata": {"category": "software"}},
           {"id": "or3", "text": "gamma doc", "metadata": {"category": "firmware"}}])
    s.store.client.indices.refresh(index=INDEX)
    hits = s.search("gamma", top_k=10, mode="keyword",
                    filter={"$or": [{"category": "hardware"}, {"category": "software"}]})
    ids = set(h.id for h in hits)
    assert {"or1", "or2"} <= ids
    assert "or3" not in ids, "$or filter was ignored — the query ran unfiltered (SDK-C2)"


def test_unsupported_filter_operator_raises_instead_of_running_unfiltered(session):
    from search_as_code.errors import InvalidFilterError
    with pytest.raises(InvalidFilterError):
        session.store._to_filter({"$nor": [{"category": "hardware"}]})


def test_sample_is_reproducible_so_the_corpus_fingerprint_is_stable(session):
    """SDK-C4: an unseeded random_score made corpus_fingerprint differ on every call, so
    fingerprint_changed() was always True and explore re-ran every stage every run."""
    from search_as_code.explore.engine import corpus_fingerprint
    s = session
    s.add([{"id": f"fp{i}", "text": f"fingerprint document number {i}"} for i in range(30)])
    s.store.client.indices.refresh(index=INDEX)
    first = [d.id for d in s.store.sample(8)]
    second = [d.id for d in s.store.sample(8)]
    assert first == second, "sample() is not reproducible — resume/drift detection is broken"
    assert corpus_fingerprint(s.store) == corpus_fingerprint(s.store)


def test_ensure_index_refuses_a_conflicting_dimension(session):
    from search_as_code.errors import DimensionMismatchError
    store = session.store
    with pytest.raises(DimensionMismatchError):
        store.ensure_index(store.dim + 7)
