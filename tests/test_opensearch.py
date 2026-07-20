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
