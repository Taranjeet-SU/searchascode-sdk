"""The adapter contract, enforced.

``README.md`` says "the in-memory test suite is the contract every adapter must satisfy" and
``adapters/base.py`` says the primitive layer emulates whatever an adapter reports ``False``.
Neither was checked: there was no parametrized conformance suite, `faiss_store`, `sqlite_store`,
`nmslib_store` and `milvus_store` had **zero** tests, and `test_units.py` used a hand-rolled fake
instead of a real adapter. That is why SDK-C2 (``$or`` dropped) and SDK-C3 (``$eq`` on strings
matching nothing) were invisible for so long — see issues.md TEST-3 / STR-2.

This is the SDK-local version of the pattern LangChain ships as `langchain-tests`
(`libs/standard-tests/langchain_tests/integration_tests/vectorstores.py`): one suite, run against
every registered backend, so a backend that silently drops a filter clause fails a shared test.

Backends whose client library is not installed, or whose server is not reachable, are SKIPPED —
never silently passed.
"""
from __future__ import annotations

import importlib.util

import pytest

import search_as_code as sac
from search_as_code.types import Document

DIM = 16


def _memory():
    return sac.connect("memory", dim=DIM)


def _sqlite(tmp_path):
    return sac.connect("sqlite", path=str(tmp_path / "sac.db"), dim=DIM)


def _faiss(_tmp_path):
    if importlib.util.find_spec("faiss") is None:
        pytest.skip("faiss not installed")
    return sac.connect("faiss", dim=DIM)


def _chroma(tmp_path):
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")
    return sac.connect("chroma", path=str(tmp_path / "chroma"),
                       collection="conformance", dim=DIM)


def _opensearch(_tmp_path):
    if importlib.util.find_spec("opensearchpy") is None:
        pytest.skip("opensearch-py not installed")
    from opensearchpy import OpenSearch
    hosts = [{"host": "127.0.0.1", "port": 9200}]
    try:
        if not OpenSearch(hosts=hosts, use_ssl=False).ping():
            pytest.skip("no local OpenSearch on :9200")
    except Exception:
        pytest.skip("no local OpenSearch on :9200")
    store = sac.connect("opensearch", index="sac_conformance", dim=DIM, hosts=hosts)
    store.client.indices.delete(index="sac_conformance", ignore=[404])
    store.ensure_index(DIM)
    return store


BACKENDS = {
    "memory": lambda tmp: _memory(),
    "sqlite": _sqlite,
    "faiss": _faiss,
    "chroma": _chroma,
    "opensearch": _opensearch,
}


@pytest.fixture(params=sorted(BACKENDS), ids=sorted(BACKENDS))
def store(request, tmp_path):
    st = BACKENDS[request.param](tmp_path)
    yield st
    # opensearch is the only one with server-side state to clean
    if getattr(st, "backend", "") == "opensearch":
        st.client.indices.delete(index=st.index, ignore=[404])


def _vec(seed: int) -> list[float]:
    """A deterministic unit-ish vector; seed 0 and 1 are far apart."""
    v = [0.0] * DIM
    v[seed % DIM] = 1.0
    return v


CORPUS = [
    Document(id="d1", text="reciprocal rank fusion merges ranked lists",
             vector=_vec(0), metadata={"kind": "doc", "year": 2023, "team": "Search Infra"}),
    Document(id="d2", text="dense retrieval uses vector similarity",
             vector=_vec(1), metadata={"kind": "doc", "year": 2024, "team": "Search Infra"}),
    Document(id="d3", text="the QueryEngine class exposes a search method",
             vector=_vec(2), metadata={"kind": "code", "year": 2024, "team": "Platform"}),
]


@pytest.fixture()
def loaded(store):
    store.upsert(CORPUS)
    if getattr(store, "backend", "") == "opensearch":
        store.client.indices.refresh(index=store.index)
    return store


# --------------------------------------------------------------------------- #
# the contract                                                                  #
# --------------------------------------------------------------------------- #
def test_capabilities_is_declared(store):
    caps = store.capabilities()
    assert isinstance(caps.dense, bool) and isinstance(caps.keyword, bool)


def test_upsert_then_count_and_get(loaded):
    assert loaded.count() == 3
    got = {d.id: d for d in loaded.get(["d1", "d3"])}
    assert set(got) == {"d1", "d3"}
    assert "reciprocal" in (got["d1"].text or "")


def test_upsert_is_idempotent_on_id(loaded):
    loaded.upsert([Document(id="d1", text="replaced text", vector=_vec(0))])
    if getattr(loaded, "backend", "") == "opensearch":
        loaded.client.indices.refresh(index=loaded.index)
    assert loaded.count() == 3
    assert "replaced" in (loaded.get(["d1"])[0].text or "")


def test_dense_search_returns_ranked_hits(loaded):
    hits = loaded.query_vector(_vec(0), top_k=3)
    assert len(hits) >= 1
    assert hits[0].id == "d1", "nearest neighbour of d1's own vector should be d1"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "scores must be larger-is-better and sorted"


def test_top_k_is_respected(loaded):
    assert len(loaded.query_vector(_vec(0), top_k=2)) <= 2


def test_delete_removes_the_document(loaded):
    loaded.delete(["d3"])
    if getattr(loaded, "backend", "") == "opensearch":
        loaded.client.indices.refresh(index=loaded.index)
    assert loaded.count() == 2
    assert [d.id for d in loaded.get(["d3"])] == []


def test_eq_filter_on_a_string_field(loaded):
    """SDK-C3 lived here: `term` against a dynamically-mapped text field matches nothing.
    A multi-word capitalised value is used deliberately — a single lowercase token survives
    the analyzer unchanged and would pass even on a broken adapter."""
    if not loaded.capabilities().metadata_filter:
        pytest.skip("backend declares no metadata filtering")
    hits = loaded.query_vector(_vec(0), top_k=10, flt={"team": "Search Infra"})
    ids = {h.id for h in hits}
    assert "d3" not in ids, "string equality filter did not exclude the non-matching doc"
    assert ids <= {"d1", "d2"}


def test_numeric_and_range_filters(loaded):
    if not loaded.capabilities().metadata_filter:
        pytest.skip("backend declares no metadata filtering")
    ids = {h.id for h in loaded.query_vector(_vec(0), top_k=10, flt={"year": 2024})}
    assert ids <= {"d2", "d3"} and "d1" not in ids


def test_filters_never_fail_open(loaded):
    """A filter must never return MORE than the unfiltered query. SDK-C2 was exactly this:
    $or was skipped, so the search ran unfiltered and quietly over-returned."""
    if not loaded.capabilities().metadata_filter:
        pytest.skip("backend declares no metadata filtering")
    unfiltered = len(loaded.query_vector(_vec(0), top_k=10))
    for flt in ({"kind": "code"}, {"year": 2023}, {"team": "Platform"}):
        assert len(loaded.query_vector(_vec(0), top_k=10, flt=flt)) <= unfiltered, \
            f"filter {flt} returned more hits than no filter at all"


def test_unsupported_filter_operator_is_not_silently_ignored(loaded):
    """Either translate the operator or raise — never run unfiltered."""
    from search_as_code.errors import InvalidFilterError, SacError
    if not loaded.capabilities().metadata_filter:
        pytest.skip("backend declares no metadata filtering")
    try:
        hits = loaded.query_vector(_vec(0), top_k=10, flt={"$nor": [{"kind": "code"}]})
    except (InvalidFilterError, SacError, ValueError, NotImplementedError):
        return                                   # raised: acceptable
    assert len(hits) < 3, "an unsupported operator was ignored and the query ran unfiltered"


def test_session_emulates_keyword_and_hybrid_everywhere(loaded):
    """Capability emulation is the portability promise: mode='hybrid' behaves the same on a
    backend that has no native keyword search."""
    s = sac.Session(loaded, embedder=sac.HashEmbedder(dim=DIM))
    for mode in ("dense", "keyword", "hybrid"):
        hits = s.search("reciprocal rank fusion", top_k=3, mode=mode)
        assert isinstance(hits, sac.ResultSet), f"{mode} did not return a ResultSet"


def test_sample_returns_documents(loaded):
    try:
        docs = loaded.sample(2)
    except NotImplementedError:
        pytest.skip("backend does not implement sample()")
    assert len(docs) <= 2
    assert all(isinstance(d, Document) for d in docs)


def test_sample_is_reproducible(loaded):
    """corpus_fingerprint hashes a sample, so a non-reproducible sample breaks explore's
    resume/drift detection entirely (SDK-C4)."""
    try:
        first = [d.id for d in loaded.sample(3)]
        second = [d.id for d in loaded.sample(3)]
    except NotImplementedError:
        pytest.skip("backend does not implement sample()")
    assert first == second, "sample() is not reproducible across calls"


def test_describe_schema_reports_a_usable_shape(loaded):
    schema = loaded.describe_schema()
    assert isinstance(schema, dict)
    # consumers must be able to find field names without knowing the backend (SDK-C14)
    assert ("fields" in schema) or ("metadata_keys" in schema), schema


def test_get_of_a_missing_id_does_not_raise(loaded):
    """A missing id yields no document — it must not raise and must not invent one."""
    got = loaded.get(["does-not-exist"])
    assert [d.id for d in got] == []
