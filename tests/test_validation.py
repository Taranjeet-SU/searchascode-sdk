"""Boundary input validation: bad args fail fast with typed errors instead of
surfacing as deep tracebacks from inside a backend.
"""

import pytest

import search_as_code as sac
from search_as_code import errors

CORPUS = [
    {"id": "1", "text": "hello world", "metadata": {"year": 2020}},
    {"id": "2", "text": "vector search", "metadata": {"year": 2024}},
]


def make() -> sac.Session:
    s = sac.Session("memory")
    s.add(CORPUS)
    return s


@pytest.mark.parametrize("bad", ["", "   ", None, 123, []])
def test_empty_or_nonstring_query_rejected(bad):
    with pytest.raises(errors.InvalidArgumentError):
        make().search(bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "5"])
def test_bad_top_k_rejected(bad):
    with pytest.raises(errors.InvalidArgumentError):
        make().search("hello", top_k=bad)


def test_unknown_mode_rejected():
    with pytest.raises(errors.InvalidModeError):
        make().search("hello", mode="fuzzy")


def test_bad_filter_operator_rejected_at_boundary():
    with pytest.raises(errors.InvalidFilterError):
        make().search("hello", filter={"year": {"$bogus": 2020}})


def test_valid_search_still_works():
    hits = make().search("hello", top_k=2, mode="keyword", filter={"year": {"$gte": 2020}})
    assert len(hits) >= 1


def test_search_many_requires_nonempty_queries():
    with pytest.raises(errors.InvalidArgumentError):
        make().search_many([])


def test_search_many_concurrency_must_be_positive():
    with pytest.raises(errors.InvalidArgumentError):
        make().search_many(["a"], concurrency=0)


def test_document_needs_id():
    with pytest.raises(errors.InvalidArgumentError):
        sac.Session("memory").add([{"id": "", "text": "x"}])


def test_dimension_mismatch_rejected():
    s = sac.Session("memory")
    with pytest.raises(errors.DimensionMismatchError):
        s.add([{"id": "a", "vector": [0.1, 0.2, 0.3]},
               {"id": "b", "vector": [0.1, 0.2]}])
