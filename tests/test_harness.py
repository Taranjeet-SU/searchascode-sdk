"""End-to-end tests over the zero-dependency in-memory backend.

These double as the executable spec any real adapter must satisfy.
"""

import search_as_code as sac
from search_as_code import LocalExecutor, Session
from search_as_code.filters import matches, normalize


CORPUS = [
    {"id": "d1", "text": "the quick brown fox jumps over the lazy dog", "metadata": {"lang": "en", "year": 2020}},
    {"id": "d2", "text": "a fast auburn fox leaps above a sleepy hound", "metadata": {"lang": "en", "year": 2023}},
    {"id": "d3", "text": "le renard brun rapide saute par-dessus le chien", "metadata": {"lang": "fr", "year": 2021}},
    {"id": "d4", "text": "vector databases power semantic retrieval for agents", "metadata": {"lang": "en", "year": 2024}},
    {"id": "d5", "text": "reciprocal rank fusion merges search result lists", "metadata": {"lang": "en", "year": 2024}},
]


def make_session() -> Session:
    s = Session("memory")
    s.add(CORPUS)
    return s


def test_connect_and_capabilities():
    assert "memory" in sac.available()
    store = sac.connect("memory")
    assert store.capabilities().dense


def test_dense_search_returns_ranked_hits():
    s = make_session()
    hits = s.search("fox jumping over dog", top_k=3)
    assert len(hits) == 3
    # scores are larger-is-better and sorted descending
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert "d1" in hits.ids() or "d2" in hits.ids()


def test_metadata_filter():
    s = make_session()
    hits = s.search("fox", top_k=10, filter={"lang": "en"})
    assert all(h.get("lang") == "en" for h in hits)
    assert "d3" not in hits.ids()


def test_filter_operators():
    md = {"year": 2023, "lang": "en"}
    assert matches(md, {"year": {"$gte": 2020}})
    assert not matches(md, {"year": {"$gt": 2023}})
    assert matches(md, {"lang": {"$in": ["en", "fr"]}})
    assert matches(md, {"$and": [{"lang": "en"}, {"year": {"$lte": 2023}}]})
    assert normalize({"lang": "en"}) == {"lang": {"$eq": "en"}}


def test_keyword_and_hybrid_modes():
    s = make_session()
    kw = s.search("reciprocal rank fusion", top_k=2, mode="keyword")
    assert kw.ids()[0] == "d5"
    hy = s.search("fusion of search results", top_k=3, mode="hybrid")
    assert "d5" in hy.ids()


def test_search_many_fans_out_and_fuses():
    s = make_session()
    fused = s.search_many(["fox", "hound", "dog"], top_k=5)
    # fused hits are deduped by id
    assert len(fused.ids()) == len(set(fused.ids()))
    assert len(fused) > 0


def test_fuse_primitive_is_rank_based():
    s = make_session()
    a = s.search("fox", top_k=3)
    b = s.search("vector retrieval", top_k=3)
    fused = s.fuse([a, b])
    assert len(fused.ids()) == len(set(fused.ids()))


def test_evidence_is_compact():
    s = make_session()
    ev = s.search("fox", top_k=2).to_evidence(fields=["lang"], max_chars=20)
    assert all(set(row) <= {"id", "score", "text", "lang"} for row in ev)
    assert all(len(row.get("text", "")) <= 20 for row in ev)


def test_state_store_stays_out_of_band():
    s = make_session()
    s.remember("seeds", s.search("fox", top_k=3))
    assert "seeds" in s.state_keys()
    assert len(s.recall("seeds")) == 3
    s.forget("seeds")
    assert "seeds" not in s.state_keys()


def test_sandbox_returns_only_evidence():
    s = make_session()
    box = LocalExecutor(s)
    code = """
seeds = sac.search_many(["fox", "dog"], top_k=5)
sac.remember("seeds", seeds)
top = seeds.top(2)
print("found", len(seeds), "candidates")
evidence = top.to_evidence(fields=["lang", "year"])
"""
    res = box.run(code)
    assert res.ok, res.error
    assert res.evidence and len(res.evidence) == 2
    assert "candidates" in res.stdout
    assert "seeds" in res.state_keys
    payload = res.for_model()
    assert "evidence" in payload and "stdout" in payload


def test_sandbox_persists_state_across_runs():
    s = make_session()
    box = LocalExecutor(s)
    box.run("sac.remember('x', sac.search('fox', top_k=2))")
    res = box.run("evidence = sac.recall('x').to_evidence()")
    assert res.ok, res.error
    assert len(res.evidence) == 2


def test_sandbox_reports_errors_without_crashing():
    s = make_session()
    box = LocalExecutor(s)
    res = box.run("evidence = sac.search('fox', top_k=2)[999].id")
    assert not res.ok
    assert "Error" in (res.error or "") or "IndexError" in (res.error or "")


def test_regex_search_exact_match():
    s = make_session()
    # only d5 mentions "fusion"; regex is exact, unlike semantic search
    hits = s.search(r"fusion", top_k=5, mode="regex")
    assert hits.ids() == ["d5"]
    # anchored / case-sensitive pattern
    assert s.search(r"\bfox\b", top_k=5, mode="regex").ids() and \
        set(s.search(r"\bfox\b", top_k=5, mode="regex").ids()) <= {"d1", "d2"}


def test_regex_respects_metadata_filter():
    s = make_session()
    hits = s.search(r"fox", top_k=5, mode="regex", filter={"lang": "en"})
    assert "d3" not in hits.ids()


def test_mmr_diversifies():
    s = make_session()
    pool = s.search("fox", top_k=5)
    div = s.mmr("fox", pool, lambda_=0.5, top_k=3)
    assert len(div) == 3
    assert len(set(div.ids())) == 3  # no dupes


def test_mmr_primitive_prefers_relevance_at_lambda_one():
    import search_as_code as sac
    s = make_session()
    pool = s.search("fox jumps over dog", top_k=5)
    qv = s._embed_query("fox jumps over dog")
    relevance_only = sac.mmr(qv, pool, lambda_=1.0, top_k=1)
    assert relevance_only.ids() == pool.top(1).ids()


def test_route_across_stores_fuses_and_tags():
    import search_as_code as sac
    s1 = Session("memory"); s1.add(CORPUS[:3])
    s2 = Session("memory"); s2.add(CORPUS[3:])
    merged = sac.route([s1, s2], "vector fusion retrieval", top_k=5)
    assert all(h.store == "memory" for h in merged)
    assert len(merged.ids()) == len(set(merged.ids()))


def test_pluggable_query_primitives_use_generator():
    # generator stub: returns canned variants / sub-questions
    def gen(prompt):
        return ["fox", "hound", "dog"]

    s = Session("memory", generator=gen)
    s.add(CORPUS)
    expanded = s.expand_search("animal", top_k=5, n=3)
    assert len(expanded) > 0
    decomposed = s.decompose_search("animals", top_k=5)
    assert len(decomposed) > 0
    hyde = s.hyde_search("animal jumping", top_k=3)
    assert len(hyde) > 0


def test_query_primitive_without_generator_raises():
    import pytest
    s = make_session()
    with pytest.raises(RuntimeError):
        s.expand_search("anything")


def test_compress_shrinks_text():
    s = Session("memory")
    long = "Fusion merges lists. The weather is nice today. RRF combines rankings by rank."
    s.add([{"id": "L", "text": long}])
    hits = s.search("reciprocal rank fusion", top_k=1)
    compressed = s.compress("reciprocal rank fusion", hits, keep=1)
    assert len(compressed[0].text) < len(long)


def test_hash_embedder_is_deterministic():
    e = sac.HashEmbedder(dim=64)
    v1 = e.embed(["hello world"])[0]
    v2 = e.embed(["hello world"])[0]
    assert v1 == v2
    assert len(v1) == 64
