"""Unit tests for the primitives layer (TEST-5 / issues.md §17).

The audit found 21 of 26 primitives had zero tests, which is exactly where four
correctness bugs hid (SDK-C15/16/17/18). These tests lock the documented
invariants, not the happy path:

- mmr's diversified ORDER survives `.top()` (SDK-C15)
- `.info` side-signals survive every ResultSet-constructing primitive (SDK-C16)
- normalize_scores maps singletons/ties to 1.0, not 0.0 (SDK-C17)
- fusion is weight- and permutation-sane; consensus votes are correct
"""
from __future__ import annotations

import math

import pytest

import search_as_code.primitives as P
from search_as_code.types import Document, Hit, ResultSet


def H(id, score, vector=None, text=None, meta=None):
    return Hit(id=id, score=score,
               document=Document(id=id, text=text or f"text {id}", vector=vector,
                                 metadata=meta or {}))


def RS(*hits, info=None):
    return ResultSet(hits, info=info)


# ---------------------------------------------------------------- fuse / rrf
def test_fuse_rrf_prefers_id_in_both_lists():
    a = RS(H("x", 0.9), H("y", 0.8))
    b = RS(H("y", 0.7), H("z", 0.6))
    out = P.fuse([a, b])
    assert out.ids()[0] == "y"          # appears in both lists
    assert set(out.ids()) == {"x", "y", "z"}


def test_fuse_weights_bias_the_ranking():
    a = RS(H("x", 0.9))
    b = RS(H("z", 0.9))
    out = P.fuse([a, b], weights=[1.0, 3.0])
    assert out.ids()[0] == "z"


def test_rrf_is_fuse_alias():
    assert P.rrf is P.fuse


def test_fuse_carries_info_and_sums_degraded():
    a = ResultSet([H("x", 0.9)], info={"degraded": {"hyde_failed": 1}, "agreement": 0.5})
    b = ResultSet([H("y", 0.8)], info={"degraded": {"hyde_failed": 2, "regex_empty": 1}})
    out = P.fuse([a, b])
    assert out.info["degraded"] == {"hyde_failed": 3, "regex_empty": 1}
    assert out.agreement == 0.5


# ------------------------------------------------------- normalize / fusion
def test_normalize_scores_minmax_range():
    out = P.normalize_scores(RS(H("a", 3.0), H("b", 1.0), H("c", 2.0)))
    by = {h.id: h.score for h in out}
    assert by["a"] == 1.0 and by["b"] == 0.0 and 0.0 < by["c"] < 1.0


def test_normalize_scores_singleton_is_one_not_zero():
    # SDK-C17: a single hit used to map to 0.0, collapsing relative_score_fusion.
    out = P.normalize_scores(RS(H("only", 7.3)))
    assert out[0].score == 1.0


def test_normalize_scores_all_tied_is_one():
    out = P.normalize_scores(RS(H("a", 2.0), H("b", 2.0)))
    assert [h.score for h in out] == [1.0, 1.0]


def test_relative_score_fusion_of_singletons_does_not_tie_everything():
    # Regression for SDK-C17's downstream symptom: two singleton lists fused.
    out = P.relative_score_fusion([RS(H("a", 9.0)), RS(H("b", 0.1))])
    assert {h.score for h in out} != {0.0}
    assert len(out) == 2


def test_relative_score_fusion_double_membership_wins():
    # y is mid-pack in a and top in b; x is top only in a. (Lists need >2 entries:
    # minmax pins each list's minimum to 0, so a 2-list's runner-up contributes nothing.)
    a = RS(H("x", 1.0), H("y", 0.6), H("w", 0.2))
    b = RS(H("y", 1.0), H("z", 0.2))
    assert P.relative_score_fusion([a, b]).ids()[0] == "y"


# ----------------------------------------------------------------- mmr (C15)
def test_mmr_order_survives_top():
    # SDK-C15: .top() re-sorts by score; with ORIGINAL scores kept it silently
    # undid the diversification — and surface.py teaches exactly this chain.
    # "aligned" is most relevant to q but carries the LOWEST original score.
    q = [1.0, 0.0]
    hits = RS(H("aligned", 0.1, vector=[1.0, 0.0]),
              H("offaxis", 0.9, vector=[0.0, 1.0]))
    out = P.mmr(q, hits, lambda_=1.0, top_k=2)      # pure relevance: aligned first
    assert out.ids() == ["aligned", "offaxis"]
    assert out.top(2).ids() == ["aligned", "offaxis"]   # the old code returned offaxis first here


def test_mmr_scores_strictly_decreasing():
    q = [1.0, 0.0]
    hits = RS(H("a", 0.1, vector=[1.0, 0.0]), H("b", 0.9, vector=[0.0, 1.0]),
              H("c", 0.5, vector=[0.7, 0.714]))
    out = P.mmr(q, hits, lambda_=0.5, top_k=3)
    scores = [h.score for h in out]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_mmr_novec_hits_appended_below_selected():
    q = [1.0, 0.0]
    hits = RS(H("v", 0.1, vector=[1.0, 0.0]), H("novec", 99.0))
    out = P.mmr(q, hits, top_k=2)
    assert out.ids() == ["v", "novec"]
    assert out.top(2).ids() == ["v", "novec"]


def test_mmr_diversity_penalty_skips_near_duplicate():
    # With diversity-heavy lambda (<0.5), the near-duplicate of pick 1 must lose
    # to a distinct-but-relevant candidate.
    q = [1.0, 0.0]
    hits = RS(H("c1", 0.5, vector=[1.0, 0.0]),
              H("dup_of_c1", 0.5, vector=[0.999, 0.045]),
              H("distinct", 0.5, vector=[0.6, 0.8]))
    out = P.mmr(q, hits, lambda_=0.3, top_k=2)
    assert out.ids() == ["c1", "distinct"]


# --------------------------------------------------------- info survives (C16)
def test_info_survives_every_chaining_primitive():
    cons = P.consensus([RS(H("a", 1.0), H("b", 0.5)), RS(H("a", 0.9))], top_k=5)
    assert cons.agreement > 0
    assert P.rerank("q", cons).agreement == cons.agreement          # rerank kept it
    assert P.normalize_scores(cons).agreement == cons.agreement
    assert P.score_cutoff(cons, min_k=1).agreement == cons.agreement
    assert P.diversity_quota(cons, key=lambda h: h.id).agreement == cons.agreement
    assert P.fuse([cons]).agreement == cons.agreement
    assert cons.top(1).agreement == cons.agreement                  # was already fixed (C13)


def test_mmr_and_freshness_carry_info():
    rs = ResultSet([H("a", 1.0, vector=[1.0, 0.0]), H("b", 0.9, vector=[0.0, 1.0])],
                   info={"agreement": 0.7})
    assert P.mmr([1.0, 0.0], rs, top_k=2).agreement == 0.7
    fresh = P.freshness(rs, timestamp=lambda h: 0.0, now=10.0, half_life=5.0)
    assert fresh.agreement == 0.7


# ----------------------------------------------------------------- consensus
def test_consensus_votes_and_agreement():
    lists = [RS(H("a", 1.0), H("b", 0.5)), RS(H("a", 0.9), H("c", 0.4)), RS(H("a", 0.8))]
    out = P.consensus(lists, top_k=3)
    assert out.ids()[0] == "a"
    assert out.votes["a"] == 3
    assert out.n_lists == 3
    assert out.agreement == 1.0


def test_consensus_empty_lists_are_ignored():
    out = P.consensus([RS(), RS(H("a", 1.0))])
    assert out.ids() == ["a"] and out.n_lists == 1


# ------------------------------------------------------------- gates & shapes
def test_score_cutoff_band_keeps_flat_curves_wide():
    flat = RS(*[H(str(i), 1.0 - i * 0.001) for i in range(30)])
    peaked = RS(H("top", 1.0), *[H(str(i), 0.2) for i in range(29)])
    assert len(P.score_cutoff(flat, min_k=5)) > len(P.score_cutoff(peaked, min_k=5))


def test_score_cutoff_respects_min_and_max_k():
    rs = RS(*[H(str(i), 1.0 - i * 0.2) for i in range(10)])
    assert len(P.score_cutoff(rs, min_k=4, max_k=6)) >= 4
    assert len(P.score_cutoff(rs, rel_band=5.0, min_k=1, max_k=6)) <= 6


def test_confidence_and_abstain():
    strong = RS(H("a", 0.95), H("b", 0.2))
    weak = RS(H("a", 0.15), H("b", 0.14))
    c = P.confidence(strong)
    assert c["top"] == pytest.approx(0.95) and c["gap"] == pytest.approx(0.75)
    assert not P.abstain(strong, min_top=0.5, min_gap=0.1)
    assert P.abstain(weak, min_top=0.5)
    assert P.abstain(RS())                      # empty always abstains


def test_score_cliff_detects_dropoff():
    cliffy = RS(*[H(str(i), s) for i, s in enumerate([1.0, 0.98, 0.96, 0.94, 0.3, 0.28])])
    flat = RS(*[H(str(i), 1.0 - i * 0.01) for i in range(6)])
    assert P.score_cliff(cliffy)["has_cliff"]
    assert not P.score_cliff(flat)["has_cliff"]


def test_diversity_quota_caps_per_group():
    rs = RS(H("a1", 1.0, meta={"src": "a"}), H("a2", 0.9, meta={"src": "a"}),
            H("b1", 0.8, meta={"src": "b"}))
    out = P.diversity_quota(rs, key=lambda h: h.get("src"), max_per_group=1)
    assert out.ids() == ["a1", "b1"]


def test_result_diversity_flags_near_duplicates():
    dup = RS(H("a", 1.0, vector=[1.0, 0.0]), H("b", 0.9, vector=[0.999, 0.01]))
    div = RS(H("a", 1.0, vector=[1.0, 0.0]), H("b", 0.9, vector=[0.0, 1.0]))
    assert P.result_diversity(dup)["redundant"]
    assert not P.result_diversity(div)["redundant"]


def test_max_similarity_probe():
    rs = RS(H("a", 1.0, vector=[1.0, 0.0]))
    assert P.max_similarity([1.0, 0.0], rs) == pytest.approx(1.0)
    assert P.max_similarity([0.0, 1.0], rs) == pytest.approx(0.0, abs=1e-6)
    assert P.max_similarity([1.0, 0.0], RS()) == 0.0


# ---------------------------------------------------------------- rerank etc.
def test_rerank_with_injected_scorer_and_lexical_fallback():
    rs = RS(H("a", 0.1, text="the cat sat"), H("b", 0.9, text="stock market news"))
    out = P.rerank("cat", rs, reranker=lambda q, ts: [10.0 if "cat" in t else 0.0 for t in ts])
    assert out.ids()[0] == "a"
    out2 = P.rerank("cat sat", rs)              # lexical-overlap emulation
    assert out2.ids()[0] == "a"
    assert P.rerank("q", RS()) == []


def test_dedup_keeps_best_per_key():
    rs = RS(H("a", 0.5), H("a", 0.9), H("b", 0.3))
    out = P.dedup(rs)
    assert sorted(out.ids()) == ["a", "b"]
    assert max(h.score for h in out if h.id == "a") == 0.9


def test_freshness_blends_decay():
    old, new = H("old", 0.9), H("new", 0.85)
    ts = {"old": 0.0, "new": 100.0}
    out = P.freshness(RS(old, new), timestamp=lambda h: ts[h.id], now=100.0,
                      half_life=10.0, weight=0.5)
    assert out.ids()[0] == "new"


def test_fan_out_parallel_matches_serial():
    assert P.fan_out(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert P.fan_out(lambda x: x * 2, [1, 2, 3], concurrency=1) == [2, 4, 6]
    assert P.fan_out(lambda x: x * 2, []) == []


# ------------------------------------------------- generator-driven primitives
def _gen_lines(*lines):
    return lambda prompt: list(lines)


def test_expand_always_includes_original_query():
    out = P.expand("orig query", _gen_lines("alt one", "alt two"), n=2)
    assert "orig query" in out and "alt one" in out


def test_decompose_returns_subquestions():
    subs = P.decompose("who directed A and who wrote B?",
                       _gen_lines("who directed A?", "who wrote B?"))
    assert len(subs) == 2


def test_quality_filter_drops_short_docs():
    rs = RS(H("long", 1.0, text="x" * 100), H("short", 0.9, text="hi"))
    assert P.quality_filter(rs, min_chars=40).ids() == ["long"]


def test_normalize_query_and_rare_terms():
    assert isinstance(P.normalize_query("What is a CD rate?"), str)
    rare = P.rare_terms("the reactor XJ-900 manual")
    assert isinstance(rare, list)


def test_content_type_tags():
    assert isinstance(P.content_type("def foo():\n    return 1"), str)
