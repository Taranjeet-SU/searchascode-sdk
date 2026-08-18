"""Regression tests for the generator-contract bug family (issues.md GEN-1 / GEN-2 / GEN-3).

Every test here emulates ``phase1.llm.LLM.as_generator()`` — the only generator adapter used in
this repo — which returns a completion **split into lines**. Before the fix, six consumers took
``out[0]`` and saw only the first line. These tests assert the documented behaviour (all the
sub-questions, the whole passage, the whole profile) rather than the happy path, so the same
class of bug cannot come back silently.
"""
from __future__ import annotations

import search_as_code as sac
from search_as_code._genutil import gen_lines, gen_text

THREE_SUBS = ("What year was Film A released?\n"
              "Who directed Film B?\n"
              "Which studio produced Film C?")


def line_splitting_generator(text):
    """Exactly what LLM.as_generator() does: split the completion into stripped lines."""
    def gen(_prompt):
        return [ln.strip("-*0123456789. \t") for ln in text.splitlines() if ln.strip()]
    return gen


def single_string_generator(text):
    """The other legal adapter shape: one element holding the whole completion."""
    def gen(_prompt):
        return [text]
    return gen


# --------------------------------------------------------------------------- #
# the helper itself                                                             #
# --------------------------------------------------------------------------- #
def test_gen_text_joins_a_line_list_instead_of_indexing_it():
    out = line_splitting_generator(THREE_SUBS)("q")
    assert gen_text(out) == THREE_SUBS
    assert gen_text([THREE_SUBS]) == THREE_SUBS      # single-string adapter round-trips too
    assert gen_text("plain string") == "plain string"
    assert gen_text(None, default="fallback") == "fallback"
    assert gen_text([], default="fallback") == "fallback"


def test_gen_lines_is_identical_for_both_adapter_shapes():
    a = gen_lines(line_splitting_generator(THREE_SUBS)("q"))
    b = gen_lines(single_string_generator(THREE_SUBS)("q"))
    assert a == b == ["What year was Film A released?",
                      "Who directed Film B?",
                      "Which studio produced Film C?"]


def test_gen_lines_strips_markers_and_applies_limits():
    out = ["1. first item", "- second item", "  ", "x", "* third item"]
    assert gen_lines(out, min_len=3) == ["first item", "second item", "third item"]
    assert gen_lines(out, min_len=3, max_items=2) == ["first item", "second item"]


# --------------------------------------------------------------------------- #
# GEN-1 — the six consumers                                                     #
# --------------------------------------------------------------------------- #
def test_decompose_query_uses_the_llm_not_the_lexical_fallback():
    """harness/loop.py: len(subs) >= 2 used to fail, silently discarding the LLM output."""
    from search_as_code.harness.loop import decompose_query
    subs = decompose_query("q", line_splitting_generator(THREE_SUBS))
    assert len(subs) == 3
    assert subs[0] == "What year was Film A released?"


def test_decompose_query_still_falls_back_when_the_generator_is_useless():
    from search_as_code.harness.loop import decompose_query
    subs = decompose_query("Which film and which studio?", line_splitting_generator("one line only"))
    assert len(subs) >= 1          # lexical split still engages, no crash


def test_primitives_decompose_and_loop_decompose_agree_on_count():
    """The two decompose entry points saw different numbers of sub-facts (GEN-1 / SDK-R3)."""
    from search_as_code.harness.loop import decompose_query
    gen = line_splitting_generator(THREE_SUBS)
    assert len(sac.decompose("q", gen)) == len(decompose_query("q", gen)) == 3


def test_llm_profile_returns_every_line_not_just_the_first():
    """session._llm_profile: item (3) — the recommended primitives — was being dropped."""
    profile_text = ("(1) prose documentation pages\n"
                    "(2) key entities: product names, version numbers\n"
                    "(3) best primitives: fielded match for versions, dense for prose")
    s = sac.Session("memory", generator=line_splitting_generator(profile_text))
    s.add([{"id": "1", "text": "hello world"}])
    got = s.describe(llm=True)["llm"]
    assert "(3)" in got and "best primitives" in got
    assert got.count("\n") == 2


def test_generate_multihop_parses_pretty_printed_json():
    """explore/multihop.py (GEN-2): the regex ran on line 1, so indented JSON never matched."""
    from search_as_code.explore.multihop import _gen

    class _Doc:
        def __init__(self, i):
            self.id, self.text, self.metadata = i, f"text of {i}", {"title": f"T{i}"}

    pretty = '{\n  "question": "Which studio produced both films?",\n  "facts": ["a", "b"]\n}'
    got = _gen(line_splitting_generator(pretty), [_Doc("d1"), _Doc("d2")])
    assert got is not None, "pretty-printed JSON was dropped — the chain would be skipped"
    assert got["query"] == "Which studio produced both films?"
    assert got["gold_ids"] == ["d1", "d2"]


def test_rephrase_returns_n_paraphrases_not_one():
    """explore/fit.py: rephrases=2 was a no-op beyond the first paraphrase."""
    from search_as_code.explore.fit import _rephrase
    three = "how do agents retrieve\nwhat is agentic retrieval\nagent retrieval methods"
    s = sac.Session("memory", generator=line_splitting_generator(three))
    assert len(_rephrase(s, "agentic retrieval", n=3)) == 3


# --------------------------------------------------------------------------- #
# GEN-3 — HyDE embedded only the first line of the hypothetical document        #
# --------------------------------------------------------------------------- #
def test_hyde_search_embeds_the_whole_passage():
    passage = ("Here is a passage:\n"
               "Agentic retrieval systems compose several searches in one program, "
               "keeping intermediate state out of the model context.")
    seen = {}

    class RecordingEmbedder:
        dim = 8

        def embed(self, texts):
            seen.setdefault("texts", []).extend(texts)
            return [[0.1] * 8 for _ in texts]

    s = sac.Session("memory", embedder=RecordingEmbedder(),
                    generator=line_splitting_generator(passage))
    s.add([{"id": "1", "text": "hello"}])
    seen["texts"] = []
    s.hyde_search("how do agents retrieve?")
    embedded = seen["texts"][-1]
    assert "keeping intermediate state" in embedded, "HyDE embedded only the preamble line"


def test_hyde_falls_back_to_the_query_when_the_generator_returns_nothing():
    s = sac.Session("memory", generator=lambda _p: [])
    s.add([{"id": "1", "text": "hello"}])
    s.hyde_search("fallback query")          # must not raise / must not embed ""
