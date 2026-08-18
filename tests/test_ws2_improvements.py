"""WS2/WS5 improvements (fable.md §2b): sufficiency-premise judge, convergence stop,
sub-fact cap, asymmetric query/passage embedding."""
from __future__ import annotations

import numpy as np

import search_as_code as sac
from search_as_code.harness.diagnostic_judge import (
    DIAGNOSTIC_PROMPT,
    SUFFICIENCY_PROMPT,
    DiagnosticJudge,
)


# ---------------------------------------------------------- judge premise
def test_sufficiency_premise_selects_the_new_prompt():
    j = DiagnosticJudge(generator=None, premise="sufficiency")
    assert j.prompt is SUFFICIENCY_PROMPT
    assert "PLAUSIBLY DERIVE" in j.prompt
    assert "checklist" in j.prompt                     # sub-facts are hints, not a checklist
    assert DiagnosticJudge(generator=None).prompt is DIAGNOSTIC_PROMPT   # default unchanged


def test_sufficiency_prompt_keeps_the_output_contract():
    # parse_verdict must work unchanged on both premises' output format
    for field in ("COVERED:", "MISSING:", "TECHNIQUE:", "NEXT_QUERY:", "CONFIDENCE:", "VERDICT:"):
        assert field in SUFFICIENCY_PROMPT


# ------------------------------------------------- asymmetric embedding (P2-5)
class _VecEmbedder:
    """Returns a fixed vector per call — lets the test see WHICH embedder ran."""
    def __init__(self, vec):
        self.vec = list(vec)
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return [self.vec for _ in texts]


def test_query_embedder_is_used_for_queries_and_embedder_for_passages():
    passage_emb = _VecEmbedder([1.0, 0.0])            # docs land at e1
    query_emb = _VecEmbedder([0.0, 1.0])              # queries point at e2
    s = sac.Session("memory", embedder=passage_emb, query_embedder=query_emb)
    s.add([{"id": "a", "text": "doc"}])
    s.search("q", top_k=1, mode="dense")
    assert passage_emb.calls >= 1                     # indexing used the passage embedder
    assert query_emb.calls >= 1                       # the query used the query embedder
    # and the stored vector is the passage one, not the query one
    assert s.store.get(["a"])[0].vector == [1.0, 0.0]


def test_query_embedder_defaults_to_symmetric():
    s = sac.Session("memory")
    assert s.query_embedder is s.embedder


# -------------------------------------------- convergence stop + sub-fact cap
class _StubGen:
    """Author the SAME dense one-liner every hop -> pools converge immediately."""
    def complete(self, prompt, system=None):
        return ("```python\ndef run(session, query, top_k):\n"
                "    return session.search(query, top_k=top_k, mode='dense').ids()\n```")


class _FailJudge:
    premise = "coverage"
    def judge(self, query, subfacts, candidates, coverage):
        return {"verdict": "FAIL", "covered": "", "missing": "1", "diagnosis": "absent",
                "technique": "hyde", "next_query": subfacts[0]}


def _session():
    s = sac.Session("memory")
    s.add([{"id": str(i), "text": f"doc {i} about topic"} for i in range(5)])
    s.generator = lambda p: ["sub one", "sub two", "sub three"]   # decompose source
    return s


def test_convergence_stop_beats_a_judge_that_never_passes():
    from search_as_code.harness.agentic import agentic_solve
    s = _session()
    emb = s.embedder.embed
    res = agentic_solve(s, "topic", generator=_StubGen(), judge=_FailJudge(),
                        judge_stop=True, embedder=emb,
                        reranker=lambda q, ts: [0.0] * len(ts), max_hops=10)
    assert res["stopped_by"] == "converged"           # not maxhops: stopped at zero LLM cost
    assert res["hops"] < 10


def test_max_subfacts_caps_the_decomposition():
    from search_as_code.harness.agentic import agentic_solve
    s = _session()
    captured = []
    class SpyJudge(_FailJudge):
        def judge(self, query, subfacts, candidates, coverage):
            captured.append(list(subfacts))
            return super().judge(query, subfacts, candidates, coverage)
    emb = s.embedder.embed
    agentic_solve(s, "topic", generator=_StubGen(), judge=SpyJudge(), judge_stop=True,
                  embedder=emb, reranker=lambda q, ts: [0.0] * len(ts),
                  max_hops=3, max_subfacts=2)
    assert captured and all(len(sf) <= 2 for sf in captured)
