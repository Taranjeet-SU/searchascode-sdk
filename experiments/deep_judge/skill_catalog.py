"""A retrievable SKILL CATALOG the diagnostic judge looks up to pick the next-hop technique.

Seeded from NirDiamant/RAG_Techniques (https://github.com/NirDiamant/RAG_Techniques): each entry pairs a
`when_to_use` description (what failure it fixes) with an executable `technique` id. The judge, given its
DIAGNOSIS of the missing sub-fact, does a semantic lookup over `when_to_use` and gets back the best
technique + a focused query — turning a fixed 5-way menu into an extensible, literature-grounded playbook.

`runtime=True` techniques act at query time on our fixed OpenSearch index; `runtime=False` (semantic
chunking, hierarchical/RAPTOR, HyPE, document augmentation) are INDEX-time and listed for completeness
but not selectable as a next hop here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TechCard:
    name: str
    technique: str        # executable id understood by run_playbook.apply_technique / os_query authoring
    when_to_use: str
    family: str           # query | rank | context | authored
    runtime: bool = True
    source: str = "RAG_Techniques"


CATALOG = [
    TechCard("HyDE", "hyde",
             "The target entity is only DESCRIBED generically, not named (low lexical overlap but plausibly "
             "present) — hallucinate a hypothetical answer document and embed THAT to bridge the vocabulary gap.",
             "query"),
    TechCard("Query decomposition", "decompose",
             "The sub-fact still bundles several conditions, or nothing is close — split it into smaller "
             "answerable sub-questions and retrieve each, then fuse.",
             "query"),
    TechCard("Step-back / broaden", "prf",
             "The query is too specific and returns near-duplicates or nothing — take a step back to a more "
             "general formulation, or expand with pseudo-relevance feedback terms from the top results.",
             "query"),
    TechCard("Fielded / title match", "fielded",
             "The missing item is a NAMED entity that should match a document title — search title+text fields "
             "directly instead of relying on semantic similarity.",
             "query"),
    TechCard("Fusion retrieval (RRF)", "arsenal",
             "One retriever alone misses the entity — run keyword + vector + HyDE together and reciprocal-rank "
             "fuse for coverage.",
             "query"),
    TechCard("Cross-encoder reranking", "rerank",
             "A strong match EXISTS in the candidate pool but is ranked low (a big score cliff above it) — "
             "rerank the pool with a cross-encoder to lift it into the top-k.",
             "rank"),
    TechCard("Metadata / self-query filter", "os_query",
             "The sub-fact names a constrainable attribute (a title phrase, a year, a type) — author a raw "
             "OpenSearch query with a phrase match / filter / field boost to isolate exactly that document.",
             "authored"),
    TechCard("Authored OpenSearch DSL", "os_query",
             "None of the packaged retrievers surface the document — have the LLM author a bespoke OpenSearch "
             "query body (phrase match, function_score, boosted fields) targeting the missing sub-fact.",
             "authored"),
    # index-time techniques (kept for completeness, not selectable as a runtime next hop)
    TechCard("Semantic chunking", "index", "Chunk by semantic coherence rather than fixed size.", "context", runtime=False),
    TechCard("Hierarchical indices / RAPTOR", "index", "Summary + detail tiers for abstraction-level retrieval.", "context", runtime=False),
    TechCard("HyPE / document augmentation", "index", "Precompute hypothetical questions per doc at index time.", "context", runtime=False),
    TechCard("Contextual compression", "index", "Compress retrieved chunks to query-relevant spans.", "context", runtime=False),
]

RUNTIME = [c for c in CATALOG if c.runtime]


class SkillLookup:
    """Semantic lookup over the runtime catalog's `when_to_use`, so the judge's diagnosis maps to a
    technique even when it doesn't name one exactly."""

    def __init__(self, embed):
        import numpy as np
        self.cards = RUNTIME
        self.M = np.asarray(embed([c.when_to_use for c in self.cards]), dtype="float32")

    def suggest(self, diagnosis_text: str, top: int = 3):
        import numpy as np
        v = np.asarray(self.embed_one(diagnosis_text), dtype="float32")
        v = v / (np.linalg.norm(v) or 1.0)
        M = self.M / (np.linalg.norm(self.M, axis=1, keepdims=True) + 1e-9)
        order = np.argsort(M @ v)[::-1][:top]
        return [(self.cards[i].name, self.cards[i].technique) for i in order]

    # set by caller
    embed_one = None


def catalog_summary() -> str:
    return "\n".join(f"- {c.name} [{c.technique}]: {c.when_to_use}" for c in RUNTIME)


if __name__ == "__main__":
    print(catalog_summary())
