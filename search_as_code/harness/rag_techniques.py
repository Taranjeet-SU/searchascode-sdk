"""A retrievable catalog of RAG techniques the diagnostic judge looks up to pick the next-hop technique.

Seeded from NirDiamant/RAG_Techniques (https://github.com/NirDiamant/RAG_Techniques): each entry pairs a
`when_to_use` description with an executable technique id. Given the judge's diagnosis of a missing
sub-fact, `SkillLookup.suggest` returns the best technique — turning a fixed menu into an extensible,
literature-grounded playbook. `runtime=False` entries are INDEX-time (listed for completeness, not
selectable as a next hop on a fixed index).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TechCard:
    name: str
    technique: str        # id understood by playbook.apply_technique / os_query authoring
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
             "answerable sub-questions and retrieve each, then fuse.", "query"),
    TechCard("Step-back / broaden", "prf",
             "The query is too specific and returns near-duplicates or nothing — take a step back to a more "
             "general formulation, or expand with pseudo-relevance-feedback terms from the top results.", "query"),
    TechCard("Fielded / title match", "fielded",
             "The missing item is a NAMED entity that should match a document title — search title+text fields "
             "directly instead of relying on semantic similarity.", "query"),
    TechCard("Fusion retrieval (RRF)", "arsenal",
             "One retriever alone misses the entity — run keyword + vector + HyDE together and reciprocal-rank "
             "fuse for coverage.", "query"),
    TechCard("Cross-encoder reranking", "rerank",
             "A strong match EXISTS in the candidate pool but is ranked low (a big score cliff above it) — "
             "rerank the pool with a cross-encoder to lift it into the top-k.", "rank"),
    TechCard("Metadata / self-query filter", "os_query",
             "The sub-fact names a constrainable attribute (a title phrase, a year, a type) — author a raw "
             "OpenSearch query with a phrase match / filter / field boost to isolate exactly that document.",
             "authored"),
    TechCard("Authored OpenSearch DSL", "os_query",
             "None of the packaged retrievers surface the document — have the LLM author a bespoke OpenSearch "
             "query body (phrase match, function_score, boosted fields) targeting the missing sub-fact.", "authored"),
    TechCard("Semantic chunking", "index", "Chunk by semantic coherence rather than fixed size.", "context", runtime=False),
    TechCard("Hierarchical indices / RAPTOR", "index", "Summary + detail tiers for abstraction-level retrieval.", "context", runtime=False),
    TechCard("HyPE / document augmentation", "index", "Precompute hypothetical questions per doc at index time.", "context", runtime=False),
    TechCard("Contextual compression", "index", "Compress retrieved chunks to query-relevant spans.", "context", runtime=False),
]

RUNTIME = [c for c in CATALOG if c.runtime]


def catalog_summary() -> str:
    return "\n".join(f"- {c.name} [{c.technique}]: {c.when_to_use}" for c in RUNTIME)


class SkillLookup:
    """Semantic lookup over the runtime catalog's `when_to_use`, so the judge's diagnosis maps to a
    technique even when it doesn't name one exactly. `embed(list[str]) -> list[list[float]]`."""

    def __init__(self, embed):
        import numpy as np
        self.embed = embed
        self.cards = RUNTIME
        self.M = np.asarray(embed([c.when_to_use for c in self.cards]), dtype="float32")

    def suggest(self, diagnosis_text: str, top: int = 3):
        import numpy as np
        v = np.asarray(self.embed([diagnosis_text])[0], dtype="float32")
        v = v / (np.linalg.norm(v) or 1.0)
        M = self.M / (np.linalg.norm(self.M, axis=1, keepdims=True) + 1e-9)
        order = np.argsort(M @ v)[::-1][:top]
        return [(self.cards[i].name, self.cards[i].technique) for i in order]
