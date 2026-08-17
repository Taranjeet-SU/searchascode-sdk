"""Query triage — classify *what kind* of query this is, so the harness spends the right effort.

The recurring lesson from our experiments: routing to *one of 16 templates* ties dense, but the
few genuinely different cases (IDs/error-codes, vocabulary gaps, multi-fact) are **detectable by
cheap signals**. So triage is rule-based intent detection, not a learned classifier — it decides the
plan (single lookup vs decompose+fuse vs exact) from surface features.

Intents:
  - ``error_code``   : contains an error/status code or part-number token → exact/regex lookup
  - ``definition``   : "what is / define / meaning of X" → single focused lookup
  - ``entity_factoid``: "who/where/when is X" single-entity attribute → single lookup
  - ``multi_hop``    : needs several facts combined ("and", "compare", "relate", multiple clauses)
  - ``exploratory``  : open-ended / broad → wide pool, maybe diversify
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# error/status codes and part numbers: ORA-01017, HTTP 500, E1234, 0x80070005, ERR_CONN, AGFC012
_CODE_RE = re.compile(
    r"\b(?:0x[0-9a-fA-F]{2,}"                        # hex codes
    r"|[A-Z]{2,}[-_ ]?\d{2,}[A-Z0-9]*"              # ORA-01017 / ERR_1234 / E404
    r"|[A-Z]{2,}\d*_[A-Z0-9_]+"                      # ERR_CONN_REFUSED
    r"|\b\d{3}\b(?=\s*(?:error|status|code)))\b"     # 500 error / status 404
)
_HTTP_RE = re.compile(r"\b(?:error|status|exception|code|errno|traceback|stack ?trace)\b", re.I)
_DEF_RE = re.compile(r"^\s*(?:what\s+(?:is|are|does|do)|define|definition of|meaning of|explain)\b", re.I)
_ENTITY_RE = re.compile(r"^\s*(?:who|where|when|which)\s+(?:is|are|was|were)\b", re.I)
# STRONG multi-document signals: the question explicitly contrasts or enumerates several
# DOCUMENTS. A bare "and" is deliberately NOT here — it is the single most common word in a
# conjunctive-constraint question ("a film released in 1994 AND directed by X"), which needs
# ONE document, not a decomposition. Classifying those as multi_hop routed them to
# decompose_arsenal, the structure experiments/deep_judge/README.md §5 measures as wrong for
# conjunctive corpora (decompose 0.025 vs dense 0.079 recall@10 on BrowseComp) — SDK-A4.
_MULTI_HINT = re.compile(
    r"\b(?:both|compare|comparison|versus|vs\.?|difference between|relate[sd]?|relationship|"
    r"how (?:do|does|are).*(?:relate|connect|differ)|respectively|as well as|"
    r"each of the|which of the (?:two|three))\b", re.I)

# WEAK hints — conjunctions that only suggest multi-document when several distinct named
# entities are also present. "and"/"each" live here.
_WEAK_MULTI_HINT = re.compile(r"\b(?:and|each)\b", re.I)
# A proper-noun-ish run, used to tell "two entities joined by and" from "one entity with
# several constraints joined by and".
_ENTITY_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’\-]{2,}(?:\s+[A-Z][A-Za-z0-9'’\-]{2,})*\b")


def _n_named_entities(q: str) -> int:
    """Count distinct capitalised multi-word runs, ignoring a leading sentence capital."""
    body = q[1:] if q[:1].isupper() else q
    return len({m.group(0) for m in _ENTITY_TOKEN_RE.finditer(body)})


@dataclass
class QueryIntent:
    kind: str                                   # error_code | definition | entity_factoid | multi_hop | exploratory
    signals: dict = field(default_factory=dict)
    recommended_skill: str = "dense_lookup"
    depth: str = "single"                       # single | multi
    confidence: float = 0.5

    def __str__(self) -> str:
        return f"{self.kind}(depth={self.depth}, skill={self.recommended_skill}, conf={self.confidence:.2f})"


def extract_codes(query: str) -> list[str]:
    """Error/status codes and part-number tokens in the query (drives exact/regex lookup)."""
    seen, out = set(), []
    for m in _CODE_RE.findall(query):
        m = m.strip()
        if m and m not in seen:
            seen.add(m); out.append(m)
    return out[:4]


def _n_clauses(q: str) -> int:
    return len([c for c in re.split(r"[,;]|\band\b|\bor\b", q) if c.strip()])


def triage(query: str) -> QueryIntent:
    """Classify a query into an intent + recommended skill + depth, from cheap surface signals."""
    q = (query or "").strip()
    codes = extract_codes(q)
    n_entities = _n_named_entities(q)
    sig = {"len": len(q), "n_words": len(q.split()), "codes": codes,
           "has_error_word": bool(_HTTP_RE.search(q)), "n_clauses": _n_clauses(q),
           "multi_hint": bool(_MULTI_HINT.search(q)),
           "weak_multi_hint": bool(_WEAK_MULTI_HINT.search(q)),
           "n_entities": n_entities}

    # 1) multi-hop: a STRONG contrastive/enumerating hint, or a weak conjunction that comes
    #    with several distinct named entities (two documents), or many clauses AND several
    #    entities. A long conjunctive-constraint question about ONE entity stays single-depth
    #    and is handled whole — which is what the BrowseComp evidence supports (SDK-A4).
    #    Checked BEFORE the code branch: "the difference between BM25 and dense retrieval"
    #    trips the part-number regex on "BM25", and a comparison of two coded entities is
    #    still a two-document question.
    strong = sig["multi_hint"]
    weak_with_entities = sig["weak_multi_hint"] and n_entities >= 2
    clausal_with_entities = sig["n_clauses"] >= 3 and n_entities >= 2
    if strong or weak_with_entities or clausal_with_entities:
        conf = 0.85 if (strong and n_entities >= 2) else (0.7 if strong else 0.6)
        return QueryIntent("multi_hop", sig, "decompose_arsenal", "multi", conf)

    # 2) error / status codes → exact match wins (embeddings blur IDs)
    if codes or (sig["has_error_word"] and re.search(r"\d", q)):
        return QueryIntent("error_code", sig, "exact_lookup", "single", 0.9 if codes else 0.6)

    # 3) definition / factoid → one focused lookup
    if _DEF_RE.match(q):
        return QueryIntent("definition", sig, "definition_lookup", "single", 0.75)
    if _ENTITY_RE.match(q):
        return QueryIntent("entity_factoid", sig, "dense_lookup", "single", 0.7)

    # 4) long / broad / open-ended → exploratory (wide pool)
    if sig["n_words"] >= 18 or sig["n_clauses"] >= 2:
        return QueryIntent("exploratory", sig, "hybrid_search", "single", 0.55)

    return QueryIntent("entity_factoid", sig, "dense_lookup", "single", 0.5)
