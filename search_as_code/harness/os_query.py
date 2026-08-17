"""LLM-authored OpenSearch queries as a next-hop technique.

When the packaged retrievers miss a sub-fact, the judge can suggest `os_query`: the LLM authors a raw
OpenSearch query body (phrase match, field boosts, function_score) targeting exactly that sub-fact. The
body is VALIDATED before execution — only read-only query clauses are allowed (no scripts / no
aggregations / bounded size) — then run via the read-only `store._search`. A validate-and-retry loop
surfaces real errors back to the LLM. Successful bodies are captured so the forge can persist them as
authored primitives (Phase B/C).
"""
from __future__ import annotations

import json
import re
from typing import Sequence

# read-only clause whitelist — the only keys allowed anywhere in the authored body
_ALLOWED = {"query", "bool", "must", "should", "must_not", "filter", "match", "match_phrase",
            "multi_match", "term", "terms", "range", "prefix", "wildcard", "fuzzy", "boosting",
            "positive", "negative", "negative_boost", "function_score", "field_value_factor",
            "functions", "boost", "boost_mode", "score_mode", "minimum_should_match", "fields",
            "query", "type", "operator", "fuzziness", "slop", "gte", "lte", "gt", "lt", "value",
            "field", "factor", "modifier", "missing", "tie_breaker", "analyzer", "lenient"}
_BANNED = {"script", "script_score", "aggs", "aggregations", "_source", "size", "from", "sort", "pit"}

_SYSTEM_TMPL = """You write ONE OpenSearch query BODY (a JSON object) to retrieve the single document \
that answers a sub-fact. {schema_line} Use match_phrase for exact names/titles, multi_match over the \
text fields with a boost on the most title-like one for named entities, bool/should to combine, and \
function_score/field_value_factor only if needed. Only these fields exist — do NOT invent field names: \
{fields}. Return ONLY the JSON for the value of "query" (do NOT include size/sort/_source/aggs/scripts). \
Example shape: {example}"""


def describe_fields(store) -> list[str]:
    """The queryable field names of this index, from the store's own schema (SDK-A7).

    ``describe_schema`` reports ``fields`` on OpenSearch and ``metadata_keys`` on memory
    (SDK-C14), so both are accepted. The configured text field is always included, and the
    vector field is always excluded — you cannot write a lexical clause against it.
    """
    names: list[str] = []
    try:
        schema = store.describe_schema() or {}
    except Exception:
        schema = {}
    raw = schema.get("fields") or schema.get("metadata_keys") or {}
    names = list(raw.keys()) if isinstance(raw, dict) else list(raw)
    text_field = getattr(store, "text_field", None) or "text"
    if text_field not in names:
        names.append(text_field)
    vector_field = getattr(store, "vector_field", None)
    return [n for n in names if n and n != vector_field]


def build_author_system(fields: Sequence[str]) -> str:
    """The author prompt for THIS index — real field names, no hardcoded corpus (SDK-A6).

    Previously this asserted ``text``/``title`` exist and showed a HotpotQA example. On a
    corpus without a ``title`` field (BrowseComp) the authored body matched nothing and
    silently burned the retry budget.
    """
    fields = list(fields) or ["text"]
    title_like = next((f for f in fields if "title" in f.lower() or "name" in f.lower()), None)
    primary = title_like or fields[0]
    others = [f for f in fields if f != primary][:4]
    boosted = [f"{primary}^2", *others] if others else [primary]
    example = json.dumps({"bool": {"should": [
        {"match_phrase": {primary: "<an exact name or phrase>"}},
        {"multi_match": {"query": "<the key terms>", "fields": boosted}},
    ]}})
    schema_line = (f"The index exposes the lexical field(s) {', '.join(repr(f) for f in fields)}."
                   if fields else "")
    return _SYSTEM_TMPL.format(schema_line=schema_line, fields=", ".join(fields), example=example)


# Kept for backwards compatibility with callers that imported the constant directly; the
# schema-derived prompt from build_author_system() is what author_os_query actually uses.
AUTHOR_SYSTEM = build_author_system(["title", "text"])


def _validate(qbody: dict, fields: Sequence[str] = ()) -> tuple[bool, str]:
    """Reject anything that is not a read-only clause, parameter, or known field name.

    The previous condition ended in ``not isinstance(k, str)`` — always False for JSON object
    keys — so the whole allowlist branch was unreachable and only ``_BANNED`` was enforced,
    contradicting the module docstring's "only read-only query clauses are allowed" (SDK-C1).
    """
    known = set(fields)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in _BANNED:
                    return f"banned key: {k}"
                if not k.startswith("_") and k not in _ALLOWED and k not in known:
                    return f"unexpected key: {k}"
                e = walk(v)
                if e:
                    return e
        elif isinstance(o, list):
            for v in o:
                e = walk(v)
                if e:
                    return e
        return None
    if not isinstance(qbody, dict) or not qbody:
        return False, "query must be a non-empty JSON object"
    err = walk(qbody)
    return (err is None), (err or "")


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def author_os_query(store, gen, subfact: str, top_k: int = 30, gold=None, tries: int = 2, log=print):
    """Returns (ids, body, ok). ok=True if it ran (and, when gold given, retrieved a gold).
    Only applies to stores exposing a raw `_search` (OpenSearch); degrades to empty otherwise."""
    if not hasattr(store, "_search"):
        return [], None, False
    # Introspect before authoring: the field names come from the index, not from a hardcoded
    # assumption that `title` exists (SDK-A6/A7, docs/INTROSPECTION.md's introspect-first rule).
    fields = describe_fields(store)
    system = build_author_system(fields)
    feedback = ""
    for attempt in range(tries):
        prompt = f"SUB-FACT to retrieve: {subfact}"
        if feedback:
            prompt += f"\n\nYour previous body failed: {feedback}\nFix it."
        raw = gen.complete(prompt, system=system)
        qbody = _extract_json(raw)
        if qbody is None:
            feedback = "could not parse JSON"
            continue
        ok, err = _validate(qbody, fields)
        if not ok:
            feedback = f"validation failed: {err}"
            continue
        try:
            res = store._search({"size": top_k, "query": qbody})
            hits = res.get("hits", {}).get("hits", [])
            ids = [str(h["_id"]) for h in hits]
        except Exception as e:  # surface the real OpenSearch error for the retry
            feedback = f"execution error: {type(e).__name__}: {str(e)[:160]}"
            continue
        if not ids:
            feedback = "returned 0 results — broaden or fix field names"
            continue
        if gold is not None:
            g = set(str(x) for x in gold)
            if not (g & set(ids[:top_k])):
                feedback = "ran but did not retrieve the target — try a phrase/title match"
                # keep the body but mark not-ok if it never hits gold
                if attempt == tries - 1:
                    return ids, qbody, False
                continue
        return ids, qbody, True
    return [], None, False
