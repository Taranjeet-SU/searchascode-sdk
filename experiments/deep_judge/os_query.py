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

# read-only clause whitelist — the only keys allowed anywhere in the authored body
_ALLOWED = {"query", "bool", "must", "should", "must_not", "filter", "match", "match_phrase",
            "multi_match", "term", "terms", "range", "prefix", "wildcard", "fuzzy", "boosting",
            "positive", "negative", "negative_boost", "function_score", "field_value_factor",
            "functions", "boost", "boost_mode", "score_mode", "minimum_should_match", "fields",
            "query", "type", "operator", "fuzziness", "slop", "gte", "lte", "gt", "lt", "value",
            "field", "factor", "modifier", "missing", "tie_breaker", "analyzer", "lenient"}
_BANNED = {"script", "script_score", "aggs", "aggregations", "_source", "size", "from", "sort", "pit"}

AUTHOR_SYSTEM = """You write ONE OpenSearch query BODY (a JSON object) to retrieve the single document \
that answers a sub-fact. The index has text fields `text` and `title` (a dense `vector` field exists but \
you write LEXICAL queries here). Use match_phrase for exact names/titles, multi_match over [title, text] \
with a title boost (title^2) for named entities, bool/should to combine, and function_score/field_value_factor \
only if needed. Return ONLY the JSON for the value of "query" (do NOT include size/sort/_source/aggs/scripts). \
Example: {"bool":{"should":[{"match_phrase":{"title":"The Cardboard Crown"}},{"multi_match":{"query":"Australian novel series royal title","fields":["title^2","text"]}}]}}"""


def _validate(qbody: dict) -> tuple[bool, str]:
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in _BANNED:
                    return f"banned key: {k}"
                if not k.startswith("_") and k not in _ALLOWED and not isinstance(k, str):
                    return f"unexpected key type: {k}"
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
    feedback = ""
    for attempt in range(tries):
        prompt = f"SUB-FACT to retrieve: {subfact}"
        if feedback:
            prompt += f"\n\nYour previous body failed: {feedback}\nFix it."
        raw = gen.complete(prompt, system=AUTHOR_SYSTEM)
        qbody = _extract_json(raw)
        if qbody is None:
            feedback = "could not parse JSON"
            continue
        ok, err = _validate(qbody)
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
