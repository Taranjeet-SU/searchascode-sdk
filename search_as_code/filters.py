"""A tiny, portable metadata-filter language.

Agent code always writes filters in one Mongo-ish dialect; each adapter either
translates it to its native filter DSL or, when the backend can't filter
server-side, we evaluate it client-side with :func:`matches`.

    {"lang": "en"}                        # equality shorthand
    {"year": {"$gte": 2020}}              # operators
    {"tag": {"$in": ["cve", "advisory"]}}
    {"$and": [{"lang": "en"}, {"year": {"$gte": 2020}}]}
"""

from __future__ import annotations

from typing import Any, Mapping

from .errors import InvalidFilterError

_OPS = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
    "$in": lambda a, b: a in b,
    "$nin": lambda a, b: a not in b,
    "$exists": lambda a, b: (a is not None) == bool(b),
    "$contains": lambda a, b: b in a if a is not None else False,
}


def matches(metadata: Mapping[str, Any], flt: Mapping[str, Any] | None) -> bool:
    """Evaluate a filter against a metadata dict (client-side emulation)."""
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(metadata, c) for c in cond):
                return False
        elif key == "$or":
            if not any(matches(metadata, c) for c in cond):
                return False
        elif key == "$not":
            if matches(metadata, cond):
                return False
        elif isinstance(cond, Mapping) and any(k.startswith("$") for k in cond):
            value = metadata.get(key)
            for op, operand in cond.items():
                fn = _OPS.get(op)
                if fn is None:
                    raise InvalidFilterError("unknown filter operator", operator=op)
                if not fn(value, operand):
                    return False
        else:
            if metadata.get(key) != cond:
                return False
    return True


_LOGICAL = {"$and", "$or"}


def validate(flt: Mapping[str, Any] | None) -> None:
    """Raise :class:`InvalidFilterError` for malformed filters (unknown operators,
    bad logical structure). Called at the boundary so every backend — including
    server-side adapters that would otherwise silently drop bad operators — fails
    fast with a clear, typed error."""
    if not flt:
        return
    if not isinstance(flt, Mapping):
        raise InvalidFilterError("filter must be a mapping", got=type(flt).__name__)
    for key, cond in flt.items():
        if key in _LOGICAL:
            if not isinstance(cond, (list, tuple)):
                raise InvalidFilterError(f"{key} expects a list of filters", operator=key)
            for c in cond:
                validate(c)
        elif key == "$not":
            validate(cond)
        elif isinstance(cond, Mapping) and any(k.startswith("$") for k in cond):
            for op in cond:
                if op not in _OPS:
                    raise InvalidFilterError("unknown filter operator", operator=op)


def normalize(flt: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expand equality shorthand into explicit ``$eq`` so adapters translate a
    single canonical form."""
    if not flt:
        return {}
    out: dict[str, Any] = {}
    for key, cond in flt.items():
        if key in ("$and", "$or"):
            out[key] = [normalize(c) for c in cond]
        elif key == "$not":
            out[key] = normalize(cond)
        elif isinstance(cond, Mapping) and any(k.startswith("$") for k in cond):
            out[key] = dict(cond)
        else:
            out[key] = {"$eq": cond}
    return out
