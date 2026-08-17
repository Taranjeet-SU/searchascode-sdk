"""Normalizing the ``generate(prompt) -> list[str] | str`` generator contract.

The SDK's generator slot is documented as ``generate(prompt) -> list[str]``, but adapters
disagree on what an element *is*: some return one element per completion, while the adapter
used throughout this repo (``phase1.llm.LLM.as_generator``) splits a single completion into
**lines**. A consumer that wants the whole answer must therefore **join**, not index — taking
``out[0]`` silently truncates a line-splitting generator to its first line, which is how a
"2-6 sub-question" decomposition became a 1-sub-question one and a 4-6 line corpus profile
became a single line.

Use :func:`gen_text` when the prompt asks for prose (a passage, a JSON object, a profile) and
:func:`gen_lines` when it asks for a list (sub-questions, paraphrases). Both accept either
shape, so a consumer is correct against any adapter.

See ``issues.md`` GEN-1 / GEN-2 / GEN-3.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["gen_text", "gen_lines"]

_MARKER_RE = re.compile(r"^[-*•\d.)\s]+")


def gen_text(out: Any, default: str = "") -> str:
    """Return the generator's **full** response as one string, whatever shape it came in.

    A list is joined with newlines (not indexed), so a line-splitting adapter round-trips to
    the original completion instead of losing every line after the first.
    """
    if out is None:
        return default
    if isinstance(out, (list, tuple)):
        parts = [str(x) for x in out if str(x).strip()]
        return "\n".join(parts) if parts else default
    text = str(out)
    return text if text.strip() else default


def gen_lines(out: Any, *, max_items: int | None = None, min_len: int = 0,
              strip_markers: bool = True) -> list[str]:
    """Return the generator's response as a clean list of non-empty lines.

    Handles both adapter shapes: a list is *joined then re-split*, so an adapter that already
    split into lines and one that returned a single multi-line string yield the same result.

    ``strip_markers`` removes leading bullets/numbering; ``min_len`` drops fragments shorter
    than N characters; ``max_items`` truncates.
    """
    text = gen_text(out)
    lines = []
    for raw in text.splitlines():
        line = _MARKER_RE.sub("", raw).strip() if strip_markers else raw.strip()
        if line and len(line) > min_len:
            lines.append(line)
    return lines[:max_items] if max_items is not None else lines
