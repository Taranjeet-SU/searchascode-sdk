"""Agent memory — working (in-session) + long-term (cross-session, persisted), with semantic recall.

Follows the standard split (working / episodic / semantic; see 2026 agent-memory best practices):
- **working** memory: transient events within the current run (queries, plans, results). Bounded.
- **long-term** memory: facts/experiences/what-worked persisted to disk (JSONL) and recalled
  semantically across sessions — the compounding "what strategy worked on queries like this" store
  that a static router lacks.

The write phase (``flush``) promotes durable facts from working → long-term. Recall is embedding-
cosine if an embedder is given, else lexical overlap.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class MemoryItem:
    content: str
    kind: str = "fact"                  # fact | event | outcome | skill_win | note
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)
    vec: Optional[list] = None          # cached embedding (long-term only)


def _as_embed(embedder) -> Optional[Callable]:
    if embedder is None:
        return None
    if callable(embedder):
        return embedder
    if hasattr(embedder, "embed"):
        return embedder.embed
    return None


class AgentMemory:
    """In-session working memory + cross-session long-term memory with semantic recall."""

    def __init__(self, path: Optional[str] = None, embedder=None, working_cap: int = 50):
        self.path = Path(path) if path else None
        self._embed = _as_embed(embedder)
        self.working: list[MemoryItem] = []
        self.longterm: list[MemoryItem] = []
        self.working_cap = working_cap
        if self.path and self.path.exists():
            self.load()

    # ---- in-session (working) ----------------------------------------------
    def observe(self, content: str, kind: str = "event", **meta) -> None:
        """Record a transient in-session event (query, plan, result, error)."""
        self.working.append(MemoryItem(content=content, kind=kind, meta=meta))
        if len(self.working) > self.working_cap:
            self.working = self.working[-self.working_cap:]

    def working_context(self, max_chars: int = 1200, kinds: Optional[set] = None) -> str:
        """The recent working memory as a compact string for the prompt (most recent last)."""
        items = [w for w in self.working if kinds is None or w.kind in kinds]
        lines, total = [], 0
        for w in reversed(items):
            line = f"- [{w.kind}] {w.content}"
            if total + len(line) > max_chars:
                break
            lines.append(line); total += len(line)
        return "\n".join(reversed(lines))

    # ---- cross-session (long-term) -----------------------------------------
    def remember(self, content: str, kind: str = "fact", **meta) -> MemoryItem:
        """Persist a durable fact/experience to long-term memory (embedded if possible)."""
        item = MemoryItem(content=content, kind=kind, meta=meta)
        if self._embed:
            try:
                item.vec = list(np.asarray(self._embed([content])[0], dtype=np.float32))
            except Exception:
                item.vec = None
        self.longterm.append(item)
        if self.path:
            self.save()
        return item

    def recall(self, query: str, k: int = 5, kind: Optional[str] = None) -> list[MemoryItem]:
        """Recall the k most relevant long-term items for a query (semantic, else lexical)."""
        pool = [m for m in self.longterm if kind is None or m.kind == kind]
        if not pool:
            return []
        if self._embed and all(m.vec is not None for m in pool):
            try:
                qv = np.asarray(self._embed([query])[0], dtype=np.float32)
                qv = qv / (np.linalg.norm(qv) + 1e-9)
                M = np.asarray([m.vec for m in pool], dtype=np.float32)
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
                order = np.argsort(-(M @ qv))[:k]
                return [pool[i] for i in order]
            except Exception:
                pass
        qtok = set(query.lower().split())
        scored = sorted(pool, key=lambda m: -len(qtok & set(m.content.lower().split())))
        return scored[:k]

    def flush(self, kinds: Optional[set] = None, summarize: Optional[Callable] = None) -> int:
        """Promote durable working items → long-term (the memory write phase). Returns #promoted."""
        keep = kinds or {"outcome", "skill_win", "fact"}
        promote = [w for w in self.working if w.kind in keep]
        if summarize and promote:
            text = summarize([w.content for w in promote])
            self.remember(text, kind="note", promoted=len(promote))
        else:
            for w in promote:
                self.remember(w.content, kind=w.kind, **w.meta)
        self.working = [w for w in self.working if w.kind not in keep]
        return len(promote)

    # ---- persistence -------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for m in self.longterm:
                d = asdict(m)
                d["vec"] = list(map(float, m.vec)) if m.vec is not None else None
                f.write(json.dumps(d) + "\n")

    def load(self) -> None:
        self.longterm = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self.longterm.append(MemoryItem(content=d["content"], kind=d.get("kind", "fact"),
                                            ts=d.get("ts", time.time()), meta=d.get("meta", {}),
                                            vec=d.get("vec")))

    def stats(self) -> dict:
        return {"working": len(self.working), "longterm": len(self.longterm),
                "embedded": sum(1 for m in self.longterm if m.vec is not None)}
