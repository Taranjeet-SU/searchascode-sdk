"""Persist every chatbot/arena query with its full trace, for future reference & analysis.

Appends one JSON line per query to ``chatbot/logs/queries.jsonl`` (gitignored). Each record:
timestamp, agent, query, answer, cited ids, hops, arrived, latency, token usage/cost, and the
agent trace (SAC generated code + consensus agreement, or the tool-call steps).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "logs" / "queries.jsonl"


def log_query(agent: str, query: str, answer, extra: dict | None = None, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent": agent,
        "query": query,
        "answer": getattr(answer, "answer", ""),
        "ids": getattr(answer, "ids", []),
        "hops": getattr(answer, "hops", None),
        "arrived": getattr(answer, "arrived", None),
        "latency_s": getattr(answer, "latency_s", None),
        "usage": getattr(answer, "usage", {}),
        "trace": getattr(answer, "trace", {}),
    }
    if extra:
        rec.update(extra)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def read_log(path: Path = LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
