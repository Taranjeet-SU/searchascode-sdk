"""ProfilePack — the on-disk artifact produced by ``sac.explore``.

A ProfilePack is a *versioned directory* holding everything the exploration phase
learns about a corpus (schema, content profile, ontology, synthetic queries, a
primitive router, few-shots/templates, prompt overrides). A ``Session`` loads it
at query time to tune retrieval to the data — and works fine without it.

Design goals (robustness):
- **Everything serializes here** — nothing the explorer learns is implicit.
- **Resumable** — each stage writes its own artifact; a crash re-runs only what's missing.
- **Drift-aware** — the manifest stores a corpus *fingerprint*; when the data changes
  enough, stages are re-run.
- **Self-describing** — the manifest records every stage's status/timestamp/summary so a
  human (or agent) can see exactly what was built and what was rejected.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


class ProfilePack:
    """A directory of exploration artifacts + a manifest tracking stage status."""

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest

    # ---- lifecycle -------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path) -> "ProfilePack":
        """Open an existing pack or initialise a fresh one at ``path``."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        mpath = root / MANIFEST_NAME
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
        else:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created": time.time(),
                "fingerprint": None,
                "stages": {},  # name -> {status, ts, seconds, summary, artifacts, note}
            }
        pack = cls(root, manifest)
        pack.save_manifest()
        return pack

    def save_manifest(self) -> None:
        self.manifest["updated"] = time.time()
        (self.root / MANIFEST_NAME).write_text(json.dumps(self.manifest, indent=2, default=str))

    # ---- artifacts -------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return (self.root / name).exists()

    def write_json(self, name: str, obj: Any) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, default=str))

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.path(name)
        return json.loads(p.read_text()) if p.exists() else default

    def write_jsonl(self, name: str, rows: list) -> None:
        with self.path(name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    def read_jsonl(self, name: str) -> list:
        p = self.path(name)
        if not p.exists():
            return []
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]

    # ---- stage bookkeeping ----------------------------------------------
    def stage(self, name: str) -> dict:
        return self.manifest["stages"].get(name, {})

    def stage_status(self, name: str) -> Optional[str]:
        return self.stage(name).get("status")

    def is_done(self, name: str) -> bool:
        """A stage counts as done if it succeeded and its artifacts still exist."""
        st = self.stage(name)
        if st.get("status") != "ok":
            return False
        return all(self.has(a) for a in st.get("artifacts", []))

    def record_stage(self, name: str, status: str, *, seconds: float = 0.0,
                     summary: Optional[dict] = None, artifacts: Optional[list[str]] = None,
                     note: str = "") -> None:
        self.manifest["stages"][name] = {
            "status": status,          # ok | rejected | error | planned | skipped
            "ts": time.time(),
            "seconds": round(seconds, 3),
            "summary": summary or {},
            "artifacts": artifacts or [],
            "note": note,
        }
        self.save_manifest()

    # ---- drift -----------------------------------------------------------
    def set_fingerprint(self, fp: str) -> None:
        self.manifest["fingerprint"] = fp
        self.save_manifest()

    def fingerprint_changed(self, fp: str) -> bool:
        prev = self.manifest.get("fingerprint")
        return prev is not None and prev != fp

    # ---- summary ---------------------------------------------------------
    def report(self) -> str:
        lines = [f"ProfilePack {self.root}  (schema v{self.manifest.get('schema_version')})"]
        for name, st in self.manifest["stages"].items():
            summ = ", ".join(f"{k}={v}" for k, v in (st.get("summary") or {}).items())
            lines.append(f"  [{st.get('status','?'):8}] {name:14} {summ}"
                         + (f"  # {st['note']}" if st.get("note") else ""))
        return "\n".join(lines)
