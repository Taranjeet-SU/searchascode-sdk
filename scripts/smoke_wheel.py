#!/usr/bin/env python3
"""Build the wheel, install it into a clean venv, and exercise the README quickstart.

This is the control for STR-1. `search_as_code/` sits at the repo root, so pytest and every
experiment import the WORKING COPY — nothing ever validated the artifact `pip install
search-as-code` actually delivers. That is not hypothetical: it is the mechanism behind DOC-1,
where `phase1/sac_surface.py::SAC_SYSTEM` (which docs/SELECTION.md calls the LLM-facing
surface) is excluded from the wheel and no test noticed.

Flask makes this impossible by construction with a `src/` layout. Short of moving the package,
this job is the equivalent guarantee.

    python3 scripts/smoke_wheel.py [--keep]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every symbol the README's quickstart and capability table promise a base install provides.
QUICKSTART = r"""
import search_as_code as sac

# README: "the base install ships a dependency-free embedder + in-memory backend"
backends = sac.available()
assert "memory" in backends, backends

s = sac.Session("memory")
s.add([{"id": "1", "text": "how do agents retrieve documents?"},
       {"id": "2", "text": "reciprocal rank fusion merges ranked lists"},
       {"id": "3", "text": "an unrelated cooking recipe"}])

# the three modes the capability table says work on every backend
for mode in ("dense", "keyword", "hybrid"):
    hits = s.search("how do agents retrieve?", top_k=2, mode=mode)
    assert len(hits) >= 1, mode

# the README's headline snippet
cands = s.search_many(["how do agents retrieve?", "agentic RAG"], top_k=3, mode="hybrid")
ev = cands.to_evidence(fields=["title"])
assert isinstance(ev, list)

# primitives advertised on the front page must be importable from the top level
for name in ("fan_out", "fuse", "rerank", "decompose", "rephrase", "dedup", "mmr",
             "recall_at_k", "bootstrap_ci"):
    assert hasattr(sac, name), f"{name} missing from the installed wheel"

# the sandbox is the product's core claim — it must work from an install
ex = sac.LocalExecutor(s)
res = ex.run("results = sac.search('agents', top_k=2)\n")
assert res.ok, res.error

print("WHEEL SMOKE OK  version=%s  backends=%d" % (sac.__version__, len(backends)))
"""


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main(argv: list[str]) -> int:
    keep = "--keep" in argv
    work = Path(tempfile.mkdtemp(prefix="sac-wheel-"))
    dist = work / "dist"
    try:
        run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "build"])
        run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)], cwd=ROOT)
        wheels = sorted(dist.glob("*.whl"))
        if not wheels:
            print("no wheel produced", file=sys.stderr)
            return 1
        wheel = wheels[-1]
        print(f"built {wheel.name} ({wheel.stat().st_size // 1024} KiB)")

        venv = work / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        py = venv / "bin" / "python"
        run([str(py), "-m", "pip", "install", "--quiet", str(wheel)])

        script = work / "quickstart.py"
        script.write_text(QUICKSTART)
        # cwd=/ so the repo's working copy cannot shadow the installed package — the whole
        # point of the check.
        run([str(py), str(script)], cwd="/")
        print("\nPASS: the built wheel installs clean and the README quickstart runs.")
        return 0
    finally:
        if keep:
            print(f"(kept {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
