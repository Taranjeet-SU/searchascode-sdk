"""Run the zero-setup examples as a smoke test.

`examples/` was two loose scripts that CI never executed, so if `demo.py` broke nothing said so
— while the README leads with it as the zero-setup entry point (issues.md STR-11 / EX-1). Flask's
examples are installable projects exercised by its own CI; this is the equivalent guarantee.

Examples that need a live backend or an API key are skipped, not silently passed.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ZERO_SETUP = ["01_quickstart.py", "03_explore_first.py", "04_harness_judge_forge.py"]


@pytest.mark.parametrize("name", ZERO_SETUP)
def test_example_runs(name, capsys):
    path = EXAMPLES / name
    assert path.exists(), f"{name} is referenced by the docs but missing"
    argv = sys.argv[:]
    sys.argv = [str(path)]
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert out.strip(), f"{name} produced no output"


def test_every_example_is_either_zero_setup_or_documented_as_needing_a_service():
    """No orphan examples: each file is either smoke-tested here or explicitly service-bound."""
    needs_service = {"02_opensearch.py"}
    on_disk = {p.name for p in EXAMPLES.glob("*.py")}
    assert on_disk == set(ZERO_SETUP) | needs_service, (
        f"examples/ changed: {on_disk ^ (set(ZERO_SETUP) | needs_service)} — add it to "
        f"ZERO_SETUP or to needs_service")
