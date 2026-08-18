"""Back-compat shim.

The prompt surface moved into the shipped package (``search_as_code.surface``) so that a
``pip install`` actually contains the LLM-facing surface the docs describe — see issues.md
DOC-1. This module re-exports it unchanged for the benchmark harness and the experiment
modules that already import ``phase1.sac_surface``.
"""
from search_as_code.surface import *          # noqa: F401,F403
from search_as_code.surface import (          # noqa: F401  (explicit, for star-import-averse tools)
    JUDGE_SYSTEM,
    SAC_DEEP_RETRY_TEMPLATE,
    SAC_DEEP_SYSTEM,
    SAC_RETRY_TEMPLATE,
    SAC_SYSTEM,
    TOOLCALL_RETRY_TEMPLATE,
    TOOLCALL_SYSTEM,
    TOOLCALL_TOOLS,
)
