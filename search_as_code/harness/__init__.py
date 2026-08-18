"""`search_as_code.harness` — a self-improving agentic retrieval harness.

Plan–Execute–Verify loop + query triage + Anthropic-style skills (progressive disclosure) +
cross-session memory + dynamic prompt + subagents + pluggable reward. See ``harness.Harness``.
"""
from __future__ import annotations

from .agentic import agentic_solve
from .context import HarnessContext, HarnessResult, StepResult
from .diagnostic_judge import DIAGNOSTIC_PROMPT, DiagnosticJudge, coverage_signals, parse_verdict
from .forge import CodePrimitive, HarnessForge, HarnessStore, LearnedSkill, LearnedSubagent, reflect
from .harness import BASE_PROMPT, Harness
from .hooks import DEFAULT_POST_HOOKS, DEFAULT_PRE_HOOKS
from .loop import decompose_query, default_verify, fuse_ids, plan_execute_verify
from .memory import AgentMemory, MemoryItem
from .os_query import author_os_query
from .playbook import apply_technique, diagnostic_solve, sf_arsenal
from .rag_techniques import CATALOG, SkillLookup, catalog_summary
from .skills import BUILTIN_SKILLS, Skill, SkillRegistry
from .triage import QueryIntent, extract_codes, triage

__all__ = [
    "Harness", "BASE_PROMPT",
    "AgentMemory", "MemoryItem",
    "Skill", "SkillRegistry", "BUILTIN_SKILLS",
    "triage", "QueryIntent", "extract_codes",
    "HarnessContext", "HarnessResult", "StepResult",
    "plan_execute_verify", "decompose_query", "fuse_ids", "default_verify",
    "DEFAULT_PRE_HOOKS", "DEFAULT_POST_HOOKS",
    "HarnessForge", "HarnessStore", "LearnedSkill", "LearnedSubagent", "CodePrimitive", "reflect",
    "DiagnosticJudge", "DIAGNOSTIC_PROMPT", "coverage_signals", "parse_verdict",
    "agentic_solve", "diagnostic_solve", "apply_technique", "sf_arsenal",
    "SkillLookup", "CATALOG", "catalog_summary", "author_os_query",
]
