"""Shared context objects passed through the harness pipeline (pre-hooks → loop → post-hooks)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StepResult:
    skill: str
    ids: list
    ok: bool = False
    score: float = 0.0


@dataclass
class HarnessResult:
    ids: list
    intent: str = ""
    skill: str = ""                     # winning skill (or "subagents")
    score: float = 0.0
    steps: list = field(default_factory=list)        # list[StepResult] — the control-loop trace
    subagents: list = field(default_factory=list)     # sub-task results, if any
    dynamic_prompt: str = ""            # the assembled pre-loop prompt (for an LLM agent)
    meta: dict = field(default_factory=dict)


@dataclass
class HarnessContext:
    """Everything a hook / the loop needs; mutated in place through the pipeline."""
    query: str
    session: Any = None                 # SDK Session (retrieval)
    memory: Any = None                  # AgentMemory
    skills: Any = None                  # SkillRegistry
    generator: Any = None
    top_k: int = 10
    intent: Any = None                  # QueryIntent (set by triage hook)
    plan: list = field(default_factory=list)          # ordered skill names to try
    recalled: list = field(default_factory=list)      # long-term memory items recalled
    prompt_parts: list = field(default_factory=list)  # assembled into dynamic_prompt
    result: Optional[HarnessResult] = None
    scratch: dict = field(default_factory=dict)
