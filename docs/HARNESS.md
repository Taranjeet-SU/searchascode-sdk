# `search_as_code.harness` — a self-improving agentic retrieval harness

The **harness** is everything wrapped around the raw retrieval primitives that makes a *reliable
agent*. This module implements the four necessary harness elements (loop, tool/skill interface,
context+memory, control) plus the upgrades our own experiments and the 2026 harness-engineering /
Anthropic Agent-Skills / prime-agent *Continual-Harness* patterns point to.

```python
import search_as_code as sac
h = sac.Harness(session)                       # memory + skills auto-wired from the session
r = h.run("Which two films share a director, and what year did each release?")
r.ids              # fused doc ids
r.intent           # 'multi_hop'
r.skill            # 'subagents'  (or the winning single skill)
r.subagents        # [{query, ids, skill}, ...]  — one child agent per sub-question
r.dynamic_prompt   # the pre-loop prompt (intent + recalled memory + skill catalog)
```

## Components

| component | file | what it does |
|---|---|---|
| **Triage** | `triage.py` | rule-based intent detection — `error_code` / `definition` / `entity_factoid` / `multi_hop` / `exploratory` → picks the right skill + depth. Cheap signals (part-number/error-code regex, "compare/and", "what is X"), *not* a learned classifier (our experiments showed that ties dense). |
| **Skills** | `skills.py` | Anthropic-style skills: `name`, `when_to_use`, `cost`, `run`. A `SkillRegistry` with **progressive disclosure** (`summaries()` — a few tokens each) + semantic `find()` (the skill "search API"). 10 built-in retrieval skills (dense/hybrid/keyword/exact/decompose_fuse/hyde/prf/rerank/diversify). Grow the library without bloating the prompt. |
| **Memory** | `memory.py` | `AgentMemory` = **working** (in-session events, bounded) + **long-term** (cross-session, JSONL-persisted, semantic recall). `flush()` promotes durable wins; `recall()` surfaces "what worked on queries like this" into the next run's prompt — the compounding memory a static router lacks. |
| **Control loop** | `loop.py` | bounded **Plan–Execute–Verify**: try planned skills in order, verify each against a **pluggable reward**, keep the best, stop when accepted. `verify` defaults to non-empty but takes a gold check (eval) or a teacher-reranker score (prod) — a *reliable* reward instead of the unreliable self-judge. |
| **Subagents** | `loop.py` + `harness.py` | multi-hop → `decompose_query` → one **child harness per sub-question** (`spawn`), results **RRF-fused** for coverage. Bounded recursion depth. |
| **Hooks + dynamic prompt** | `hooks.py` | pre-loop hooks (triage → recall memory → select skills → **assemble the dynamic prompt**) and post-loop hooks (write outcome + remember the winning skill). Swap/extend hooks to customize the pipeline. |

## How it maps to best practices
- **4 harness elements + Plan-Execute-Verify** — the 2026 harness-engineering consensus (loop / tools / context+memory / control).
- **Skills with progressive disclosure + a registry** — the open Anthropic Agent-Skills pattern.
- **working / long-term memory, write-phase promotion** — standard agent-memory architecture.
- **Triage before spend** — intent-aware routing (factoid vs multi-hop), matching Adaptive-RAG's "right effort per query" (but rule-based, per our findings).
- **Subagents as function calls + self-modifiable memory** — the prime-agent RLM / Continual-Harness shape.

## Honest status (what's done vs the frontier)
**Done:** triage, skills+registry, working+cross-session memory with recall, Plan-Execute-Verify loop
with pluggable reward, subagents for multi-hop, pre/post hooks + dynamic prompt, `sac.Harness`.
**Not yet (the continual-learning frontier — see `open_problems.md`):** (1) a *teacher-scored* reward
wired in by default (we expose the hook; the self-judge is still the fallback); (2) *skill creation*
(the agent synthesizing new skills from observed gaps — we ship a fixed library); (3) *online*
refinement of the dynamic prompt during a run (we write memory post-run, not mid-loop). These are the
prime-agent Continual-Harness capabilities our data argues for next.

Tests: `tests/test_agent_harness.py` (no GPU/LLM — memory backend + dependency-free embedder).
