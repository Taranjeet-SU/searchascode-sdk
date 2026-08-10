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
| **Skills** | `skills.py` | Anthropic-style skills: `name`, `when_to_use`, `cost`, `run`. A `SkillRegistry` with **progressive disclosure** (`summaries()` — a few tokens each) + semantic `find()` (the skill "search API"). Built-in retrieval skills incl. dense/hybrid/keyword/exact/hyde/prf/rerank/diversify, plus the **arsenal** — `arsenal_single` (hybrid + HyDE + fielded fused) and `decompose_arsenal` (decompose → arsenal per sub-fact → RRF fuse), the validated multi-hop recipe. Grow the library without bloating the prompt. |
| **Memory** | `memory.py` | `AgentMemory` = **working** (in-session events, bounded) + **long-term** (cross-session, JSONL-persisted, semantic recall). `flush()` promotes durable wins; `recall()` surfaces "what worked on queries like this" into the next run's prompt — the compounding memory a static router lacks. |
| **Control loop** | `loop.py` | bounded **Plan–Execute–Verify**: try planned skills in order, verify each against a **pluggable reward**, keep the best, stop when accepted. `verify` defaults to non-empty but takes a gold check (eval) or a teacher-reranker score (prod) — a *reliable* reward instead of the unreliable self-judge. |
| **Subagents** | `loop.py` + `harness.py` | multi-hop → `decompose_query` → one **child harness per sub-question** (`spawn`), results **RRF-fused** for coverage. Bounded recursion depth. |
| **Hooks + dynamic prompt** | `hooks.py` | pre-loop hooks (triage → recall memory → select skills → **assemble the dynamic prompt**) and post-loop hooks (write outcome + remember the winning skill). Swap/extend hooks to customize the pipeline. |
| **Forge (self-modification)** | `forge.py` | after a solve, the agent **creates/modifies what it learned** and persists it — `create_skill`/`create_primitive` (compose retrievers into a new named recipe), `create_subagent`, `refine_prompt` (append a learned rule to the self-modifiable supplemental prompt), `remember`. `reflect()` runs this **online** after each query; a `HarnessStore` persists skills/subagents/rules so the **next session loads and uses them**. |

## Self-improvement (online, cross-session)

```python
h = sac.Harness(session, store_path="agent_store", learn=True)
h.run("Compare the Agilex 7 transceivers and the Quartus install steps")
#  -> solves, then FORGES: a 'learned_multihop_*' skill (decompose+dense+keyword fused),
#     a sub-agent template, a learned rule ("decompose + FUSE, don't rerank the union"),
#     and a memory win — all persisted to agent_store/.
h2 = sac.Harness(session, store_path="agent_store", learn=True)   # NEW session
#  -> loads the forged skill + learned rules; they show up in h2.skills and every dynamic_prompt.
```
The base prompt stays **immutable**; the forged rules are the **self-modifiable supplemental prompt**
(`LEARNED RULES (…)`), matching the Continual-Harness prefix + refinable-state design.

## How it maps to best practices
- **4 harness elements + Plan-Execute-Verify** — the 2026 harness-engineering consensus (loop / tools / context+memory / control).
- **Skills with progressive disclosure + a registry** — the open Anthropic Agent-Skills pattern.
- **working / long-term memory, write-phase promotion** — standard agent-memory architecture.
- **Triage before spend** — intent-aware routing (factoid vs multi-hop), matching Adaptive-RAG's "right effort per query" (but rule-based, per our findings).
- **Subagents as function calls + self-modifiable memory** — the prime-agent RLM / Continual-Harness shape.

## Honest status (what's done vs the frontier)
**Done:** triage; skills+registry; working+cross-session memory with recall; Plan-Execute-Verify loop
with pluggable reward; subagents for multi-hop; pre/post hooks + dynamic prompt; **the Forge —
online creation of skills / subagents / composed primitives / memory + a self-modifiable supplemental
prompt, persisted and reused across sessions**; `sac.Harness(learn=True)`.

**Not yet (the remaining frontier — see `open_problems.md`):**
1. **LLM-proposed forging.** `reflect()` is rule-based (deterministic, testable); it doesn't yet ask a
   model to *propose* novel skills/primitives from the trajectory (only promotes known-good compositions).
2. **Composition-only primitives.** A forged "primitive" composes *existing* retrievers (a safe recipe
   DSL) — the agent can't yet author a brand-new *atomic* retrieval algorithm as sandboxed code.
3. **Teacher-scored reward by default.** The reward is pluggable; the default is a non-empty heuristic,
   not a frontier-teacher score (Continual-Harness process-reward co-learning).
4. **Mid-loop refinement.** Forging happens post-solve; not yet *during* the loop.

Tests: `tests/test_agent_harness.py` (no GPU/LLM — memory backend + dependency-free embedder).
