# explore_forge — explore as a self-improving Forge loop

The explore pipeline no longer *only* trains an XGBoost router. It now **explores to solve, then
forges reusable modules from the wins** — the prime-agent Continual-Harness "second iteration",
applied to retrieval.

## The loop (`run_forge.py`)
On synthetic multi-hop data (2/3/4-hop, gold **known**):

1. **EXPLORE** — for each TRAIN query, `run_sac(oracle_gold=gold, max_retries=9, hint=…)` writes
   **raw OpenSearch queries** (keyword / hybrid / hyde / per-sub-fact decompose → fuse — *not* a
   canned recipe), **oracle-guided** (gold is the stop signal), **up to 10 depths**, using the FULL
   harness: `AgentMemory` recall injected as the **dynamic prompt** each run, and sub-question
   decomposition. It keeps going until all gold docs are found.
2. **FORGE** — from the wins, `HarnessForge` creates a composed **skill** (`explored_multihop` =
   decompose+dense+keyword fused), a **subagent** (`sub_fact`), and a **learned prompt-rule**,
   persisted to `forge_store/` (`skills.jsonl` / `subagents.jsonl` / `learnings.md`).
3. **REPLICATE** — a fresh `sac.Harness(store=forge_store)` on **held-out** queries uses **only the
   forged modules** (memory recall biases the plan → the forged skill is selected), no free
   exploration.
4. **XGBoost router** — still trained on the same queries (`do_xgb=1`), so explore does *both*.

## Validation (per_hop=2 → explore 6, replicate 6; HotpotQA multi-hop)
| stage | metric | value |
|---|---|---|
| **Explore** (oracle, ≤10 depths, raw queries, full harness) | all_golds@10 solve-rate | **0.667** (avg 4.0 hops) |
| **Forge** | artifacts created | `explored_multihop` skill · `sub_fact` subagent · 1 learned rule |
| **Replicate** (forged modules only, no exploration) | all_golds@10 | **0.667** · **6/6 queries used the forged skill** |

**The forged module reproduces the exploration's success** (0.667 → 0.667) at a fraction of the cost
(one composed skill vs up to 10 oracle-guided hops). That is the point: explore *discovers* the
retrieval strategy by writing raw queries against the corpus, then *bottles it* as a reusable skill +
subagent + prompt-rule that later queries apply directly.

## Files
- `run_forge.py` — the pipeline. `forge_results.json` — the numbers above.
- `forge_store/` — the persisted forged skills / subagents / learned rules (loadable by any `Harness`).
- Harness pieces used: `search_as_code/harness/` (`run_sac` oracle exploration; `AgentMemory`,
  `HarnessForge`, `HarnessStore`, `Harness`). See `docs/HARNESS.md`.

## Coverage finding — the "unsolvable" 4-hop queries were under-retrieval, not a ceiling
Digging into the ~33% of 4-hop queries the first run left unsolved: the golds are **reachable**, and
the failure was **under-using the retrieval arsenal**, not a coverage ceiling.
- Exact **title** match finds every gold — but the title *is* the answer (oracle-only, not usable).
- The **query's own** distinctive phrases retrieve only 2/4–3/4 golds; the rest are described *too
  generically* ("an Australian novel series") for any phrase/keyword to isolate.
- The **full arsenal — decompose × {hybrid + HyDE + fielded}, RRF-fused** — reaches them: **HyDE**
  hallucinates the answer doc for a generic description and retrieves the missing entity (e.g.
  *The Cardboard Crown* via HyDE+hybrid). Wiring this into the harness (`decompose_arsenal` /
  `arsenal_single` per sub-fact) lifted the multi-hop solve-rate **0.67 → 0.83** on the same set;
  the residual misses are slot-competition (all golds in top-30, not all in top-10 → needs rerank/k=20).

The forged multi-hop skill/subagent now use the arsenal (`decompose_arsenal` / `arsenal_single`).

## Honest status
- Exploration uses the SDK's atomic query surface (which *is* the OpenSearch query layer) — raw DSL
  via `store._search(body)` is available but not yet the default exploration channel.
- `reflect()`/forge decides *what* to create by **rule** (promotes the known decompose+fuse win); the
  **LLM-proposes-novel-skills** version is the remaining frontier (see `open_problems.md`).
- Small validation (n=6/6); scale `per_hop` up as a background run to firm up the numbers.
