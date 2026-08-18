<h1 align="center">Search as Code</h1>

<p align="center"><b>Your LLM writes retrieval code. Your corpus teaches it what works.<br>One <code>pip install</code>, any vector database.</b></p>

<p align="center">
  <a href="https://pypi.org/project/search-as-code/"><img alt="pypi" src="https://img.shields.io/pypi/v/search-as-code"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="backends" src="https://img.shields.io/badge/backends-opensearch·qdrant·chroma·pgvector·faiss·sqlite·memory-orange">
</p>

---

**Search as Code (SaC)** replaces the fixed `search()` endpoint with a **primitive SDK the
LLM programs against**: one model turn writes a short Python strategy — search modes,
fan-out, RRF fusion, rerank, HyDE, MMR, dedup — executed in a sandbox where the bulky
intermediate results **never enter the model context**. A **continual harness** then makes
it corpus-specific: `explore` discovers what retrieval strategy your index rewards,
`forge` persists it as an inspectable, provenance-carrying primitive, and a **deep
LLM-as-judge** decides per query whether one cheap search was enough or the system should
go deeper — up to 10 authored hops.

Perplexity calls this architecture [the future of search for agents](https://research.perplexity.ai/articles/rethinking-search-as-code-generation).
**This is the open, bring-your-own-index version** — same agent code over OpenSearch,
Qdrant, Chroma, pgvector, FAISS, SQLite, or in-memory.

## 📊 The numbers (measured, with intervals — not vibes)

Tool-calling agents pay per hop: every result flows back through the context window.
Code-mode pays once. On **BrowseComp-Plus** (100k docs, Qwen3-Embedding-8B, n=100,
identical tools and search budget for both arms, gpt-4.1-mini):

| arm | model turns | input tokens/query | latency | recall@10 | all-golds@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| tool-calling agent | 9.5 | 19,369 | 32.0 s | 0.147 | 0.100 |
| Search as Code (one authored program) | 1.0 | 648 | 13.7 s | 0.125 | 0.060 |
| **SAC + explore (runs the vetted strategy)** | **1.0** | **689 (28× less)** | **2.5 s (13× faster)** | **0.221 (+50%)** | **0.140** |
| SAC product flow (judge-gated, ≤5 hops) | 6.0 | 14,159 | 66.7 s | 0.242 | 0.130 |
| tool product flow (same policy via tools) | 44.3 | 91,015 (6.4× more) | 143.4 s | 0.215 | 0.130 |
| *dense single-shot (the baseline)* | *0* | *0* | *0.1 s* | *0.265* | *0.160* |

The same judge-gated policy costs **14k tokens in code-mode vs 91k through tool-calling** —
execution model is the cost story even when the policy is identical. And on corpora where the
baseline retriever is *weak*, the product flow **beats it**: dense 0.036 → sac_product 0.054
(+50% relative, replicated twice on the gte-base index).

<!-- GRAPH: docs/assets/cost_tokens.png — tokens per query, tool vs sac vs sac_explored -->
<!-- GRAPH: docs/assets/cost_hops.png — input tokens vs hop depth, tool vs sac -->
<!-- SLOT:HOTPOT_COST — per-hop-depth cost table (2/3/4-hop), tokens grow for tools, flat for SaC -->

Two things this table shows. First, **explore's corpus knowledge compounds in code-mode**:
the vetted-strategy arm reaches 84% of dense's recall at 689 tokens and 2.5 seconds — one
turn, no flailing. Second — read the dense row — **a strong single-shot retriever still tops
raw recall here.** We print that on purpose, because it is the design:

### What explore learned, and what the gate enforced

On this corpus (BrowseComp-Plus, a strong 8B embedder), `explore` authored strategies for
30 training queries across up to 10 hops each — including raw OpenSearch DSL probes on
exact constraints — and discovered: **keep the query whole (0/30 decomposed), and nothing
it authored beat one plain dense search.** The acceptance gate then did its job and
recorded `selected_strategy: dense` — the vetted primitive for this corpus *is* the
baseline. Deploy the gate's selection and your floor is the dense row above (0.265), at
one cheap model turn for the queries that need escalation and zero turns for those that
don't.

The agent rows above are the harness running *standalone* (every query, no baseline
hop-0) — the worst case, shown so you can see the cost mechanics. And one lesson we
publish because we hit it ourselves: an earlier version of this benchmark seeded the
agents with the forge store's *candidate* strategy instead of the **gate's selection** —
the rejected multi-mode fusion — and recall diluted on every query. **The gate's decision
is the product of explore. Deploy that, not the candidate it rejected.**
(`HarnessForge.accept_code_primitive` persists the winning side under the requested name,
so stores written through the gate can't make this mistake.)

The guarantee, stated plainly: **SaC never *ships a strategy* that underperforms your
baseline.** Every forged primitive must beat `max(dense, hybrid)` on held-out queries —
paired bootstrap confidence intervals — or the baseline ships under the same name. That
gate is in the SDK with tests, and we publish it firing. At runtime the guarantee is
*judge-conditional*: PASS returns the vetted baseline verbatim; escalation trades a
measured, bounded risk for measured gains — see the tier table below for exactly how we
know the bound. A retrieval system that can prove it won't make things worse is worth
more than one that claims magic.

### The escalation tiers (and the receipts)

The runtime is a ladder of increasingly expensive mechanisms, each engaged only when the
one below it can't decide:

1. **Hop 0 — the vetted strategy** (gate-selected; often just dense). Zero/near-zero LLM cost.
2. **Convergence stop** [4]: if extra hops stop changing the fused top-k, stop free of charge.
3. **The deep judge** (sufficiency premise [3]): PASS → return hop-0 verbatim; FAIL → a
   structured diagnosis (which fact is missing, why, which technique next) steers…
4. **Authored escalation** (≤5 hops): the LLM writes retrieval code per hop — including raw
   OpenSearch DSL on exact constraints — with skills lookup and cross-query memory.
5. **Verified selection** (the RLM tier [1]): where retrieval *scores* cannot recognize the
   answers, sub-models **read** the pooled candidates against the query's constraints
   (`llm_map`), and verified documents take the final slots.

We publish the iteration history that produced this design, including what failed — on
BrowseComp-Plus, whose defining property is that golds score poorly on every retrieval
signal:

| stop/fusion mechanism | result | verdict |
|---|---|---|
| coverage-checklist judge + plain weighted RRF | 0.242 vs dense 0.265; fusion evicted solved-query golds | shipped baseline, superseded |
| sufficiency-premise prompt [3] alone | escalation rate unchanged (1.00) — LLM judges resist prompt-side calibration [2] | measured null |
| universal CE floor guard (τ=0.5) | missed weak-CE golds (kept 6/10 solved) | rejected |
| per-corpus calibrated CE gate (TASR-style [4]) | best τ separates solved/unsolved at only 0.602 balanced accuracy here | rejected, kept as negative result |
| CE-replace fusion (head-to-head displacement) | −0.036 vs dense, CI [−0.122, +0.052] — statistical tie | superseded |
| **verified selection (sub-LM reads candidates)** [1] | **exact per-query tie with dense (Δ=0.0000, n=60), floor 11/11 solved queries kept** | **ships** |

Every row has a runnable artifact behind it. This is what "self-auditing retrieval" means
in practice: the failures are part of the documentation, because they are how you know the
successes are real.

The punchline: **reading beats scoring.** Five signal-based mechanisms could not protect the
baseline's answers on a corpus whose golds defeat every retrieval score; one pass of sub-model
*reads* held the floor perfectly. On corpora where the baseline is weak, the same escalation
ladder gains +50% relative recall; where the baseline is strong, verified selection guarantees
you keep it. Deeper gains on sparse-gold corpora need the full recursive tier [1] — that is the
roadmap, stated plainly, not a promise buried in a footnote.

## 📦 Install

```bash
pip install search-as-code                 # core: in-memory backend, zero further deps (numpy only)
pip install 'search-as-code[opensearch]'   # + your backend (also: qdrant / chroma / pgvector)
```

```python
import search_as_code as sac

s = sac.Session("opensearch", index="docs", dim=768, embedder=my_embedder)
#   swap "opensearch" -> "qdrant" / "chroma" / "pgvector" / "memory" — nothing else changes

cands = s.search_many(["how do agents retrieve?", "agentic RAG"], top_k=40, mode="hybrid")
best  = s.rerank("how do agents retrieve?", cands, top_k=10)
print(best.to_evidence(fields=["title"]))   # compact, context-friendly — this is ALL the model sees
```

Every example below runs offline with zero setup and no API key (a scripted generator
stands in for the LLM), and CI executes them so they cannot rot:

```bash
python examples/01_quickstart.py           # the primitive API + sandbox
python examples/02_opensearch.py           # a real backend
python examples/03_explore_first.py        # explore: discover the corpus strategy
python examples/04_harness_judge_forge.py  # the continual harness end to end
```

## 🔁 The continual harness: explore → forge → judge

This is the part no managed search API gives you: **the system learns your corpus and
keeps what it learned as code you can read.**

### 1. `explore` — discover what your index rewards

Point it at your corpus with a handful of labeled (or synthesized) queries. The LLM
authors a retrieval strategy per hop — including **raw OpenSearch DSL** probes
(`match_phrase` on a year, a part number, a proper name — the exact constraints dense
embeddings blur) — and the oracle records which strategies actually won. Structure is
**discovered, not assumed**: on entity-dense corpora it learns to keep queries whole; on
genuinely multi-document questions it learns to decompose and fuse.

### 2. `forge` — persist the win as a primitive, with provenance

The winning strategies are synthesized into a named, reusable primitive + skill +
few-shot exemplars, persisted to disk. Every artifact carries **provenance**: when it was
forged, what it was measured on, what baseline it beat and by how much (with the CI), and
what it superseded. Overwrites archive the old version. Contradictory learned rules are
retired, not accumulated.

```python
forge = sac.HarnessForge(store, registry)
report = forge.accept_code_primitive(          # the acceptance gate
    "my_corpus_strategy", "when to use it", authored_code,
    session=s, held=held_queries)              # beats max(dense, hybrid) or the baseline ships
print(report["accepted"], report["delta_vs_baseline"])
```

### 3. The deep LLM-as-judge — depth only when it pays

Most queries need one cheap search. Some need ten hops. The `DiagnosticJudge` reads
per-sub-fact cross-encoder coverage of the current candidates and decides: **PASS** (stop,
you have it) or **FAIL + a structured diagnosis** — which sub-fact is missing, why, which
technique to try next, what query to use. That diagnosis steers the next authored hop.

Validated leak-free (query-grouped splits, shipped renderer, grouped bootstrap):
**0.771 [0.666, 0.870]** balanced accuracy against the oracle, and judge-stopped runs
recover **95–103% of oracle-stopped recall** on HotpotQA and SearchUnify multi-hop
corpora. We also publish the no-LLM references it must beat (a tuned threshold gate
scores 0.738) — if a cheap gate ever matches it on your corpus, use the cheap gate.

```python
res = sac.agentic_solve(session, query, generator=llm)   # judge decides when to stop
res["ids"], res["hops"], res["codes"]                    # results + the authored strategies
```

## 🧰 The primitive surface

| group | primitives |
|---|---|
| retrieval modes | dense (ANN) · keyword (BM25) · hybrid (RRF) · phrase/proximity · fuzzy · wildcard · prefix · fielded · more-like-this · **raw OpenSearch DSL** |
| query-side | rephrase · expand · decompose · HyDE · PRF (Rocchio) · auto_filter (self-query) |
| rank / fuse | cross-encoder rerank · RRF / weighted / relative-score fusion · MMR · semantic dedup · diversity quota · reserve-per-subfact |
| gating | score_cutoff · confidence / abstain · score_cliff · consensus (with agreement signals that survive chaining) |
| introspection | describe_schema · seeded sample · content_type · describe(llm=True) |

Full catalog: [`docs/PRIMITIVES.md`](docs/PRIMITIVES.md) · per-backend support/emulation
matrix: [`docs/DATABASES.md`](docs/DATABASES.md). If a backend lacks a mode, the SDK
emulates it so agent code stays portable; where emulation is approximate, it says so
instead of pretending.

## 🧠 How it works

```
LLM writes Python  ─▶  Session (unified API, out-of-context state)
                        ├─ primitives: fan_out · fuse · rerank · hyde · mmr · dedup …
                        ├─ VectorStore protocol + capability emulation   ← DB differences end here
                        └─ adapters: opensearch · qdrant · chroma · pgvector · faiss · sqlite · memory
                   ─▶  Sandbox (timeouts, output caps; only final evidence returns to the model)
                   ─▶  Judge: PASS → done (1 turn) · FAIL → diagnosis → next authored hop (≤10)
                   ─▶  Forge: winning strategies → provenance-carrying primitives, reused next query
```

The database owns retrieval; the harness owns everything around it. Composite macros stay
bypassable, so generated code can always reach the atoms.

## 🔍 Why trust these numbers

This project maintains a standing, adversarial audit of its own claims — every headline
number carries a bootstrap confidence interval, negative results are published, and
retracted claims stay retracted. The summary lives in [`AUDIT.md`](AUDIT.md); the full
research trail (experiments, defect log, methodology corrections) lives in the companion
research repository. If you find a number this repo can't back, that's a bug — file it.

## 🗺️ Roadmap

- **GEPA-tuned judges per corpus** [10] — the judge prompt becomes a forge artifact behind the same acceptance gate
- **Full recursive RLM mode** [1] — one model turn authors a program that loops all hops inside the sandbox, spawning sub-model reads recursively (shipped today: `llm_map` + verified selection; the recursion is next)
- **StopGate interface** — judge / logistic / threshold gates swappable and A/B-able on your corpus (the measurement harness for this exists; see the tier table)
- **More adapters** — Pinecone, Weaviate, LanceDB · **MCP server wrapper** · hardened sandbox backends (Docker/e2b)

*Shipped since 0.1.0's first cut:* asymmetric query/passage embedders (`Session(query_embedder=…)`),
`coverage_fuse`, `llm_map`, the convergence stop, the sufficiency-premise judge, subagent runtime,
AST-based structure attribution, and `Session.reset_state`.

## 🤝 Contributing

```bash
git clone <this repo> && pip install -e '.[dev]'
make check          # lint + typecheck + tests + doc links — one target, same as CI
```

Add a backend by implementing one `VectorStore`
([`adapters/base.py`](search_as_code/adapters/base.py)); the parametrized conformance
suite ([`tests/test_conformance.py`](tests/test_conformance.py)) is the contract — it runs
against every installed backend in CI and has caught real bugs in three adapters.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 📚 References

The ideas this SDK builds on, cited where they shaped the design:

1. **Recursive Language Models** — Zhang, Kraska, Khattab (2025). Context-as-object, recursive
   sub-model calls; 91.3% on BrowseComp-Plus where retrieval-scored approaches stall — the basis
   of our verified-selection tier and `llm_map`. [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
2. **LLM-as-judge calibration limits** — production judge configurations plateau near 0.65 AUROC at
   detecting false success; motivates signals-first stop gates over prompt tuning.
   [arXiv:2606.09863](https://arxiv.org/abs/2606.09863)
3. **Sufficient context** — Joren et al. (Google). "Can the answer plausibly be derived?" beats
   per-sub-claim coverage checking as the stop question — our judge's sufficiency premise.
   [arXiv:2411.06037](https://arxiv.org/abs/2411.06037)
4. **TASR: training-free adaptive stopping** — answer convergence + calibrated margin retains ~95%
   of quality at ~63% of the calls — our convergence stop. [arXiv:2606.13814](https://arxiv.org/abs/2606.13814);
   see also **Stop-RAG** (value-based stopping) [arXiv:2510.14337](https://arxiv.org/abs/2510.14337)
5. **Adaptive-RAG** — route cheap-first, escalate on demand; routing's real win is cost.
   [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)
6. **Search as Code** — Perplexity's argument that agents should program search primitives rather
   than call a monolithic endpoint; this SDK is the open, bring-your-own-index counterpart.
   [research.perplexity.ai](https://research.perplexity.ai/articles/rethinking-search-as-code-generation)
7. **BrowseComp-Plus** — the sparse-gold benchmark used throughout.
   [arXiv:2508.06600](https://arxiv.org/abs/2508.06600)
8. **Reciprocal Rank Fusion** — Cormack, Clarke, Büttcher (SIGIR 2009) — `fuse`/`rrf`.
9. **Maximal Marginal Relevance** — Carbonell & Goldstein (SIGIR 1998) — `mmr`.
10. **GEPA: reflective prompt evolution** — the planned per-corpus judge-tuning mechanism.
    [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)

## 📄 License

Apache-2.0 © 2026 search-as-code contributors.

<p align="center"><sub>search as code · agentic retrieval · code-mode · continual harness · RAG · vector search · LLM agents</sub></p>
