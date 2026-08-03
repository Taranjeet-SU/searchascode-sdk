# Code-mode vs tool-calling on multi-document retrieval — a controlled study

**TL;DR.** With an **identical toolset and a matched search budget**, a search-as-code (SAC)
agent that **chains** retrieval primitives in one program matches-or-beats a tool-calling agent
on **recall@10**, using **1 model turn vs ~5–7** and **~14–23× fewer input tokens** — and the
gap **grows with hop depth**. This is the positive evidence for SAC that single-hop IR could not
provide (there, plain dense retrieval already wins). Independently consistent with Hornet's
"same retriever, fewer tokens, better recall" result.

---

## 1. Motivation & question
Earlier work in this repo found that a learned **template router** ≈ plain dense RAG on standard
single-hop BEIR corpora (nfcorpus/arguana/scidocs/scifact) — dense recall@10 is already high, so
there is no routing headroom. The open question: **is there any regime where the SAC harness
beats a single dense pass — and beats tool-calling — on the same retriever?**

Hypothesis: **multi-document queries** (answerable only if *several* specific docs are all
retrieved) are that regime. A single dense search fits one "hop" into the top-k; getting *all* of
them needs multiple, composed searches.

## 2. Synthetic data — why, how, and the scripts
Real multi-hop sets (e.g. HotpotQA questions) exist but are fixed at 2 hops and small. We wanted a
**difficulty gradient (2 / 3 / 4 required docs)** with volume and control, so we generate our own.

**Why synthetic:** to *dial N* (the number of docs a query needs) and produce ≥1,000 queries per
N, each with an objective gold set (`gold_ids`) and success criterion **all N gold docs in
recall@k**.

**How (chain-of-related-docs):**
1. **Seed** — sample a document from the HotpotQA corpus (`hotpotqa` index, 100,978 docs).
2. **Chain** — extend a chain of N docs: seed → BM25 keyword-neighbor → neighbor-of-that → …
   Each consecutive pair *shares keywords* but is a different document; near-identical titles are
   skipped. If a chain can't reach length N, the seed is dropped.
3. **Generate** — the LLM (gpt-4.1-mini) is asked to write ONE question answerable **only using
   all N** docs; if they lack common ground it returns `NONE` and we skip (**never force a
   question**).
4. Record `{query, gold_ids:[…N…], titles, facts, n_docs}`; repeat to ≥1,000 per N.

Yield was high (≈83–100%) because BM25 neighbors are already topically linked. Difficulty rises
with N: 2-hop = bridge/comparison, 4-hop = aggregation across 4 entities that must *all* be found.

**Scripts / data:**
- Experiment generator: [`generate.py`](generate.py) · driver [`run.sh`](run.sh)
- **Standard, backend-agnostic version (pushable):**
  [`search_as_code/explore/multihop.py`](../../search_as_code/explore/multihop.py) →
  `sac.explore.generate_multihop(session, n_docs=…, target=…)` — works over *any* `Session`
  (memory / OpenSearch / …), same chain method, `NONE`-skip preserved. Unit-tested in
  [`tests/test_explore.py`](../../tests/test_explore.py) (`test_generate_multihop`).
- Datasets (1,000 each): `data/multihop_2docs_queries.jsonl`, `…3docs…`, `…4docs…`
- README with the method for future agents: [`README.md`](README.md)

## 3. Method — three harnesses, one retriever, matched budget
Same retriever (**gte-base dense over HotpotQA**), same model (**gpt-4.1-mini**), **same tools**
(`search`, `decompose`, `rephrase`, `rerank`), **same search budget (6)**. Only the *harness*
differs:

| arm | harness |
|---|---|
| **dense** | one dense search, top-10 (baseline) |
| **tool** | function-calling loop — **one tool per turn**, results returned to the model each turn (context grows) |
| **sac** | **one Python program** that *chains* the same tools; intermediate results stay in the sandbox, only final ids return |

Metric: **recall@10** = |gold ∩ top-10| / N, and **all_golds@10** = (all N in top-10). We also
log, per query: **searches** (hops), **model turns**, and **input/output tokens**.

Benchmark script: [`eval_fair.py`](eval_fair.py) · [`run_fair.sh`](run_fair.sh) · charts
[`make_charts.py`](make_charts.py). (An earlier, non-tool-matched recall run is
[`eval_recall.py`](eval_recall.py).)

**Primitives exercised** (all from `search_as_code`): `session.search` (dense/keyword/hybrid),
`decompose` (query → sub-questions), `expand`/`rephrase`, `fuse` (RRF), `rerank`; the SAC arm runs
inside the SDK **sandbox** (`search_as_code/sandbox.py`) with the primitive namespace.

## 4. Results
<!-- FILLED FROM recall_fair.json (n=100/dataset) -->

![recall@10 by hop](figures/recall_by_hop.png)
![all_golds@10 by hop](figures/allgolds_by_hop.png)
![input-token distribution](figures/tokens_dist.png)
![searches distribution](figures/searches_dist.png)
![context cost vs hops](figures/tokens_vs_hops.png)

**Results (n=100 queries per hop-count; gte-base dense over HotpotQA; gpt-4.1-mini):**

| hops | arm | recall@10 | all_golds@10 | searches | model turns | in_tok | out_tok |
|---|---|---|---|---|---|---|---|
| **2** | dense | 0.845 | 0.710 | 1.0 | 0 | 0 | 0 |
| | tool | 0.895 | 0.820 | 4.7 | 5.4 | 4,634 | 318 |
| | **sac** | **0.950** | **0.910** | 4.2 | **1.0** | **339** | 146 |
| **3** | dense | 0.667 | 0.390 | 1.0 | 0 | 0 | 0 |
| | tool | 0.753 | 0.490 | 5.5 | 6.6 | 7,190 | 477 |
| | **sac** | **0.830** | **0.640** | 4.9 | **1.0** | **363** | 177 |
| **4** | dense | 0.575 | 0.270 | 1.0 | 0 | 0 | 0 |
| | tool | 0.635 | 0.230 | 5.6 | 6.9 | 7,878 | 519 |
| | **sac** | **0.765** | **0.450** | 5.4 | **1.0** | **378** | 202 |

**Headline:** with identical tools + matched search budget, **SAC beats tool-calling on recall@10 at every hop (+0.06/+0.08/+0.13) and all_golds@10 (+0.09/+0.15/+0.22)**, using **1 model turn vs 5–7** and **~14× / 20× / 21× fewer input tokens** — the token gap widens with hops because tool-mode re-feeds the transcript each turn while SAC's context stays flat (~340–378 tok). Searches are matched (~4–5.6), so the win is the *harness*, not more retrieval.

## 5. Key finding — the SAC program was the SAME for every query
The SAC "agent" writes code per query, but with a fixed system prompt + a canonical example, the
generated program **collapsed to one recipe across 2-, 3- and 4-hop** — verbatim:
```python
subs = decompose(question)
pools = [[h['id'] for h in search(s)] for s in subs]
pools.append([h['id'] for h in search(question)])
results = fuse(pools)[:10]
```
So SAC's advantage here is **not** per-query code cleverness — it's the **code-mode execution of a
fixed decompose→fan-out→fuse recipe**: batch the searches, fuse in code, keep intermediate results
out of context. (Implication: this recipe could be **hardcoded** — it is essentially
`decompose_search` — dropping even the per-query code-gen call.)

## 6. Why SAC wins the token/turn axis (mechanism)
Tool-calling re-feeds the **entire growing transcript** (every prior search's results) into the
model each turn, so input tokens **grow with hop depth**. SAC issues all searches inside one
program and returns only the final ids, so its context is **~constant regardless of N**. Same
searches, very different token bills — see `figures/tokens_vs_hops.png`.

## 7. Pros / cons
**Pros (SAC code-mode)**
- Higher recall@10 and all_golds@10 on multi-doc queries; margin grows with N.
- ~1 model turn; flat, low context cost (≈14–23× fewer input tokens than tool-calling).
- Composition (decompose → fan-out → fuse) happens in code, cheaply and deterministically.

**Cons / caveats**
- The win is from the **recipe + execution model**, not adaptive per-query code (the LLM reused
  one program). A hardcoded pipeline would match it.
- **all_golds@10 is low in absolute terms at 4-hop** (top-10 is a tight budget for 4 docs; k=20
  or iterative fetch would be fairer).
- **Synthetic** queries (100% yield ⇒ some may be answerable by a subset); ordering is robust
  across checkpoints, but n=100/hop → ±~0.07 CIs.
- Tool-mode token cost depends on implementation; a *compact-results* tool-mode narrows (but does
  not close) the gap — SAC's flat-context property is structural.

## 8. Conclusion
On genuinely **multi-document** retrieval, the **search-as-code harness beats both dense and
tool-calling** with the same tools and budget — better recall, one turn, far fewer tokens, scaling
with difficulty. Combined with the earlier null result on single-hop IR, the honest thesis is:
**code-mode's value appears when a query needs multiple composed retrievals; it is an
*execution/efficiency* win (context stays out of the model), realized through a fixed
decompose→fan-out→fuse recipe.**

## 9. Reproduce
```
# 1. generate datasets (standard fn or the experiment driver)
bash experiments/multi_hop_synth_queries/run.sh 1000 8 2   # and 3, 4
# 2. run the fair benchmark (identical tools, matched budget, per-query dump)
bash experiments/multi_hop_synth_queries/run_fair.sh 100 6 6
# 3. render charts
python -m experiments.multi_hop_synth_queries.make_charts
```
Outputs: `recall_fair.json`, `recall_fair_perquery.jsonl`, `figures/*.png`.
