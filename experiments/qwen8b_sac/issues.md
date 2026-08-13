# issues.md — running the standard SAC explore+forge+full pipeline with Qwen3-Embedding-8B

Live log of friction / bugs hit while driving BrowseComp + SU through the **standard** SAC pipeline
(`run_explore_pipeline.py` → `run_forged_on_full.py`) with Qwen3-Embedding-8B (4096-d) instead of the
default gte-base (768-d). Newest issues appended at the bottom. Severity: 🟥 blocker · 🟧 friction · 🟨 minor.

## Setup / model
- Retriever: `Qwen/Qwen3-Embedding-8B` (4096-d), query instruction handled by the pipeline's embedder.
- BrowseComp: OpenSearch index `browsecomp_qwen8b` (100195 docs, already embedded this session).
- SU: `~/scripts/data/su_docs_2.csv` (396 docs) embedded in-memory by `build("su")`; 150 multihop queries.
- Generation/judge LLM: OpenAI `gpt-4.1-mini` (API — no GPU). GPU budget = Qwen3-8B (~16 GB) + reranker.

## Issues

### 🟧 1. Pipeline output paths are hardcoded to `experiments/deep_judge/` (`HERE`)
`run_explore_pipeline.py` / `run_forged_on_full.py` write `forge_store_{corpus}_explored/`,
`explore_{corpus}_memory.jsonl`, `explore_pipeline_{corpus}.json`, `explore_full_{corpus}.json` under their
own dir, keyed only on `corpus`. Re-running `browsecomp`/`su` with a *different embedder* silently clobbers
the gte-base artifacts and offers no per-experiment isolation.
**Handled:** added backward-compatible env overrides `SAC_EXP_DIR` (output dir) and `SAC_TAG` (artifact tag)
so this experiment writes into `experiments/qwen8b_sac/` with a `*_qwen8b` tag. Defaults preserve old behavior.

### 🟧 2. `build()` hardcodes the gte-base dimension (`dim=common.DIM` = 768)
`run_forge_playbook.build("su", ...)` creates the in-memory store with `dim=common.DIM`, so a 4096-d
embedder produces vectors that mismatch the store dimension.
**Handled:** added `SAC_DIM` env override (default `common.DIM`) consumed by `build()`'s SU memory store.

### 🟨 3. SU re-embeds all 396 docs in-process on every pipeline invocation
Unlike BrowseComp (pre-embedded to OpenSearch), `build("su")` re-embeds the corpus each run with the live
embedder. Cheap at 396 docs (~seconds) but O(corpus) and repeated for explore + full; would not scale to a
large SU corpus without an OpenSearch/persistent path like BrowseComp's.

### 🟥 4. The standard pipeline embeds queries PLAIN — Qwen3-8B's query instruction is never applied
`embed = lambda t: em.encode(list(t), normalize_embeddings=True)` is symmetric: the **same** embedder serves
(a) query search in `session.search`, (b) HyDE (embeds a generated *document*), and (c) doc indexing in
`build()`. Qwen3-Embedding needs its query instruction (`Instruct: …\nQuery:{q}`) on the **query side only**;
we measured this is worth a lot on BrowseComp — dense Recall@10 **0.149 (plain) → 0.277 (instruct)**, R@5
0.106 → 0.200. A single symmetric `Session.embedder` can't express that asymmetry: prefixing it would also
(wrongly) instruction-prefix HyDE's hypothetical documents and the indexed corpus.
**Decision for this run:** run the standard pipeline **as-is (plain)** so it stays faithful and the
forged-vs-dense comparison is fair (both arms share one embedder — which is exactly what the explore/forge
experiment measures: does SAC augmentation help *on top of this retriever?*). Absolute recall is therefore
Qwen3-8B's *plain* mode, not its instructed ceiling; the instructed dense reference (R@5 0.200 / R@10 0.277,
from `deep_judge/repro_dense_extended.json`) is reported alongside so the gap is visible.
**SDK follow-up:** `Session` needs a query-side instruction distinct from doc/HyDE embedding (a
`query_instruction` or separate `query_embedder`) to use instruction-tuned retrievers at full strength.

### 🟥 5. SU explore OOM'd — pipeline never caps `max_seq_length`, 8B defaults to 32K context
First SU run crashed: `torch.OutOfMemoryError: Tried to allocate 7.31 GiB … 25.35 GiB in use`. Cause: the
pipeline builds `SentenceTransformer(BC_EMB)` without setting `max_seq_length`; Qwen3-Embedding-8B's default
is **32768 tokens**, so `build("su")` embedding full-length SU docs in-process blew up activation memory.
BrowseComp dodged this only because its corpus was pre-truncated to 512 tokens at index time (by
`embed_and_index.py`), while SU embeds live through the pipeline.
**Fixed:** pipeline now sets `em.max_seq_length = SAC_MAX_SEQ` (default **512**, matching the paper's
"512 tokens across all methods" protocol) in both `run_explore_pipeline.py` and `run_forged_on_full.py`.
Relaunched SU explore after the fix → passed (GPU peak ~20 GB vs 32 GB, no OOM); SU full also clean.
**SDK follow-up:** the pipeline should set a sane default context cap (or truncate corpus docs) rather than
inheriting a retriever's 32K default.

### 🟥 6. Explore/forge cannot discover "plain dense is best" — no dense baseline, no selection gate
On BrowseComp the forged primitive (0.103 R@10) *lost* to plain dense (0.149). Root cause is a design gap,
not a fluke:
- The forged primitive does **hybrid (dense+BM25) → cross-encoder rerank** (see
  `forge_store_browsecomp_qwen8b_explored/code_primitives.jsonl`). On a strong 8B retriever both steps hurt:
  hybrid injects lexical/BM25 noise on a reasoning corpus, and the small MS-MARCO cross-encoder is weaker
  than the 8B embedder so it demotes good dense hits (documented domain-mismatch caveat).
- **Why explore didn't pick dense:** (a) the forge prompt literally templates whole-query as
  "dense/hybrid + rerank", so the LLM never authors bare dense; (b) the forge acceptance gate is only
  `mean recall@20 > 0` — it never compares the primitive to dense; (c) dense is computed only *after* forging
  in `run_forged_on_full` (a report, not a selection signal), so explore never sees a dense baseline.
- Stage-1's apparent win (0.175 > dense 0.149) was **oracle-stopping**; judge-stop (0.147 ≈ 0.149) already
  showed no real gain, and the single-shot forge fell to 0.103.
**SDK fix:** add a plain-dense arm to explore + a selection gate — accept the forged primitive only if it
beats dense on the held set; otherwise emit `dense` (or `dense→rerank` only if rerank helps *this* retriever)
as the primitive. This is what would let explore *discover* that light dense wins on a strong retriever.

## Outcome
All four stages completed with the fixes above: BrowseComp explore+full and SU explore+full, Qwen3-8B, one
GPU-resident model at a time. Issues #1/#2/#5 were **fixed** (env overrides + context cap); #3 is minor
(SU re-embed, cheap here); #4 is the substantive open SDK gap (query instruction can't be applied in the
standard symmetric-embedder pipeline) — the run stays faithful to the standard pipeline and reports the
instructed dense reference alongside. OpenSearch stayed healthy throughout (no crashes this run; the
heartbeat's OS-restart watchdog was armed but not needed). See `README.md` for results.
