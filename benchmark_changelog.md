# Benchmark changelog & plan

Living log of all benchmarking activity for the **search-as-code** harness. Every
benchmark has an ID, a metric, a method, a **status**, and a **results** block
that is filled in when it runs. Newest results are appended under each item.

**Status legend:** ⬜ planned · 🟡 running · ✅ done · ⚠️ partial/blocked · ❌ failed

## Environment (captured at run time)
- Host: Linux, Python 3.13 (`.venv-dummy`), CPU + **NVIDIA RTX 5090** GPU.
- OpenSearch **2.17.1**, single node on `:9200`, index `fiqa` = **57,638 docs**.
- Embedder: `thenlper/gte-base` (768-d, GPU). Reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2`.
- Agent LLM: `gpt-4.1-mini` (OpenAI), key from `~/taxonomy/.env`.
- Harness: `benchmarks/bench.py` (subcommands below) + `phase1/benchmark.py` (agent paths).
- Each run writes raw JSON to `benchmarks/results/` and a summary here.

## How to reproduce
```bash
pip install -e '.[phase1]'                       # deps
python -m benchmarks.bench scalability           # Section A
python -m benchmarks.bench throughput            # Section B
python -m benchmarks.bench micro                 # Section E2
python -m benchmarks.bench resilience            # Section E3
python -m benchmarks.bench embedding             # Section E4
python -m phase1.benchmark -n 10 \
  --reranker cross-encoder/ms-marco-MiniLM-L-12-v2   # Sections C, D, E1
```

---

## Section A — Scalability
How the system behaves as corpus size and ingest volume grow.

| ID | Benchmark | Metric | Backend | Status |
|----|-----------|--------|---------|:--:|
| A1 | Ingest throughput vs batch size | docs/sec | OpenSearch | ✅ |
| A2 | Query latency vs corpus size (1k/10k/50k) | ms/query (p50/p95) | memory | ✅ |
| A3 | Index build time + memory footprint vs corpus size | s, MB | memory | ✅ |
| A4 | Fan-out (`search_many`) scaling vs #queries | total s, speedup | memory | ✅ |

**Results**
- A1 ✅ OpenSearch bulk ingest (5,000 docs, 64-d): batch=100 → **7,474 docs/s** · batch=500 → 8,096 · batch=1000 → **8,584 docs/s**. Larger batches help modestly (+15% from 100→1000); the batched-upsert change pays off.
- A2 ✅ in-memory dense query latency (brute-force cosine, p50/p95): 1k → **0.21 / 0.21 ms** (5,377 qps) · 10k → 2.58 / 2.92 ms (384 qps) · 50k → **14.9 / 16.7 ms** (65 qps). Latency scales ~linearly with corpus (brute-force) — fine for dev/small corpora; use OpenSearch HNSW for large ones.
- A3 ✅ in-memory build + footprint: 1k → 3 ms / 0.5 MB · 10k → 20 ms / 5.1 MB · 50k → **105 ms / 25.6 MB** (≈512 B/doc at 128-d float32). Linear, cheap.
- A4 ✅ `search_many` fan-out (10k corpus): 1 q → 37.4 ms/q · 4 q → 3.79 · 8 q → 3.73 · **16 q → 3.51 ms/q** — thread fan-out amortizes per-query cost ~10× vs serial (first call includes matrix build).

---

## Section B — Throughput (APIs per second)
Sustained query rate the retrieval layer can serve.

| ID | Benchmark | Metric | Backend | Status |
|----|-----------|--------|---------|:--:|
| B1 | Single-thread QPS per mode (dense/keyword/hybrid/regex) | queries/sec, ms/query | OpenSearch (fiqa) | ✅ |
| B2 | Concurrent QPS vs worker count (1/2/4/8/16) | queries/sec, p95 ms | OpenSearch (fiqa) | ✅ |

**Results** (live `fiqa`, 57,638 docs; 300 queries/mode)
- B1 ✅ single-thread QPS (p50): **keyword 558 qps** (1.73 ms) · **dense 377 qps** (2.64 ms) · **hybrid 84 qps** (11.9 ms — runs dense+keyword then RRF) · **regex 8.4 qps** (119 ms — scans the `.keyword` subfield, inherently costly). Guidance: reach for regex only on genuinely exact-token needs.
- B2 ✅ concurrent dense QPS: 1 → 402 · 2 → 770 · **4 → 969 (peak)** · 8 → 918 · 16 → 867. Scales ~2.4× to 4 workers, then plateaus/declines (single OpenSearch node + client GIL). Sweet spot ≈ 4–8 concurrent.

---

## Section C — AI-agent latency
End-to-end latency of the three retrieval paths (base / MCP tool-calling / SAC code-mode).

| ID | Benchmark | Metric | Status |
|----|-----------|--------|:--:|
| C1 | Per-path end-to-end latency over N queries | mean, p50, p95 s | ✅ |
| C2 | Per-hop latency + hop-count distribution (LLM paths) | s/hop, #hops | ✅ |

**Results** (`phase1.benchmark -n 8`, gpt-4.1-mini, FiQA)
- C1 ✅ mean end-to-end latency: **base 0.037 s** (no LLM) · **SAC 6.17 s** · **tool-calling 12.24 s**. **SAC is ~2× faster than MCP tool-calling** — one code program vs many serial tool round-trips.
- C2 ✅ LLM calls/query (proxy for hops): **SAC 2.38** vs **tool-calling 5.38** — SAC makes ~55% fewer model round-trips because intermediate results stay in the sandbox.

---

## Section D — Token consumption & cost
LLM economics per path (the code-mode efficiency thesis).

| ID | Benchmark | Metric | Status |
|----|-----------|--------|:--:|
| D1 | Input / output / cached tokens per query, per path | tokens, $/query | ✅ |
| D2 | Prompt-cache hit rate (SAC stable prefix) | % cached input | ✅ |

**Results** (`-n 8`, totals over 8 queries)
- D1 ✅ **SAC:** 24,564 input / 2,943 output tokens, **$0.0108 total (~$0.00135/query)**. **tool-calling:** 22,558 input / 3,897 output, **$0.015 total (~$0.00188/query)**. base: $0. **SAC is ~28% cheaper per query** than tool-calling despite similar input volume (cheaper because most input is cache-billed).
- D2 ✅ prompt-cache hit rate: **SAC 51.1%** (12,544 of 24,564 input tokens cache-billed — the stable `SAC_SYSTEM` prefix) vs **tool-calling 4.5%**. This is the core code-mode efficiency lever, measured directly from `usage.prompt_tokens_details.cached_tokens`.

---

## Section E — Quality, primitives & reliability
Retrieval quality (the ceiling), primitive micro-throughput, and resilience overhead.

| ID | Benchmark | Metric | Status |
|----|-----------|--------|:--:|
| E1 | Retrieval quality per path | Recall@10 / nDCG@10 / MRR@10 | ✅ |
| E2 | Primitive micro-throughput (fuse/mmr/semantic_dedup/rerank/score_cutoff) | ops/sec, ms/call | ✅ |
| E3 | Resilience overhead (retry wrapper, batched vs single upsert) | µs overhead, docs/sec | ✅ |
| E4 | Embedding throughput (gte-base, GPU) | texts/sec | ✅ |

**Results**
- E1 ✅ retrieval quality (`-n 8`, small sample — directional): **Recall@10** — SAC **0.500**, base 0.4375, tool-calling 0.4375. **nDCG@10** — tool-calling **0.452**, SAC 0.357, base 0.343. **MRR@10** — tool-calling **0.500**, SAC 0.356, base 0.354. SAC leads recall; tool-calling leads ranking on this tiny sample. ⚠️ N=8 is noisy — the 100-query run (README) shows SAC 0.491 R@10; rerun with `-n 100` for stable quality numbers.
- E2 ✅ (pool=200 hits, mean over 300 calls): `confidence` 166k ops/s · `score_cutoff` 125k · `dedup`/`diversity_quota` 71k · `fuse` 9.4k · `relative_score_fusion` 6.7k · `rerank(lexical)` 1.3k (0.75 ms) · **`mmr` 161 ops/s (6.2 ms)** · **`semantic_dedup` 73 ops/s (13.6 ms, embeds each hit)**. Takeaway: pure-rank/score primitives are effectively free; the vector/embedding primitives (mmr, semantic_dedup) dominate cost — apply them only to a trimmed pool.
- E3 ✅ `with_retry` overhead **0.14 µs/call** (direct 0.02 → wrapped 0.16); `chunked()` **45.1M items/sec**. Resilience wrappers are negligible on the hot path.
- E4 ✅ gte-base embedding throughput (RTX 5090, 2,000 texts): batch 32 → 8,333 texts/s · batch 128 → 11,979 · **batch 256 → 12,343 texts/s**. At ~12k texts/s, embedding is not the ingest bottleneck (OpenSearch bulk at ~8.6k docs/s is).

---

## Final summary (run 2026-07-22, all 16 benchmarks ✅)

**Scalability** — OpenSearch bulk ingest **8.6k docs/s** (batch 1000); in-memory brute-force dense scales linearly (50k docs → p95 16.7 ms, 25.6 MB); fan-out amortizes to 3.5 ms/query at 16-wide. For large corpora use OpenSearch HNSW, not the in-memory backend.

**Throughput (APIs/sec)** — live FiQA (57k docs): keyword **558 qps**, dense **377 qps**, hybrid **84 qps**, regex **8.4 qps** single-thread; concurrent dense peaks at **~970 qps @ 4 workers**.

**AI-agent** — **SAC is ~2× faster (6.2 s vs 12.2 s) and ~28% cheaper ($0.00135 vs $0.00188/query) than MCP tool-calling**, with ~55% fewer LLM round-trips (2.4 vs 5.4) and a **51% prompt-cache hit** (vs 4.5%). SAC led Recall@10 (0.50) on the 8-query sample; run `-n 100` for stable quality.

**Reliability/primitives** — resilience wrappers are free (`with_retry` 0.14 µs; `chunked` 45M items/s); embedding on the RTX 5090 hits **12.3k texts/s**; only `mmr` (161 ops/s) and `semantic_dedup` (73 ops/s) are costly primitives — apply them to trimmed pools.

**Caveats:** agent metrics are from N=8 (directional; rerun `-n 100` for publishable quality). All raw JSON in `benchmarks/results/`.

## Heartbeat log
3-minute progress heartbeats are appended to `benchmarks/HEARTBEAT.md` and posted
in-session while benchmarks run.
