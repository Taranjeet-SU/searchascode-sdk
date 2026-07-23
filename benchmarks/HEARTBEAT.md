# Benchmark heartbeat log

3-minute progress heartbeats while the benchmark suite runs. Each entry: time,
what finished, live system stats, and what's next.

---

### ❤️ Heartbeat #1 — 03:17
**Phase:** benchmarks running (detached runner). **Elapsed:** ~1 min.
- **Done & logged:** A1–A4 (scalability), E2 (micro), E3 (resilience).
- **Running:** Section B (throughput / QPS on live FiQA).
- **Queued:** E4 (embedding), then C/D/E1 (agent, `phase1.benchmark -n 8`).
- **System:** GPU 0% util, 8.1/32.6 GB used · OpenSearch health 200 · runner alive.
- **Highlights so far:** OpenSearch ingest **8.6k docs/s** @ batch 1000; in-memory 50k-doc dense query **p95 16.7 ms**; `with_retry` overhead **0.14 µs/call**; `mmr`/`semantic_dedup` are the only expensive primitives (161 / 73 ops/s).
- **Next heartbeat:** ~03:20.

---

### ❤️ Heartbeat #2 (final) — 03:22 — ✅ ALL DONE
**Phase:** complete. Runner exited cleanly (`ALL DONE` at 03:20:37); no runner process left.
- **Finished since #1:** B1/B2 (throughput), E4 (embedding), C1/C2 + D1/D2 + E1 (agent, `phase1.benchmark -n 8`). All 16 benchmarks logged in `benchmark_changelog.md`.
- **System:** GPU 0% util, 8.1/32.6 GB · OpenSearch health 200 · runner not running.
- **Headline numbers:**
  - Throughput (FiQA 57k): keyword **558 qps**, dense **377**, hybrid **84**, regex **8.4**; concurrent dense peaks **~970 qps @ 4 workers**.
  - Embedding (RTX 5090): **12.3k texts/s** @ batch 256.
  - Agent: **SAC 6.2 s vs tool-calling 12.2 s** (~2× faster), **$0.00135 vs $0.00188/query** (~28% cheaper), **cache hit 51% vs 4.5%**, LLM calls 2.4 vs 5.4.
- **Loop:** stopping the 3-min heartbeat — everything is done.

---

### ❤️ Agent-run heartbeat (final) — 06:07 — ✅ n=100 DONE
Full `phase1.benchmark -n 100` completed (`AGENT FULL DONE` at 06:07:31); no process left. GPU idle, OpenSearch 200.
**Stable results (100 queries):**
- **Recall@10:** SAC **0.549** · base 0.479 · tool-calling 0.440
- **nDCG@10:** SAC **0.408** · tool-calling 0.399 · base 0.379
- **MRR@10:** tool-calling **0.475** · SAC 0.446 · base 0.415
- **Latency:** SAC **7.7 s** vs tool-calling 15.9 s (base 0.02 s)
- **Cost/100q:** SAC **$0.145** vs tool-calling $0.229 · **cache hit 53.5% vs 26.9%** · calls 2.57 vs 6.15
`benchmark_changelog.md` Sections C/D/E1 + final summary updated with these. Loop stopped.

