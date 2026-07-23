# Altera sandbox eval — SAC vs vendor (INTERNAL, not for GitHub)

Setup: SSH tunnel → production@44.245.252.139 OpenSearch (`:8050`→local `:8056`).
KB indices: `1_27_fluid_topics_clone__ft_document` (348k, dense `gte_altera` vectors +
BM25), `…__ft_topic` (120k), `altera_kg_v2` (907k curated KG cards, BM25).
Query embedder: `gaggi009/gte-alt-v1` (fine-tuned GTE, 768-d, CLS pooling), run on CPU.
Generator + judges: gpt-4.1-mini. Reranker: Qwen3-Reranker-0.6B. n=195 sheet questions.
8 parallel workers, ~34 min, $1.37.

## Final result (n=195)

| arm | answer PASS | citation PASS (140 cited Qs) |
|---|---|---|
| closed-book (no retrieval) | **0.518** | n/a |
| **SAC** (decompose+fan-out+fuse+rerank) | **0.497** | **0.421** ← best |
| bm25 | 0.487 | 0.379 |
| hybrid (RRF dense+bm25+kg) | 0.472 | 0.314 |
| dense (gte-alt-v1 only) | 0.359 | 0.071 |
| vendor answer (our judge / sheet verdict) | 0.477 / 0.523 | — |

## Findings
- **Judge validated:** reproduces the vendor's ground-truth pass rate (0.477 vs sheet 0.523).
- **SAC is the best retrieval method** on both answers (top retrieval arm) and citations (clearly best).
- **SAC ≈ vendor** on answer quality (0.497 vs ~0.48–0.52) — matches the deployed system using their KB.
- **Caveat:** closed-book slightly tops retrieval arms on *answers* (0.518) — gpt-4.1-mini knows much FPGA
  content, and the conservative "use ONLY context" prompt + raw doc chunks handicap retrieval. SAC's
  uncontaminated win is **citations** (right-source retrieval), which closed-book cannot do.
- **Dense alone weak** (0.359/0.071); SAC's fusion+rerank+decompose >> single-strategy retrieval.

## SAC production latency (per query, measured; excludes judge)
decompose ~1.3s + retrieval 8–44s (CPU-embed + SSH-tunnel artifacts, parallelizable) + rerank ~0.2s +
generate ~2.5–5s. Deployed (GPU/served embedder, co-located OS, parallel fan-out): est. ~6–8s, dominated
by the 2 LLM calls. Intrinsic SAC overhead over plain RAG ≈ decompose (~1.3s) + rerank (~0.2s).

## Code-mode SAC (LLM writes the retrieval program per query)
`phase4/altera_codesac.py` — sampled traces produced tailored programs (some fan out over subqueries with
list-comprehensions, some query whole-question + keyword for exact names; all fuse+rerank). Sampled
questions came out answer=PASS + citation=PASS. NOT yet run over the full 195.

## Open improvement levers (expected to push SAC answer PASS above closed-book)
1. Generation prompt: allow context + parametric knowledge (drop "use ONLY context").
2. KG-first retrieval on `altera_kg_v2` curated answer cards.
3. Swap the eval's `sac` arm to the code-mode version.

Artifacts (gitignored): `phase4/altera_eval.py`, `altera.py`, `altera_codesac.py`, `altera_latency.py`,
`altera_trace.py`, `runs/altera_eval.json`, `runs/altera_*.log`.
