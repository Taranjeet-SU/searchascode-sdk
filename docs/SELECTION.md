# How the SDK is exposed to the LLM + when it should call/chain primitives

**Where the rules live:** the LLM never sees Python signatures alone — it gets
`phase1/sac_surface.py::SAC_SYSTEM`, a single **system prompt** (a stable, prompt-cached
prefix — see CACHING.md) that contains three things:

1. **The primitive surface** — every callable with a one-line "what it does".
2. **Decision rules** — *when* to call each primitive, keyed on query shape + the samples
   the code prints (e.g. "exact tokens → keyword/regex", "multi-part → decompose", "flat
   score curve → adaptive_search", "redundant → semantic_dedup/diversity_quota",
   "weak → abstain & reformulate").
3. **Chaining recipes** — the typical pipeline: reformulate → retrieve (1–2 modes) → fuse →
   [dedup/diversify] → [rerank] → confidence-check → evidence.

The model then writes a program that composes them; it doesn't emit tool-calls. As the
primitive set grows, the surface stays cacheable and uses **progressive disclosure** (a
curated core in-prompt; long tail loadable on demand) so the prompt doesn't bloat. Longer
term the decision rules become **learned** (query→primitive-combo router) and
**ontology/KG-grounded** — see PHASE2.md.

## Primitives added from source research (Vespa · hornet.dev · IR papers · Reddit)
Implemented (model-free):
- `normalize_scores`, `relative_score_fusion` — score-based fusion that keeps magnitude
  (Weaviate hybrid-fusion).
- `diversity_quota` — cap hits per source/topic (Vespa result diversity).
- `semantic_dedup` — near-duplicate collapse by embedding cosine (SemDeDup).
- `confidence` / `abstain` — retrieval-confidence gating from score + gap (R³AG).
- `score_cutoff` / `adaptive_search` — adaptive result-set sizing from the score curve.

Roadmap (need a model or backend feature):
- `colbert_maxsim` late-interaction, `splade_sparse` learned-sparse (Vespa/paper).
- `wand_pruning` / block-max WAND, `filter_strategy` pre/post/ACORN filtered-ANN,
  `binary_rescore` / `int8_rescore` quantized two-phase (Vespa/Elastic).
- `parent_child` / `sentence_window` / `auto_merge` small-to-big chunking (LangChain/LlamaIndex).
- `query_operators` / `phrase_proximity` operator & exact-phrase search (hornet.dev).
- `match_features` per-hit ranking signals, `field_boost` per-field weights (Vespa).

Sources: weaviate.io/blog/hybrid-search-fusion-algorithms · docs.vespa.ai/en/querying/result-diversity.html ·
blog.vespa.ai/announcing-colbert-embedder-in-vespa · hornet.dev/blog/this-is-what-agentic-retrieval-looks-like ·
arXiv:2107.05720 (SPLADE) · arXiv:2403.04871 (ACORN) · R³AG / SemDeDup.
