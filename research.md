# research.md — running research log

Where our ideas come from. **Newest first.** One line per source: the takeaway + how it maps to
our code. (`docs/RESEARCH.md` holds the large curated base; this file is the day-to-day running log.)

## Agent instruction files & skills (Anthropic) — informs `soul.md` (2026-07)
- **Equipping agents with Agent Skills** (Anthropic) — skills = composable folders of instructions/
  scripts, discovered/loaded dynamically; **evaluation-driven** (find the gap, then build).
  https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- **CLAUDE.md best practices** — always-loaded source of truth for commands/conventions/rules; keep
  **< ~200 lines** (models follow ~150–200 instructions before context rot); **progressive disclosure**
  (reference detail files by path, never paste). → shaped `soul.md`.
  https://maketocreate.com/claude-md-best-practices-the-complete-2026-guide/ ,
  https://medium.com/@bijit211987/the-complete-guide-to-claude-md-memory-rules-loading-and-cross-tool-compression-97cc12ed037b
- **Memory tool (Claude platform)** — just-in-time context: agents record learnings, read back on
  demand. → our `learnings_standard.md` / `research.md` / memory files.
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool

## Corpus / schema introspection primitives (2026-07)
Goal: let the agent discover *what kind of data* sits in the store (fields, types, tables vs prose
vs facts, value ranges) and feed that shape to the LLM **before** it writes retrieval code.

- **Google Agent Search — auto-detect schema by sampling** the first imported docs, then propose a
  schema to review/edit → motivates `sample_docs()` + `describe_schema()`.
  https://docs.cloud.google.com/generative-ai-app-builder/docs/provide-schema
- **RushDB Schema API** — "describes labels, properties, value samples/ranges, relationships, index
  metadata… the first call tells the agent what data exists and what queries are safe to construct;
  agents read the data shape **before** calling search/aggregation/traversal." → the *introspect-first*
  pattern; our `describe_schema()`/`data_profile()`. https://rushdb.com/features/schema-api
- **AI Schema Enrichment + MCP tool (thatjeffsmith)** — schema tools give object listings, comments,
  indexes without the LLM generating queries → introspection as a first-class tool.
  https://www.thatjeffsmith.com/archive/2025/11/ai-schema-enrichment-a-new-mcp-tool-simpler-prompts/
- **Designing Tool Schemas for AI Agents (Callsphere)** — the tool *description* is what the LLM reads
  to decide to call it (what/when/when-not) → our primitive docstrings must state when to use each.
  https://callsphere.ai/blog/designing-tool-schemas-ai-agents-json-schema-best-practices
- **Rethinking Agentic RAG: LLM-Driven Logical Retrieval Beyond Embeddings** (arXiv 2605.27123) —
  schema selection minimizes resources; structured/logical retrieval beyond pure embeddings.
  https://arxiv.org/html/2605.27123v1
- **On the Structural Memory of LLM Agents** (arXiv 2412.15266) — structured memory formats help agents
  reason over retrieved content. https://arxiv.org/pdf/2412.15266
- **Schema Retrieval with Embeddings + LLM SQL** (MDPI) — embed/retrieve the *schema*, not just data,
  for query generation. https://www.mdpi.com/2076-3417/16/2/586

## RAG answer-gen / agentic search (2026-07)
- **BEIR** (arXiv 2104.08663) — best dense model beat BM25 on only 8/18 datasets zero-shot → route per
  query type. https://arxiv.org/abs/2104.08663
- **Think Before You Retrieve: Test-Time Adaptive Search with Small LMs** (arXiv 2511.07581) — adaptive,
  per-query retrieval → our `adaptive_search` / router.
- **Claim-Aware Scientific RAG: Evidence-First Retrieval and Abstention** — verify + abstain → our
  `confidence`/`abstain`, "MCP-verify against KG".
- **MCP** (Anthropic, Nov 2024) — tools/resources/prompts JSON-RPC protocol → our verify-against-KG alignment.
- **Agentic search > vector RAG** (Claude Code writeup) — simpler, fewer staleness/security/reliability
  issues → the search-as-code thesis.
