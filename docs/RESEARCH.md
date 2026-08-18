# Search-as-code research base — 150 sources

A survey of how search companies, vector-database companies, agentic-LLM
platforms, agent frameworks, and the research literature frame **agentic
retrieval / "search as code"** and what retrieval **primitives** they expose.
Compiled to ground the primitive set in `search_as_code/primitives.py` and
`session.py`. Grouped by source type; each entry is a primary source with the
primitive(s) it contributes.

The **[Primitive taxonomy](#primitive-taxonomy)** at the bottom maps every
recurring primitive to its status in this SDK.

---

## 1. Search API & retrieval companies

- **Rethinking Search as Code Generation** — Perplexity — https://research.perplexity.ai/articles/rethinking-search-as-code-generation — Atomic primitives (retrieval, fanout, ranking, filtering, dedup, rendering) exposed as an Agentic Search SDK the model programs via generated Python.
- **How We're Building the Next Generation of Search** — Exa — https://exa.ai/blog/how-to-build-nextgen-search — Embeddings-first neural retrieval as nearest-neighbor over meaning.
- **Introducing Exa 2.0** — Exa — https://exa.ai/blog/exa-api-2-0 — Custom embedding+reranking models; a "Deep" endpoint that agentically searches, processes, and re-searches until quality is met.
- **Exa Search API Guide** — Exa — https://exa.ai/docs/reference/search-api-guide — Neural/keyword/auto modes, find-similar-links, highlights, clean content retrieval.
- **Tavily 101: AI-powered Search for Developers** — Tavily — https://www.tavily.com/blog/tavily-101-ai-powered-search-for-developers — Composable primitives: /search, /extract (URL→clean text), /crawl.
- **Connect your AI agents to the web** — Tavily — https://www.tavily.com/product — search / extract / research / crawl / map + iterative multi-search research with dedup, behind a prompt-injection firewall.
- **Introducing Rerank (Search Goggles)** — Brave — https://brave.com/blog/search-rerank/ — Query-time user-defined reranking as a primitive, not a post-process.
- **Brave Search API** — Brave — https://brave.com/search/api/ — Independent 30B+ page index; web/LLM-context/Answers/rerank endpoints.
- **Search API for the Agentic Era** — You.com — https://you.com/resources/search-api-for-the-agentic-era — Latency-typed endpoints: sub-445ms fact-check search vs a depth-tuned research endpoint.
- **Introducing ARI** — You.com — https://you.com/resources/introducing-ari-the-first-professional-grade-research-agent-for-business — Deep-research agent that decomposes a query into steps and reasons across hundreds of sources.
- **FastGPT API** — Kagi — https://help.kagi.com/kagi/api/fastgpt.html — Programmable grounded-answer endpoint with reference citations.
- **Search Quality** — Kagi — https://help.kagi.com/kagi/search-details/search-quality.html — Fan each query to a dozen+ sources in parallel, re-rank with proprietary quality signals.
- **/research vs /search** — Linkup — https://www.linkup.so/blog/what-is-the-research-endpoint-and-when-should-you-use-it — Single-pass /search vs multi-hop /research with confidence-based routing/escalation.
- **Best web search API in 2026** — Linkup — https://www.linkup.so/blog/best-web-search-api-in-2026-top-providers-compared — Cleaned content, structured sourcedAnswer objects, parallel concurrent queries.
- **Serper — Fastest Google Search API** — Serper — https://serper.dev/ — Google SERP as structured JSON across many verticals; raw retrieval agents rerank downstream.
- **AI-Powered SEO Research Agent** — SerpAPI — https://serpapi.com/blog/ai-powered-seo-research-agent-with-openai-serpapi/ — plan → execute (parallel) → synthesize a cited report.
- **SerpAPI Search API** — SerpAPI — https://serpapi.com/search-api — Multi-engine API returning structured SERP components as typed retrieval primitives.
- **Introducing Algolia NeuralSearch** — Algolia — https://www.algolia.com/blog/product/introducing-algolia-neuralsearch — Keyword + vector semantic search in one API via neural hashing.
- **The agentically self-enriching index** — Algolia — https://www.algolia.com/blog/engineering/self-enriching-index — Background agent generates semantic tags/metadata — index enrichment as search-as-code.
- **Launching the Algolia MCP Server** — Algolia — https://www.algolia.com/blog/engineering/algolia-mcp-server — retrieve/add/configure index actions exposed to agents via MCP.
- **Introducing hybrid search** — Meilisearch — https://www.meilisearch.com/blog/introducing-hybrid-search — Hybrid with a semantic-ratio control blending keyword and semantic.
- **Why traditional hybrid search falls short** — Meilisearch — https://www.meilisearch.com/blog/fixing-hybrid-search — Normalizes 0–1 relevance scores from both sides rather than rank-based RRF.
- **Vector Search (Hybrid / Rank Fusion)** — Typesense — https://typesense.org/docs/30.2/api/vector-search.html — HNSW ANN, auto-embedding, hybrid via Rank Fusion with tunable `alpha`.
- **Why Use Typesense for Knowledge Retrieval** — Typesense — https://typesense.org/docs/guide/ai-agents-typesense.html — Hybrid, typo tolerance, server-side embeddings, multi-collection federated queries, filtering.
- **Introducing Vectara's Chain Rerankers** — Vectara — https://www.vectara.com/blog/introducing-vectaras-chain-rerankers — Pipeline multiple rerankers (hybrid, cross-attentional, **MMR diversity**, custom UDFs) sequentially.
- **Introducing Vectara-agentic** — Vectara — https://www.vectara.com/blog/introducing-vectara-agentic — Agents form query plans and call create_rag_tool() with hybrid+rerank+summarization.
- **Vectara Factual Consistency Score** — Vectara — https://www.vectara.com/blog/automating-hallucination-detection-introducing-vectara-factual-consistency-score — HHEM/FCS calibrated hallucination detection over grounded generation.
- **Knowledge graphs & context for enterprise AI** — Glean — https://www.glean.com/blog/knowledge-graph-agentic-engine — (subject, predicate, object) triplets with edge props (timestamps/ACLs), traversal for multi-hop.
- **Agentic reasoning: The future of Work AI** — Glean — https://www.glean.com/blog/agentic-reasoning-future-ai — Decompose questions into multi-step plans executed via tools with self-reflection.
- **Introducing RAG 2.0** — Contextual AI — https://contextual.ai/research/introducing-rag2 — End-to-end jointly optimized retrieval+generation (Contextual Language Models).
- **World's first instruction-following reranker** — Contextual AI — https://contextual.ai/blog/introducing-instruction-following-reranker — Reranker that follows NL instructions about recency/source/metadata to resolve conflicts.
- **The case for a new retrieval engine for agents** — Hornet — https://hornet.dev/blog/the-case-for-a-new-retrieval-engine-for-agents — Agents need exact case-sensitive matches (code) + semantic + structured-span queries via schema-first APIs.
- **This is what agentic retrieval looks like** — Hornet — https://hornet.dev/blog/this-is-what-agentic-retrieval-looks-like — Agents write long structured queries with operators (phrase, `site:`, `filetype:`, wildcards, dates, OR/negation).
- **Deep research is a retrieval problem** — Hornet — https://hornet.dev/blog/deep-research-is-a-retrieval-problem — Retrieval is the bottleneck: 93% on BrowseComp-Plus with right docs vs 14% when the model must find them.
- **Code Mode for Agentic Retrieval** — Hornet — https://hornet.dev/blog/code-mode-for-agentic-retrieval — Harness = sandbox + SDK; expose retrieval primitives, keep bulky state out of context, return structured evidence.

## 2. Vector database companies

- **Cascading retrieval: dense + sparse + reranking** — Pinecone — https://www.pinecone.io/blog/cascading-retrieval/ — Unified dense→sparse→rerank pipeline (beyond parallel hybrid), +48% precision.
- **Cascading retrieval with multi-vector representations** — Pinecone — https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/ — ColBERT-style late-interaction layered into cascading retrieval.
- **Pinecone Nexus: Knowledge Engine for Agents** — Pinecone — https://www.pinecone.io/blog/knowledge-infrastructure-for-agents/ — Retrieval as agent knowledge infrastructure, not a single vector call.
- **Rerankers and Two-Stage Retrieval** — Pinecone — https://www.pinecone.io/learn/series/rag/rerankers/ — Retrieve broad, then rerank before the LLM.
- **Meet Weaviate Agents** — Weaviate — https://weaviate.io/blog/weaviate-agents — Agents orchestrate search/transform/query over the vector DB as programmable primitives.
- **Search Mode Benchmarking** — Weaviate — https://weaviate.io/blog/search-mode-benchmarking — Compound retrieval: query expansion + decomposition + schema introspection + reranking.
- **Hybrid Search Explained** — Weaviate — https://weaviate.io/blog/hybrid-search-explained — BM25F + vector fused via RRF.
- **What is Agentic RAG? (Qdrant)** — Qdrant — https://qdrant.tech/articles/agentic-rag/ — Agents orchestrate multi-step retrieval, deciding dynamically how to gather info.
- **Hybrid Search with Qdrant's Query API** — Qdrant — https://qdrant.tech/articles/hybrid-search/ — Prefetch-based multi-stage: dense + sparse + ColBERT late-interaction with RRF/rerank.
- **Qdrant 1.10 — Universal Query, IDF & ColBERT** — Qdrant — https://qdrant.tech/blog/qdrant-1.10.x/ — Universal Query API, native multi-vector MaxSim, built-in IDF sparse scoring.
- **Stop Building Vanilla RAG (DeepSearcher)** — Milvus/Zilliz — https://milvus.io/blog/stop-use-outdated-rag-deepsearcher-agentic-rag-approaches-changes-everything.md — Dynamic multi-step retrieval: decompose → iterate → self-correct.
- **A Review of Hybrid Search in Milvus** — Milvus/Zilliz — https://zilliz.com/blog/a-review-of-hybrid-search-in-milvus — Multi-vector columns; dense/sparse merged via RRF and weighted fusion.
- **Metadata Filtering, Hybrid Search or Agent** — Milvus/Zilliz — https://zilliz.com/blog/metadata-filtering-hybrid-search-or-agent-in-rag-applications — Contrasts filtering vs hybrid vs agentic routing.
- **Agentic RAG with Claude 3.5, LlamaIndex, Milvus** — Milvus/Zilliz — https://zilliz.com/blog/agentic-rag-using-claude-3.5-sonnet-llamaindex-and-milvus — LLM plans and issues tool-based queries against Milvus.
- **Chroma Context-1: Self-Editing Search Agent** — Chroma — https://www.trychroma.com/research/context-1 — Multi-turn loop over primitives: search_corpus (hybrid+RRF), **grep_corpus (regex)**, read_document + context pruning.
- **Launching regex search support** — Chroma — https://www.trychroma.com/changelog/regex — Regex / NotRegex pattern matching as a first-class retrieval primitive.
- **Full Text Search** — Chroma — https://docs.trychroma.com/docs/querying-collections/full-text-search — where_document with $contains/$not_contains, regex, logical composition.
- **Hybrid Search and Custom Reranking** — LanceDB — https://www.lancedb.com/blog/hybrid-search-and-custom-reranking-with-lancedb-4c10a6a3447e — Hybrid fused + pluggable rerankers (RRF, linear, Cohere, ColBERT).
- **Multivector Search** — LanceDB — https://docs.lancedb.com/search/multivector-search — Multiple embeddings per item for ColBERT/ColPali MaxSim.
- **Late Interaction & Multi-modal Retrievers** — LanceDB — https://www.lancedb.com/blog/late-interaction-efficient-multi-modal-retrievers-need-more-than-just-a-vector-index — Late-interaction multi-vector (ColPali) needs more than plain ANN.
- **Hybrid Search (docs)** — Turbopuffer — https://turbopuffer.com/docs/hybrid — BM25 + vector ANN with server-side RRF and optional cross-encoder rerank.
- **FTS v2: up to 20x faster full-text** — Turbopuffer — https://turbopuffer.com/blog/fts-v2 — BM25 with MAXSCORE pruning framed as a first-class agent primitive (agents fire long queries).
- **Training SID-1 to beat GPT-5 at search** — Turbopuffer — https://turbopuffer.com/blog/reinforcement-learning-sid-ai — High-concurrency agentic search: RL model drives many parallel fine-grained retrievals.
- **Eliminating the Precision–Latency Trade-Off** — Vespa — https://blog.vespa.ai/eliminating-the-precision-latency-trade-off-in-large-scale-rag/ — Multiphase/layered ranking (ANN+keyword → dense rerank → advanced), doc- and chunk-level.
- **Redefining Hybrid Search with Vespa** — Vespa — https://blog.vespa.ai/redefining-hybrid-search-possibilities-with-vespa/ — Sparse (BM25/SPLADE) + dense (HNSW/ColBERT) via tensor ranking expressions.
- **Improving Zero-Shot Ranking (part two)** — Vespa — https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa-part-two/ — ColBERT MaxSim fused with BM25 via RRF.
- **How Perplexity beat Google with Vespa** — Vespa — https://blog.vespa.ai/perplexity-show-what-great-rag-takes/ — Retrieval + filtering + ranking + ML inference in one engine for agentic search volume/latency.
- **From ts_rank to BM25: pg_textsearch** — Timescale/Tiger Data — https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres — pgvector + true-BM25 hybrid merged via RRF in one Postgres DB.
- **PostgreSQL as fast as Pinecone** — Timescale/Tiger Data — https://www.tigerdata.com/blog/how-we-made-postgresql-as-fast-as-pinecone-for-vector-data — StreamingDiskANN for accurate metadata-filtered vector search.
- **PostgreSQL Hybrid Search with pgvector + Cohere** — Timescale/Tiger Data — https://www.tigerdata.com/blog/postgresql-hybrid-search-using-pgvector-and-cohere — Vector+keyword in Postgres with a Cohere reranker.
- **MongoDB Atlas Native Hybrid Search** — MongoDB — https://www.mongodb.com/blog/post/product-release-announcements/boost-search-relevance-mongodb-atlas-native-hybrid-search — $rankFusion (RRF) and $scoreFusion aggregation stages.
- **MongoDB Vector Search Overview** — MongoDB — https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/ — ANN (HNSW), exact ENN, metadata pre-filtering.
- **Use Cases for Text, Vector, and Hybrid Search** — MongoDB — https://www.mongodb.com/blog/post/top-use-cases-for-text-vector-and-hybrid-search — When to use lexical vs semantic vs RRF hybrid.
- **Agentic Retrieval Techniques: A Complete Guide** — Redis — https://redis.io/blog/agentic-retrieval-techniques/ — Taxonomy: hybrid, cross-encoder rerank, metadata filter, LLM/semantic routing, query planning/rewrite/expand (HyDE), semantic caching.
- **Hybrid search in Redis 8.4 (FT.HYBRID)** — Redis — https://redis.io/blog/revamping-context-oriented-retrieval-with-hybrid-search-in-redis-84/ — Single command fusing full-text, vector, geo, metadata via RRF/Linear Combination.
- **Hybrid search explained** — Redis — https://redis.io/blog/hybrid-search-explained/ — Parallel BM25 + dense merged with RRF (1/(rank+60)) plus metadata/geo filters.
- **Context engineering & hybrid search for agentic AI** — Elastic — https://www.elastic.co/search-labs/blog/context-engineering-hybrid-search-evolution-agentic-ai — Hybrid+RRF into "context engineering" as intent-driven programmable queries.
- **Introducing the Elasticsearch Relevance Engine (ESRE)** — Elastic — https://www.elastic.co/search-labs/blog/introducing-elasticsearch-relevance-engine-esre — Vector + ELSER learned-sparse + BM25f + RRF in one toolkit.
- **Introducing the Elastic Rerank model** — Elastic — https://www.elastic.co/search-labs/blog/elastic-rerank-model-introduction — DeBERTa-v3 cross-encoder reranker over ELSER + RRF.
- **Semantic reranking with retrievers** — Elastic — https://www.elastic.co/search-labs/blog/semantic-reranking-with-retrievers — Composable retriever building blocks (kNN, BM25, RRF, semantic rerank) in one _search call.

## 3. Agentic LLM platforms

- **Code execution with MCP** — Anthropic — https://www.anthropic.com/engineering/code-execution-with-mcp — MCP servers as a code API; load only needed tools, filter in-sandbox (~98% token cut).
- **Contextual Retrieval in AI Systems** — Anthropic — https://www.anthropic.com/engineering/contextual-retrieval — Contextual Embeddings + Contextual BM25 + rerank; −67% failed retrievals.
- **Code Mode: the better way to use MCP** — Cloudflare — https://blog.cloudflare.com/code-mode/ — MCP tools → TypeScript API the LLM writes code against (composition, control flow) in a Workers isolate.
- **Code Mode: an entire API in 1,000 tokens** — Cloudflare — https://blog.cloudflare.com/code-mode-mcp/ — One code-execution tool to plan/call/process compactly.
- **AI Search: the search primitive for your agents** — Cloudflare — https://blog.cloudflare.com/ai-search-agent-primitive/ — Hybrid (vector+BM25 fused), metadata relevance boosting, cross-instance search as an agent primitive.
- **Introducing agentic retrieval in Azure AI Search** — Microsoft/Azure — https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-agentic-retrieval-in-azure-ai-search/4414677 — Multiturn query engine that plans and runs its own retrieval strategy.
- **Up to 40% better relevance (agentic retrieval engine)** — Microsoft/Azure — https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/up-to-40-better-relevance-for-complex-queries-with-new-agentic-retrieval-engine/4413832 — LLM-built query plan: subquery decomposition + conversation history.
- **Agentic retrieval overview (docs)** — Microsoft/Azure — https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview — Query planning, parallel subquery execution across sources, answer synthesis with citations.
- **Introducing Embed 4** — Cohere — https://cohere.com/blog/embed-4 — Multimodal, 128K context, Matryoshka dims, unified text+image embeddings.
- **Rerank** — Cohere — https://cohere.com/rerank — Cross-encoder reranker reordering candidates from initial retrieval.
- **File search (Responses API)** — OpenAI — https://developers.openai.com/api/docs/guides/tools-file-search — Auto chunk/embed into vector stores; vector + keyword retrieval with metadata filter.
- **Deep research** — OpenAI — https://developers.openai.com/api/docs/guides/deep-research — Browse/search/synthesize hundreds of sources via web search, MCP, file search.
- **Function calling** — OpenAI — https://developers.openai.com/api/docs/guides/function-calling — Predefined tool invocation — the baseline code-mode contrasts with.
- **Code Interpreter** — OpenAI — https://developers.openai.com/api/docs/guides/tools-code-interpreter — Model writes/runs Python in a sandbox — dynamic tool generation.
- **RAG and grounding on Vertex AI** — Google — https://cloud.google.com/blog/products/ai-machine-learning/rag-and-grounding-on-vertex-ai — Vertex AI Search retriever API + grounding of Gemini.
- **Vertex AI RAG Engine** — Google — https://developers.googleblog.com/en/vertex-ai-rag-engine-a-developers-tool/ — Managed retrieval + chunk/embed + context injection orchestration.
- **Grounding with Google Search** — Google — https://ai.google.dev/gemini-api/docs/google-search — google_search tool with dynamic retrieval (score/threshold) and cited answers.
- **Bedrock Knowledge Bases hybrid search** — AWS — https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-knowledge-bases-now-supports-hybrid-search/ — Semantic + full-text in parallel, fused.
- **Improve RAG with Cohere Rerank** — AWS — https://aws.amazon.com/blogs/machine-learning/improve-rag-performance-using-cohere-rerank/ — Query-time reranking layer over Bedrock retrieval.
- **Use a reranker model in Amazon Bedrock** — AWS — https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html — Reranker in Retrieve/RetrieveAndGenerate.
- **Reranking in Databricks AI Search** — Databricks — https://www.databricks.com/blog/reranking-mosaic-ai-vector-search-faster-smarter-retrieval-rag-agents — Single-parameter reranking, +~15 points accuracy.
- **Mosaic AI Agent Framework** — Databricks — https://www.databricks.com/blog/announcing-mosaic-ai-agent-framework-and-agent-evaluation — Governed data → Vector Search retrieval + serving + tracing for agents.
- **Late Chunking in Long-Context Embeddings** — Jina AI — https://jina.ai/news/late-chunking-in-long-context-embedding-models/ — Embed full doc first, then chunk, for context-aware chunk embeddings.
- **jina-embeddings-v3** — Jina AI — https://jina.ai/models/jina-embeddings-v3/ — Multilingual 8192-token embeddings with task LoRA adapters + late chunking.
- **Rerankers – Introduction** — Voyage AI — https://docs.voyageai.com/docs/reranker — Cross-encoder rerankers scoring query–doc pairs to refine retrieval.
- **Introduction (Embeddings)** — Voyage AI — https://docs.voyageai.com/docs/introduction — Domain/multimodal embeddings (voyage-code/law/finance).

## 4. Agent frameworks & orchestration

- **SelfQueryRetriever** — LangChain — https://reference.langchain.com/python/langchain-classic/retrievers/self_query/base/SelfQueryRetriever — LLM builds a structured query = semantic string + metadata filters (self-querying).
- **MultiQueryRetriever** — LangChain — https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever — LLM generates query variations, retrieve each, dedup union.
- **EnsembleRetriever** — LangChain — https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever — Combine retrievers (BM25 + dense) via weighted RRF.
- **Contextual compression** — LangChain — https://python.langchain.com/v0.2/docs/how_to/contextual_compression/ — Compress/drop retrieved docs by query relevance post-retrieval.
- **Agentic RAG with LangGraph** — LangChain/LangGraph — https://docs.langchain.com/oss/python/langgraph/agentic-rag — Retrieval-decision, document grading, query rewrite, then generate.
- **Query Transformations** — LlamaIndex — https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/ — HyDE + multi-step decomposition.
- **Query Transform Cookbook** — LlamaIndex — https://developers.llamaindex.ai/python/examples/query_transformations/query_transform_cookbook/ — Routing, rewriting, sub-question decomposition recipes.
- **Routers** — LlamaIndex — https://developers.llamaindex.ai/python/framework/module_guides/querying/router/ — Selector picks among query engines/retrievers (single/multi-selector).
- **Auto-Retrieval from a Vector Database** — LlamaIndex — https://developers.llamaindex.ai/python/framework/integrations/vector_stores/chroma_auto_retriever/ — Infer semantic query + metadata filters from NL.
- **Using LLMs for Retrieval and Reranking** — LlamaIndex — https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6 — Embedding retrieval → LLM reranking node postprocessor.
- **Agentic RAG with LlamaIndex** — LlamaIndex — https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6 — Agent loop over per-doc tools with routing + reranking postprocessors.
- **Retrievers** — Haystack/deepset — https://docs.haystack.deepset.ai/docs/retrievers — Sparse (BM25), dense, hybrid retriever variants.
- **QueryExpander** — Haystack/deepset — https://docs.haystack.deepset.ai/docs/queryexpander — LLM generates query variations to broaden recall.
- **Query Decomposition and Reasoning** — Haystack/deepset — https://haystack.deepset.ai/cookbook/query_decomposition — Decompose → answer sub-questions → reason over combined results.
- **Rankers / Choosing the Right Ranker** — Haystack/deepset — https://docs.haystack.deepset.ai/docs/rankers — Cross-encoder/API rankers reorder retrieved docs.
- **RAG tutorial (dspy.Retrieve)** — DSPy — https://dspy.ai/tutorials/rag/ — Retriever as a declarative module in a composed RAG program.
- **Multi-Hop RAG** — DSPy — https://dspy.ai/tutorials/multihop_search/ — Iteratively generate search queries from accumulated notes.
- **RAG Tool (RagTool)** — CrewAI — https://docs.crewai.com/en/tools/ai-ml/ragtool — Dynamic KB tool: agents query sources with configurable embedder + vector DB.
- **RAG with AutoGen / RetrieveChat** — AutoGen (Microsoft) — https://microsoft.github.io/autogen/0.2/docs/topics/retrieval_augmentation/ — RetrieveUserProxyAgent with customizable embed/split + vector DB.
- **Archival memory** — Letta (MemGPT) — https://docs.letta.com/guides/agents/archival-memory/ — Semantically searchable long-term store via archival_memory_search/insert.
- **Text Search with Vector Stores** — Semantic Kernel — https://learn.microsoft.com/en-us/semantic-kernel/concepts/text-search/text-search-vector-stores — Vector-store collection wrapped as a search plugin/tool.
- **Adding RAG to Semantic Kernel Agents** — Semantic Kernel — https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-rag — TextSearchProvider retrieves + injects context each turn.
- **RAG and the Search method** — Dust — https://docs.dust.tt/docs/understanding-retrieval-augmented-generation-rag-and-the-search-method-in-dust — Semantic search over selected data sources.
- **Search data sources** — Dust — https://docs.dust.tt/docs/search-your-data — Semantic search + filesystem-level browse (list folders, read by path).
- **Evaluating and improving search** — Sierra — https://sierra.ai/blog/evaluating-and-improving-search — Golden datasets + purpose-built retrieval/reranking models.
- **SWE-grep: RL for Fast Context Retrieval** — Cognition (Devin) — https://cognition.com/blog/swe-grep — RL model does highly parallel, few-serial-turn codebase context retrieval.

## 5. Retrieval techniques & agentic-search research

- **Reciprocal Rank Fusion (RRF)** — Cormack, Clarke & Büttcher, SIGIR 2009 — https://www.semanticscholar.org/paper/9e698010f9d8fa374e7f49f776af301dd200c548 — Fuse ranked lists by summing 1/(k+rank).
- **RAG-Fusion** — Rackauckas, arXiv 2024 — https://arxiv.org/abs/2402.03367 — Multiple query variants + per-query retrieval + RRF.
- **HyDE (Precise Zero-Shot Dense Retrieval)** — Gao et al., ACL 2023 — https://arxiv.org/abs/2212.10496 — Generate a hypothetical answer doc, embed, and search with it.
- **HyQE: Hypothetical Query Embeddings** — Zhou et al., EMNLP 2024 — https://arxiv.org/abs/2410.15262 — Rerank contexts via generated hypothetical queries.
- **HyPE: Hypothetical Prompt Embeddings** — NirDiamant, RAG_Techniques — https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyPE_Hypothetical_Prompt_Embeddings.ipynb — Precompute hypothetical questions per chunk at index time.
- **Query Rewriting for RAG (Rewrite-Retrieve-Read)** — Ma et al., EMNLP 2023 — https://arxiv.org/abs/2305.14283 — Trainable rewriter reformulates the query before retrieval.
- **Step-Back Prompting** — Zheng et al. (DeepMind), ICLR 2024 — https://arxiv.org/abs/2310.06117 — Derive a higher-level abstraction question to guide retrieval.
- **Least-to-Most Prompting** — Zhou et al. (Google), ICLR 2023 — https://arxiv.org/abs/2205.10625 — Decompose into ordered sub-questions.
- **Maximal Marginal Relevance (MMR)** — Carbonell & Goldstein, SIGIR 1998 — https://dl.acm.org/doi/10.1145/290941.291025 — Greedy rerank balancing relevance vs redundancy.
- **Passage Re-ranking with BERT (monoBERT)** — Nogueira & Cho, 2019 — https://arxiv.org/abs/1901.04085 — Cross-encoder relevance scoring of query+passage.
- **ColBERT: Late Interaction** — Khattab & Zaharia, SIGIR 2020 — https://arxiv.org/abs/2004.12832 — Per-token embeddings + MaxSim.
- **ColBERTv2** — Santhanam et al., NAACL 2022 — https://arxiv.org/abs/2112.01488 — Residual compression + denoised supervision for late interaction.
- **Fusion-in-Decoder** — Izacard & Grave, EACL 2021 — https://arxiv.org/abs/2007.01282 — Encode passages independently, fuse in the decoder.
- **Self-RAG** — Asai et al., ICLR 2024 — https://arxiv.org/abs/2310.11511 — Reflection tokens for on-demand retrieval + self-critique.
- **Corrective RAG (CRAG)** — Yan et al., 2024 — https://arxiv.org/abs/2401.15884 — Retrieval evaluator → correct/ambiguous/incorrect actions + web fallback.
- **GraphRAG** — Edge et al. (Microsoft), 2024 — https://arxiv.org/abs/2404.16130 — Entity graph + community summaries for global map-reduce retrieval.
- **Contextual Retrieval** — Anthropic — https://www.anthropic.com/engineering/contextual-retrieval — Prepend LLM-generated chunk context before indexing.
- **ReAct** — Yao et al., ICLR 2023 — https://arxiv.org/abs/2210.03629 — Interleave reasoning traces with tool/search actions.
- **CodeAct (Executable Code Actions)** — Wang et al., ICML 2024 — https://arxiv.org/abs/2402.01030 — Executable Python as the unified agent action space.
- **Agentic RAG: A Survey** — Singh et al., 2025 — https://arxiv.org/abs/2501.09136 — Taxonomy of routing, planning, reflection, tool use, multi-agent.
- **Deep Research: A Survey of Autonomous Research Agents** — Zhang et al., 2025 — https://arxiv.org/abs/2508.12752 — Planning → question developing → web exploration → synthesis.
- **Deep Research Agents: Examination & Roadmap** — Huang et al., 2025 — https://arxiv.org/abs/2506.18096 — Dynamic reasoning, long-horizon planning, multi-hop retrieval.
- **BrowseComp** — OpenAI, 2025 — https://openai.com/index/browsecomp/ — Benchmark of hard-to-find, easy-to-verify web questions (multi-hop retrieval).
- **BrowseComp-Plus** — Chen, Ma et al., 2025 — https://arxiv.org/abs/2508.06600 — Fixed curated corpus + hard negatives to isolate retriever vs agent.
- **Router Query Engine** — LlamaIndex — https://developers.llamaindex.ai/typescript/framework/modules/rag/query_engines/router_query_engine/ — Selector routes to a query engine/tool per query.
- **Sentence-Window / Parent-Document retrieval** — LlamaIndex — https://developers.llamaindex.ai/python/framework-api-reference/packs/sentence_window_retriever/ — Retrieve small chunks, return the surrounding window/parent.

---

## Primitive taxonomy

Mapping the recurring primitives to their status in this SDK. "✅ core" = portable,
model-free implementation shipped; "✅ pluggable" = shipped, needs a caller-supplied
LLM/model callable; "adapter" = provided by whichever backend supports it (emulated
otherwise); "roadmap" = identified, not yet built.

| Primitive | Recurrence across sources | Status in SDK |
|---|---|---|
| **dense / vector search** | universal | ✅ `Session.search(mode="dense")` |
| **keyword / BM25 / full-text** | universal | ✅ `mode="keyword"` (emulated if backend lacks it) |
| **hybrid search** | universal | ✅ `mode="hybrid"` (emulated via dense+keyword RRF) |
| **RRF fusion** | universal | ✅ core `fuse()` |
| **score / weighted fusion** | Meilisearch, MongoDB, Redis | ✅ `fuse(weights=...)` |
| **rerank (cross-encoder)** | universal | ✅ `rerank()` (pluggable reranker; lexical fallback) |
| **rerank (late-interaction / ColBERT MaxSim)** | Pinecone, Qdrant, Vespa, LanceDB, Milvus | roadmap (multi-vector capability flag added) |
| **metadata filtering** | universal | ✅ portable filter dialect (`filters.py`) |
| **dedup** | Perplexity, Tavily, LangChain | ✅ core `dedup()` / `ResultSet.dedup()` |
| **MMR / diversify** | Vectara, MMR paper, Pinecone | ✅ core `mmr()` / `Session.mmr()` |
| **fan-out / parallel multi-query** | Perplexity, Kagi, Turbopuffer, Cognition | ✅ core `fan_out()` / `search_many()` |
| **query expansion / multi-query** | RAG-Fusion, LangChain, Haystack, Weaviate | ✅ pluggable `expand()` / `expand_search()` |
| **query decomposition / sub-questions** | Azure, Glean, LlamaIndex, least-to-most | ✅ pluggable `decompose()` / `decompose_search()` |
| **HyDE (hypothetical doc)** | HyDE paper, Redis, LlamaIndex | ✅ pluggable `hyde_search()` |
| **query routing / selection** | LlamaIndex, Milvus, Glean | ✅ `route()` (multi-store fan-out + fuse) |
| **contextual compression** | LangChain, Perplexity (rendering) | ✅ `Session.compress()` (model-free, embedder-scored) |
| **regex / exact / operator search** | Hornet, Chroma, Turbopuffer | ✅ `mode="regex"` + `query_regex` (emulated if unsupported) |
| **freshness / recency** | Hornet, Brave, Contextual AI | ✅ core `freshness()` |
| **structured extraction / verification** | Perplexity, OpenAI, Vectara | ✅ pluggable `extract()` |
| **out-of-context state / read-document** | Anthropic, Hornet, Perplexity, Letta | ✅ `remember/recall`, `hydrate()`, sandbox |
| **code-as-action / sandboxed execution** | Anthropic, Cloudflare, CodeAct | ✅ `sandbox.py` (`LocalExecutor`) |
| **self-query (LLM metadata filters)** | LangChain, LlamaIndex auto-retrieval | roadmap (pluggable, feeds `filter=`) |
| **grade / reflect / correct** | Self-RAG, CRAG, LangGraph | roadmap (pluggable predicate over ResultSet) |
| **small-to-big / window / parent-document** | LlamaIndex | roadmap (needs parent map) |
| **contextualize-chunk (at index time)** | Anthropic, Jina late-chunking | roadmap (ingestion-side) |
| **knowledge-graph / multi-hop traversal** | Glean, GraphRAG | roadmap |
| **grounded answer + citations / hallucination check** | Kagi, Brave, Vectara, Google | roadmap (generation-side, out of retrieval core) |
| **semantic caching** | Redis | roadmap |

### What the research changed in the SDK
Added this pass, directly from the survey: **`mmr`/diversify**, **`route`** (multi-store
federated search + fuse), **`expand`** and **`decompose`** and **`hyde_search`** (pluggable
query-side primitives), **`compress`** (model-free contextual compression), and
**`mode="regex"`** exact/operator search with a `regex` capability flag — the last one
especially for the code-search use case Hornet and Chroma emphasize.
