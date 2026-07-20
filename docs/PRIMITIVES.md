# Search-as-code: canonical primitive taxonomy

A normalized **superset taxonomy** synthesized after screening well over 150 sources (see [RESEARCH.md](RESEARCH.md)) — research papers, product docs, and architecture articles from search and agentic companies. It is not a claim that the industry has one official vocabulary; synonymous operations are merged under canonical names. Rows feed the support matrix in the [README](../README.md).

## Atomicity model

- **Search-domain primitives** perform an operation specific to retrieval, content, ranking, evidence, or output.
- **Runtime/data combinators** are ordinary deterministic code operations (parallel map, joins, retries, persistence) supplied by the host runtime.
- **Contracts** are typed values passed between primitives.
- **Composite macros** are useful named pipelines, deliberately *not* treated as atomic primitives.

## Catalog counts

- Total catalog entries: **320**
- Data contracts: **10**
- Search-domain primitives: **223**
- Runtime/data combinators: **58**
- Evaluation primitives: **17**
- Composite macros: **12**

## Abstraction hierarchy

- **0. Data contracts**
  - Core types (10)
- **1. Source and corpus**
  - Source discovery (10)
- **2. Query processing**
  - Normalization (9)
  - Interpretation (14)
  - Transformation (15)
- **3. Search planning**
  - Planning and routing (11)
- **4. Candidate generation**
  - Lexical retrieval (12)
  - Vector and neural retrieval (8)
  - Structured, graph, and external retrieval (19)
- **5. Candidate manipulation**
  - Set and collection algebra (14)
  - Filtering and governance (18)
- **6. Content materialization**
  - Fetching and parsing (16)
  - Segmentation and representation (12)
  - Navigation and enrichment (11)
- **7. Scoring and ranking**
  - Scoring (20)
  - Ranking, fusion, and diversification (16)
- **8. Aggregation and analysis**
  - Aggregation and analytics (12)
- **9. Evidence and verification**
  - Evidence operations (17)
- **10. Context and output**
  - Selection, compression, and rendering (15)
- **11. Runtime**
  - Control-flow combinators (18)
- **12. State and observability**
  - State, provenance, and telemetry (14)
- **13. Evaluation and learning**
  - Evaluation and improvement (17)
- **14. Composite macros**
  - Non-atomic convenience pipelines (12)

## Full primitive catalog

| Abstraction class | Subclass | Primitive | Definition | Kind |
|---|---|---|---|---|
| 0. Data contracts | Core types | `TaskDirective` | The original user or parent-agent objective, including constraints, required outputs, and success criteria. | contract |
| 0. Data contracts | Core types | `Query` | A normalized request for retrieval, represented as text plus optional fields, filters, entities, locale, and intent. | contract |
| 0. Data contracts | Core types | `SourceSpec` | A typed description of a searchable source, including capabilities, schema, authority, permissions, and freshness. | contract |
| 0. Data contracts | Core types | `Candidate` | A lightweight retrieval hit containing an identifier, source, rank or score, and enough metadata for later processing. | contract |
| 0. Data contracts | Core types | `RankedList` | An ordered candidate collection with explicit ranking provenance and comparable or source-local scores. | contract |
| 0. Data contracts | Core types | `Document` | Materialized source content with stable identity, metadata, version, permissions, and provenance. | contract |
| 0. Data contracts | Core types | `Span` | A precisely addressable fragment of a document, such as a passage, sentence, table cell, code range, or image region. | contract |
| 0. Data contracts | Core types | `Evidence` | A claim-linked span plus its source, support relation, confidence, and verification state. | contract |
| 0. Data contracts | Core types | `ContextBundle` | A token-budgeted package of selected evidence prepared for a downstream model or consumer. | contract |
| 0. Data contracts | Core types | `SearchTrace` | The reproducible record of the plan, operations, parameters, intermediate states, timings, and source lineage. | contract |
| 1. Source and corpus | Source discovery | `list_sources` | Return the sources currently available to the search program. | search-domain |
| 1. Source and corpus | Source discovery | `describe_source` | Return a source's scope, authority, update behavior, latency, cost, and access model. | search-domain |
| 1. Source and corpus | Source discovery | `discover_capabilities` | Report which query, filtering, ranking, fetching, and aggregation operations a source supports. | search-domain |
| 1. Source and corpus | Source discovery | `inspect_schema` | Return searchable fields, data types, vector fields, relations, facets, and permission attributes. | search-domain |
| 1. Source and corpus | Source discovery | `select_source` | Choose one or more sources that satisfy the task's domain, authority, latency, and governance constraints. | search-domain |
| 1. Source and corpus | Source discovery | `open_index` | Bind the program to a named index, collection, graph, table, or corpus. | search-domain |
| 1. Source and corpus | Source discovery | `resolve_alias` | Resolve a logical corpus or index alias to its active physical target. | search-domain |
| 1. Source and corpus | Source discovery | `check_source_health` | Test source availability, indexing completeness, lag, and query readiness. | search-domain |
| 1. Source and corpus | Source discovery | `snapshot_source` | Bind retrieval to a stable source or index version for reproducibility. | search-domain |
| 1. Source and corpus | Source discovery | `federate_sources` | Create a logical searchable view over multiple heterogeneous sources without requiring a single physical index. | search-domain |
| 2. Query processing | Normalization | `normalize_unicode` | Canonicalize Unicode forms so visually or semantically equivalent characters compare consistently. | search-domain |
| 2. Query processing | Normalization | `normalize_case` | Apply source-appropriate case normalization while preserving case-sensitive identifiers where required. | search-domain |
| 2. Query processing | Normalization | `normalize_whitespace` | Collapse or standardize whitespace and invisible separators. | search-domain |
| 2. Query processing | Normalization | `normalize_punctuation` | Standardize punctuation, quotation marks, dashes, and delimiters without destroying query operators. | search-domain |
| 2. Query processing | Normalization | `transliterate` | Convert text between writing systems while preserving the intended terms. | search-domain |
| 2. Query processing | Normalization | `tokenize` | Split query text into searchable lexical units using language- and domain-aware rules. | search-domain |
| 2. Query processing | Normalization | `stem_or_lemmatize` | Reduce inflected terms to stems or lemmas for broader lexical matching. | search-domain |
| 2. Query processing | Normalization | `remove_or_mark_stopwords` | Remove, retain, or down-weight common terms according to the retrieval mode. | search-domain |
| 2. Query processing | Normalization | `normalize_dates_units_numbers` | Convert dates, quantities, currencies, versions, and numeric formats into canonical structured values. | search-domain |
| 2. Query processing | Interpretation | `detect_language` | Identify the query language or languages and attach locale information. | search-domain |
| 2. Query processing | Interpretation | `detect_domain` | Classify the subject or business domain needed for source and model routing. | search-domain |
| 2. Query processing | Interpretation | `classify_intent` | Identify the user's retrieval intent, such as lookup, comparison, troubleshooting, navigation, or research. | search-domain |
| 2. Query processing | Interpretation | `classify_query_type` | Determine whether the query is factual, exploratory, multi-hop, temporal, geospatial, entity-centric, or transactional. | search-domain |
| 2. Query processing | Interpretation | `estimate_query_complexity` | Estimate ambiguity, number of required hops, breadth, and expected evidence volume. | search-domain |
| 2. Query processing | Interpretation | `extract_keywords` | Identify high-information lexical terms, identifiers, and phrases. | search-domain |
| 2. Query processing | Interpretation | `extract_entities` | Detect people, organizations, products, places, concepts, and other named entities. | search-domain |
| 2. Query processing | Interpretation | `link_entities` | Resolve detected mentions to canonical entity identifiers in a catalog or knowledge graph. | search-domain |
| 2. Query processing | Interpretation | `extract_constraints` | Translate natural-language restrictions into typed filters, ranges, exclusions, and required fields. | search-domain |
| 2. Query processing | Interpretation | `parse_query_syntax` | Parse Boolean, phrase, fielded, wildcard, proximity, and precedence operators into a query tree. | search-domain |
| 2. Query processing | Interpretation | `parse_temporal_scope` | Resolve explicit and relative time expressions into intervals and freshness requirements. | search-domain |
| 2. Query processing | Interpretation | `parse_geospatial_scope` | Resolve locations into coordinates, regions, radii, routes, or bounding geometries. | search-domain |
| 2. Query processing | Interpretation | `resolve_coreference` | Resolve pronouns and references using conversation or task context. | search-domain |
| 2. Query processing | Interpretation | `contextualize_query` | Combine the current query with relevant conversation state while excluding unrelated history. | search-domain |
| 2. Query processing | Transformation | `make_standalone_query` | Rewrite a context-dependent query so it can be searched independently. | search-domain |
| 2. Query processing | Transformation | `spell_correct` | Correct likely spelling errors while preserving valid names, identifiers, and code tokens. | search-domain |
| 2. Query processing | Transformation | `generate_fuzzy_variants` | Generate controlled edit-distance or phonetic alternatives for uncertain terms. | search-domain |
| 2. Query processing | Transformation | `expand_synonyms` | Add equivalent or closely related terms from a synonym resource. | search-domain |
| 2. Query processing | Transformation | `expand_ontology` | Add broader, narrower, related, or mapped concepts from an ontology or taxonomy. | search-domain |
| 2. Query processing | Transformation | `expand_acronyms` | Expand abbreviations and optionally generate abbreviated alternatives. | search-domain |
| 2. Query processing | Transformation | `translate_query` | Translate the query for cross-language retrieval while retaining entities and constraints. | search-domain |
| 2. Query processing | Transformation | `rewrite_query` | Produce a retrieval-optimized formulation that preserves the original information need. | search-domain |
| 2. Query processing | Transformation | `simplify_query` | Remove distracting detail or syntactic complexity while retaining the required constraints. | search-domain |
| 2. Query processing | Transformation | `step_back_query` | Generate a more abstract query that retrieves general principles or background evidence. | search-domain |
| 2. Query processing | Transformation | `generate_hypothetical_document` | Create a hypothetical answer passage or document representation for embedding-based retrieval, commonly called HyDE. | search-domain |
| 2. Query processing | Transformation | `generate_multi_queries` | Produce diverse query variants intended to improve recall across different formulations. | search-domain |
| 2. Query processing | Transformation | `decompose_query` | Split a compound or multi-hop question into focused subqueries with explicit dependencies. | search-domain |
| 2. Query processing | Transformation | `expand_with_relevance_feedback` | Use judged or pseudo-relevant initial results to update the query representation or terms. | search-domain |
| 2. Query processing | Transformation | `generate_source_scoped_queries` | Instantiate source-, site-, field-, or format-specific query templates. | search-domain |
| 3. Search planning | Planning and routing | `plan_search` | Construct an executable retrieval plan from the directive, available sources, constraints, and budgets. | search-domain |
| 3. Search planning | Planning and routing | `plan_reasoning_hops` | Order dependent subqueries and specify which intermediate facts feed later retrieval. | search-domain |
| 3. Search planning | Planning and routing | `route_source` | Choose the source or source set most appropriate for the query. | search-domain |
| 3. Search planning | Planning and routing | `route_retriever` | Choose lexical, dense, sparse-neural, graph, structured, or multimodal retrieval for the query. | search-domain |
| 3. Search planning | Planning and routing | `choose_retrieval_mode` | Select exact, approximate, broad-recall, precision-first, exploratory, or hybrid behavior. | search-domain |
| 3. Search planning | Planning and routing | `choose_ranker` | Select the scoring, fusion, or reranking method appropriate for the candidate set and latency budget. | search-domain |
| 3. Search planning | Planning and routing | `set_search_budget` | Allocate limits for queries, candidates, fetches, tokens, time, and monetary cost. | search-domain |
| 3. Search planning | Planning and routing | `set_evidence_requirements` | Specify source authority, corroboration, freshness, citation, and confidence requirements. | search-domain |
| 3. Search planning | Planning and routing | `plan_parallelism` | Identify independent operations that can execute concurrently and define their fan-out. | search-domain |
| 3. Search planning | Planning and routing | `plan_fallbacks` | Define alternate sources or methods when a preferred operation fails or produces inadequate coverage. | search-domain |
| 3. Search planning | Planning and routing | `generate_clarification` | Produce a targeted clarification only when an unresolved ambiguity prevents a valid search plan. | search-domain |
| 4. Candidate generation | Lexical retrieval | `exact_term_search` | Retrieve items containing exact indexed terms or exact structured values. | search-domain |
| 4. Candidate generation | Lexical retrieval | `full_text_search` | Retrieve and rank text using an inverted index and a lexical relevance model such as BM25 or BM25F. | search-domain |
| 4. Candidate generation | Lexical retrieval | `phrase_search` | Retrieve items containing an ordered phrase, optionally with controlled slop. | search-domain |
| 4. Candidate generation | Lexical retrieval | `boolean_search` | Combine clauses using MUST, SHOULD, FILTER, and MUST_NOT semantics. | search-domain |
| 4. Candidate generation | Lexical retrieval | `fielded_search` | Restrict or weight matching to selected fields. | search-domain |
| 4. Candidate generation | Lexical retrieval | `prefix_search` | Match indexed terms beginning with a specified prefix. | search-domain |
| 4. Candidate generation | Lexical retrieval | `suffix_or_infix_search` | Match terms by suffix or internal substring where the index supports it. | search-domain |
| 4. Candidate generation | Lexical retrieval | `wildcard_search` | Match terms against wildcard patterns. | search-domain |
| 4. Candidate generation | Lexical retrieval | `regex_search` | Match indexed terms or materialized text against a regular expression. | search-domain |
| 4. Candidate generation | Lexical retrieval | `fuzzy_search` | Retrieve terms within a configured edit distance or similarity threshold. | search-domain |
| 4. Candidate generation | Lexical retrieval | `proximity_search` | Retrieve documents where terms occur within a specified positional distance or structural span. | search-domain |
| 4. Candidate generation | Lexical retrieval | `sparse_neural_search` | Retrieve with learned sparse term-weight vectors that retain inverted-index execution. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `exact_vector_search` | Compute exact nearest neighbors under a selected vector similarity or distance function. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `approximate_vector_search` | Retrieve approximate nearest neighbors using an ANN index such as HNSW, IVF, or a quantized variant. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `search_by_vector_id` | Use an already indexed item's vector as the query without re-embedding its content. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `multi_vector_search` | Score each item using multiple vectors, including late-interaction token or region representations. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `multi_target_vector_search` | Search several named vector spaces or fields and combine their distances. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `vector_range_search` | Return candidates whose similarity or distance lies within a specified interval. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `multimodal_vector_search` | Retrieve across text, image, audio, video, or other modalities in compatible embedding spaces. | search-domain |
| 4. Candidate generation | Vector and neural retrieval | `dense_sparse_dual_search` | Execute dense and learned-sparse retrieval as distinct candidate generators for later fusion. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `metadata_lookup` | Retrieve records by exact metadata keys or indexed attributes. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `scalar_query` | Retrieve records satisfying structured predicates over numeric, categorical, Boolean, or date fields. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `table_or_sql_query` | Execute a structured query over relational or analytical data and return rows or aggregates. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `geospatial_search` | Retrieve objects by point, radius, route, polygon, or geographic distance. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `temporal_search` | Retrieve events or versions using time intervals, valid-time, transaction-time, or recency semantics. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `graph_node_lookup` | Retrieve graph nodes or relationships by identifiers, properties, labels, or full-text/vector indexes. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `graph_traversal` | Expand from seed nodes through selected relationship types and depth constraints. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `path_search` | Retrieve paths satisfying connectivity, hop, cost, or relationship constraints. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `subgraph_search` | Return a query-conditioned connected subgraph rather than independent documents. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `community_search` | Retrieve graph communities or hierarchical summaries for global or thematic questions. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `recommendation_search` | Retrieve items similar to positive examples and dissimilar to negative examples. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `discovery_search` | Use positive and negative contextual examples as a one-shot preference boundary for exploration. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `browse_or_scroll` | Enumerate a filtered corpus without a relevance query, using cursors or iterators. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `random_sample` | Return a random or stratified sample for exploration, testing, or quality inspection. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `web_search` | Retrieve indexed public-web results for a textual query. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `site_or_domain_search` | Restrict web retrieval to specified sites, domains, URL patterns, or source classes. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `federated_search` | Execute searches against multiple indexes or remote systems and return source-tagged result sets. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `remote_retrieve` | Invoke a source-native retrieval API without copying its corpus into the local index. | search-domain |
| 4. Candidate generation | Structured, graph, and external retrieval | `multimedia_search` | Retrieve images, audio, or video using textual, example-based, or multimodal queries. | search-domain |
| 5. Candidate manipulation | Set and collection algebra | `concatenate` | Append candidate collections while preserving source and local-rank metadata. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `flatten` | Convert nested result collections into one candidate sequence. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `union` | Return candidates present in any input set. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `intersection` | Return candidates present in every specified input set. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `difference` | Return candidates in one set but not another. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `deduplicate_by_identity` | Merge hits that share a stable document, record, entity, or URL identity. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `deduplicate_by_content` | Merge exact or near-exact content duplicates using hashes or canonical text. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `semantic_deduplicate` | Merge candidates whose representations exceed a semantic-similarity threshold. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `join` | Combine candidates or records using a key, entity, relation, or temporal correspondence. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `group_candidates` | Partition candidates by a key such as document, source, entity, category, or thread. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `collapse_results` | Return one representative per group while retaining the ability to expand the group. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `expand_results` | Materialize hidden group members, child records, nested matches, or related records. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `sort_candidates` | Order candidates by a deterministic field or computed key. | runtime/data |
| 5. Candidate manipulation | Set and collection algebra | `slice_candidates` | Select a positional window using limit, offset, cursor, or rank boundaries. | runtime/data |
| 5. Candidate manipulation | Filtering and governance | `predicate_filter` | Retain candidates satisfying an arbitrary deterministic predicate. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `metadata_filter` | Restrict candidates using indexed metadata conditions. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `range_filter` | Restrict numeric, date, version, or similarity values to a range. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `source_filter` | Include or exclude specified sources or source classes. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `domain_filter` | Include or exclude hosts, repositories, business domains, or namespaces. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `language_filter` | Restrict candidates by detected or declared language. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `content_type_filter` | Restrict candidates by MIME type, document type, modality, or schema. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `freshness_filter` | Restrict candidates by publication, modification, effective, or crawl time. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `geospatial_filter` | Restrict candidates to a geographic boundary or distance. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `score_threshold_filter` | Discard candidates below a score, probability, confidence, or similarity threshold. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `top_k_filter` | Keep only the highest-ranked k candidates under the current ordering. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `permission_filter` | Remove content the requesting principal is not authorized to access; also called security trimming. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `tenant_filter` | Enforce tenant, workspace, project, or customer isolation. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `policy_filter` | Enforce legal, licensing, residency, retention, or organizational-use constraints. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `safety_filter` | Remove unsafe, disallowed, or inappropriate content according to the application policy. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `quality_or_spam_filter` | Remove low-quality, duplicated, manipulated, unavailable, or spam-like sources. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `authority_filter` | Require sources belonging to approved authority classes, such as official vendor or primary sources. | search-domain |
| 5. Candidate manipulation | Filtering and governance | `query_rule_filter` | Apply deterministic pin, exclude, redirect, boost, or business-rule actions triggered by query conditions. | search-domain |
| 6. Content materialization | Fetching and parsing | `fetch_document` | Materialize the complete current content for a candidate identifier or URL. | search-domain |
| 6. Content materialization | Fetching and parsing | `batch_fetch` | Fetch multiple documents in one operation with concurrency and error isolation. | search-domain |
| 6. Content materialization | Fetching and parsing | `follow_link` | Resolve and fetch an outgoing link, attachment, citation, or redirect target. | search-domain |
| 6. Content materialization | Fetching and parsing | `fetch_cached_or_archived_version` | Retrieve a cached, versioned, or archived representation rather than the live source. | search-domain |
| 6. Content materialization | Fetching and parsing | `parse_html` | Convert HTML into structured text, metadata, links, and layout elements. | search-domain |
| 6. Content materialization | Fetching and parsing | `parse_pdf` | Extract text and structural elements from a PDF while preserving page and coordinate provenance. | search-domain |
| 6. Content materialization | Fetching and parsing | `parse_office_document` | Extract structured content from word-processing, presentation, and spreadsheet formats. | search-domain |
| 6. Content materialization | Fetching and parsing | `parse_structured_data` | Parse JSON, XML, CSV, RDF, logs, or other machine-readable formats. | search-domain |
| 6. Content materialization | Fetching and parsing | `ocr` | Recognize text from scanned pages, screenshots, or images. | search-domain |
| 6. Content materialization | Fetching and parsing | `layout_parse` | Identify pages, headings, columns, blocks, tables, figures, lists, and reading order. | search-domain |
| 6. Content materialization | Fetching and parsing | `extract_main_content` | Remove navigation, boilerplate, advertisements, and unrelated chrome to retain primary content. | search-domain |
| 6. Content materialization | Fetching and parsing | `extract_metadata` | Extract title, authorship, dates, version, identifiers, permissions, and source-specific attributes. | search-domain |
| 6. Content materialization | Fetching and parsing | `extract_sections` | Split a document into titled or structurally coherent sections. | search-domain |
| 6. Content materialization | Fetching and parsing | `extract_tables` | Convert tabular regions into cells, rows, columns, and associated captions or footnotes. | search-domain |
| 6. Content materialization | Fetching and parsing | `extract_media_descriptions` | Produce searchable descriptions or transcripts for images, audio, and video. | search-domain |
| 6. Content materialization | Fetching and parsing | `parse_code_structure` | Build an AST or symbol structure for code-aware retrieval and navigation. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_fixed_window` | Split content by a fixed token or character window, optionally with overlap. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_sentence_or_paragraph` | Split content at linguistic sentence or paragraph boundaries. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_structure_aware` | Split content using headings, pages, lists, tables, code symbols, or document hierarchy. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_semantic` | Choose boundaries using topical or embedding shifts. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_parent_child` | Create small retrievable child chunks linked to larger context-bearing parents. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_propositions` | Convert content into independently retrievable factual propositions or atomic statements. | search-domain |
| 6. Content materialization | Segmentation and representation | `chunk_query_adaptive` | Select chunk granularity dynamically based on the query and document structure. | search-domain |
| 6. Content materialization | Segmentation and representation | `embed_dense` | Encode text or multimodal content as a dense vector. | search-domain |
| 6. Content materialization | Segmentation and representation | `embed_sparse` | Encode content as a sparse lexical or learned-sparse vector. | search-domain |
| 6. Content materialization | Segmentation and representation | `embed_multi_vector` | Encode an item as multiple vectors for token-, region-, field-, or aspect-level matching. | search-domain |
| 6. Content materialization | Segmentation and representation | `index_content` | Insert or update searchable lexical, vector, structured, or graph representations. | search-domain |
| 6. Content materialization | Segmentation and representation | `delete_or_tombstone_content` | Remove content from active retrieval while preserving required audit or deletion state. | search-domain |
| 6. Content materialization | Navigation and enrichment | `expand_parent` | Replace or augment a matched chunk with its parent document or larger enclosing section. | search-domain |
| 6. Content materialization | Navigation and enrichment | `expand_children` | Retrieve child chunks, nested records, or descendants of a matched object. | search-domain |
| 6. Content materialization | Navigation and enrichment | `expand_neighbor_window` | Retrieve preceding and following chunks around a match. | search-domain |
| 6. Content materialization | Navigation and enrichment | `follow_citations` | Retrieve sources referenced by a document or evidence span. | search-domain |
| 6. Content materialization | Navigation and enrichment | `retrieve_backlinks` | Retrieve documents that cite or link to the matched item. | search-domain |
| 6. Content materialization | Navigation and enrichment | `traverse_document_links` | Navigate explicit cross-document, entity, or repository links. | search-domain |
| 6. Content materialization | Navigation and enrichment | `retrieve_thread` | Materialize the surrounding email, case, chat, issue, or discussion thread. | search-domain |
| 6. Content materialization | Navigation and enrichment | `retrieve_version_history` | Retrieve previous, current, or superseding versions and their change metadata. | search-domain |
| 6. Content materialization | Navigation and enrichment | `canonicalize_document` | Resolve mirrors, aliases, redirects, and duplicate URLs to a canonical identity. | search-domain |
| 6. Content materialization | Navigation and enrichment | `enrich_metadata` | Add derived entities, topics, taxonomy labels, quality signals, or source classifications. | search-domain |
| 6. Content materialization | Navigation and enrichment | `translate_document` | Translate retrieved content while retaining span-level alignment to the original. | search-domain |
| 7. Scoring and ranking | Scoring | `score_lexical_relevance` | Compute term-based relevance such as TF-IDF, BM25, BM25F, or field-match scoring. | search-domain |
| 7. Scoring and ranking | Scoring | `score_dense_similarity` | Compute vector similarity or distance between query and candidate representations. | search-domain |
| 7. Scoring and ranking | Scoring | `score_sparse_neural_relevance` | Compute relevance from learned sparse term weights. | search-domain |
| 7. Scoring and ranking | Scoring | `score_late_interaction` | Aggregate fine-grained query-token to document-token or region interactions. | search-domain |
| 7. Scoring and ranking | Scoring | `score_cross_encoder` | Jointly encode the query and candidate to estimate relevance or answer utility. | search-domain |
| 7. Scoring and ranking | Scoring | `score_llm_relevance` | Use an LLM to grade relevance against explicit criteria. | search-domain |
| 7. Scoring and ranking | Scoring | `score_phrase_or_proximity` | Score phrase alignment, term order, positional closeness, or structural co-occurrence. | search-domain |
| 7. Scoring and ranking | Scoring | `score_freshness` | Score recency or temporal validity using a decay or task-specific freshness model. | search-domain |
| 7. Scoring and ranking | Scoring | `score_authority` | Score source trust, ownership, expertise, or primary-source status. | search-domain |
| 7. Scoring and ranking | Scoring | `score_content_quality` | Score completeness, readability, specificity, spam risk, or document quality. | search-domain |
| 7. Scoring and ranking | Scoring | `score_popularity` | Score usage, citations, links, engagement, or other collective signals. | search-domain |
| 7. Scoring and ranking | Scoring | `score_personalization` | Score relevance to the requesting user, role, history, or organizational context. | search-domain |
| 7. Scoring and ranking | Scoring | `score_geographic_relevance` | Score distance or geographic fit. | search-domain |
| 7. Scoring and ranking | Scoring | `score_graph_relevance` | Score graph distance, path quality, connectivity, centrality, or structural similarity. | search-domain |
| 7. Scoring and ranking | Scoring | `score_ontology_coverage` | Score alignment with required entities, concepts, relations, or taxonomy nodes. | search-domain |
| 7. Scoring and ranking | Scoring | `score_answerability` | Estimate whether a candidate contains sufficient evidence to answer the query. | search-domain |
| 7. Scoring and ranking | Scoring | `score_business_objective` | Apply domain-specific utility such as margin, priority, risk, contractual status, or campaign rules. | search-domain |
| 7. Scoring and ranking | Scoring | `normalize_scores` | Transform source- or model-specific scores onto a comparable scale. | search-domain |
| 7. Scoring and ranking | Scoring | `calibrate_scores` | Map raw scores to interpretable probabilities or empirically calibrated confidence. | search-domain |
| 7. Scoring and ranking | Scoring | `blend_scores` | Combine multiple normalized signals using a formula, model, or learned weights. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `rank_by_score` | Sort candidates by a selected score or blended objective. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `rerank_pointwise` | Re-score each candidate independently using a more expensive model. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `rerank_pairwise` | Order candidates using pairwise preference judgments. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `rerank_listwise` | Rank a candidate set jointly so interactions and coverage can influence order. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `rerank_cross_encoder` | Reorder a candidate window using a query-document cross-encoder. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `rerank_llm` | Reorder or grade candidates using an LLM with explicit relevance criteria. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `learning_to_rank` | Apply a trained ranking model over engineered, lexical, vector, behavioral, or business features. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `reciprocal_rank_fusion` | Fuse ranked lists using reciprocal rank contributions without requiring score comparability. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `weighted_rank_fusion` | Fuse ranked lists using source or query-specific rank weights. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `weighted_score_fusion` | Fuse candidate scores after normalization using weighted arithmetic or learned combination. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `cascade_rank` | Apply progressively more expensive ranking stages to progressively smaller candidate sets. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `interleave_rankings` | Mix rankings to compare systems online or preserve representation from multiple sources. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `diversify_mmr` | Select candidates that balance relevance against redundancy using maximal marginal relevance. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `diversify_by_group` | Apply quotas or penalties to improve source, document, topic, entity, or viewpoint diversity. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `pin_or_elevate` | Force designated candidates to selected ranks when rule conditions are satisfied. | search-domain |
| 7. Scoring and ranking | Ranking, fusion, and diversification | `exclude_or_demote` | Remove or lower candidates according to rules, risks, or business constraints. | search-domain |
| 8. Aggregation and analysis | Aggregation and analytics | `count` | Count candidates or records, optionally within groups. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `count_distinct` | Count unique values or entities. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `facet` | Return categorical value counts suitable for navigation or analysis. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `group_by` | Partition records by one or more keys for downstream aggregation. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `bucket` | Assign numeric, temporal, geographic, or score values to intervals. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `aggregate_statistics` | Compute sum, mean, minimum, maximum, variance, percentiles, or other statistics. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `pivot` | Reshape grouped values into a cross-tabulated representation. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `cluster_results` | Group candidates using lexical, embedding, graph, or metadata similarity. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `summarize_coverage` | Measure which requested entities, periods, sources, or subquestions are represented or missing. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `compare_groups` | Compute differences, ratios, overlaps, or contrasts between result groups. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `compute_trend` | Aggregate values over ordered time or version intervals. | runtime/data |
| 8. Aggregation and analysis | Aggregation and analytics | `extract_structured_records` | Convert documents or evidence into schema-conforming records for computation. | runtime/data |
| 9. Evidence and verification | Evidence operations | `extract_claims` | Identify factual, causal, comparative, or quantitative claims in text. | search-domain |
| 9. Evidence and verification | Evidence operations | `extract_evidence_spans` | Select the minimal source spans that support or refute a claim. | search-domain |
| 9. Evidence and verification | Evidence operations | `extract_quotes` | Return verbatim source excerpts with stable span provenance. | search-domain |
| 9. Evidence and verification | Evidence operations | `align_claim_to_evidence` | Associate each claim with candidate supporting and contradicting evidence. | search-domain |
| 9. Evidence and verification | Evidence operations | `verify_entailment` | Determine whether evidence logically supports a claim. | search-domain |
| 9. Evidence and verification | Evidence operations | `detect_contradiction` | Determine whether evidence conflicts with a claim or with other evidence. | search-domain |
| 9. Evidence and verification | Evidence operations | `corroborate_across_sources` | Require independent sources to support the same fact or relation. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_source_authority` | Check whether the source satisfies ownership, expertise, primary-source, or approved-domain rules. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_temporal_consistency` | Check whether evidence was valid for the relevant time and has not been superseded. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_schema` | Check extracted records against required types, fields, ranges, and invariants. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_link` | Check that a cited source resolves and that the referenced content is accessible. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_citation_relevance` | Check that the cited source or span is topically relevant to the associated claim. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_citation_correctness` | Check that the cited evidence actually supports the associated claim. | search-domain |
| 9. Evidence and verification | Evidence operations | `validate_citation_completeness` | Check that material factual claims have sufficient citations. | search-domain |
| 9. Evidence and verification | Evidence operations | `resolve_evidence_conflicts` | Represent, rank, or adjudicate conflicting claims while preserving dissenting evidence. | search-domain |
| 9. Evidence and verification | Evidence operations | `estimate_evidence_confidence` | Estimate confidence from evidence quality, agreement, model uncertainty, and missing information. | search-domain |
| 9. Evidence and verification | Evidence operations | `abstain_if_insufficient` | Return an explicit insufficient-evidence state rather than an unsupported conclusion. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `select_passages` | Choose the most useful evidence spans for the downstream consumer. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `compress_extractive` | Remove irrelevant sentences or tokens while retaining source wording and span traceability. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `compress_abstractive` | Produce a query-focused concise representation of evidence while preserving source links. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `remove_context_redundancy` | Eliminate repeated or semantically duplicative evidence. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `reorder_context` | Arrange evidence to improve coherence, priority, or model utilization. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `enforce_token_budget` | Trim or adapt selection and compression to a maximum token or byte budget. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `group_context_by_claim` | Organize evidence under the claim or subquestion it supports. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `pack_context` | Assemble instructions, evidence, metadata, and delimiters into a model-ready context bundle. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `generate_snippets_or_highlights` | Create concise result previews and mark matched or evidential regions. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `render_text` | Render selected results as readable plain text or Markdown. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `render_table` | Render schema-aligned records as a comparison or analytical table. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `render_json` | Render structured results under a specified machine-readable schema. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `render_citations` | Attach document-, passage-, sentence-, claim-, page-, or region-level citation markers. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `render_provenance` | Expose source identity, version, timestamps, retrieval path, and transformation lineage. | search-domain |
| 10. Context and output | Selection, compression, and rendering | `render_structured_response` | Return results, citations, execution metadata, and optional synthesis in a typed response object. | search-domain |
| 11. Runtime | Control-flow combinators | `compose` | Connect operations so each output becomes the next operation's input. | runtime |
| 11. Runtime | Control-flow combinators | `map` | Apply an operation independently to each item. | runtime |
| 11. Runtime | Control-flow combinators | `batch` | Execute one operation over a collection to reduce overhead. | runtime |
| 11. Runtime | Control-flow combinators | `parallel_map` | Apply an operation concurrently to independent items. | runtime |
| 11. Runtime | Control-flow combinators | `fan_out` | Expand one task into many independent operations. | runtime |
| 11. Runtime | Control-flow combinators | `fan_in` | Collect and reconcile the outputs of parallel branches. | runtime |
| 11. Runtime | Control-flow combinators | `branch` | Choose an execution path from a deterministic or model-produced condition. | runtime |
| 11. Runtime | Control-flow combinators | `iterate` | Repeat a retrieval or processing step until a stopping condition is met. | runtime |
| 11. Runtime | Control-flow combinators | `retry` | Re-execute a failed or low-quality operation under a defined retry policy. | runtime |
| 11. Runtime | Control-flow combinators | `backoff` | Delay retries according to a fixed, exponential, or source-aware schedule. | runtime |
| 11. Runtime | Control-flow combinators | `timeout` | Terminate an operation that exceeds its allowed runtime. | runtime |
| 11. Runtime | Control-flow combinators | `rate_limit` | Restrict request frequency or throughput to respect source and cost constraints. | runtime |
| 11. Runtime | Control-flow combinators | `limit_concurrency` | Bound the number of simultaneous operations. | runtime |
| 11. Runtime | Control-flow combinators | `fallback` | Invoke an alternate source, model, or strategy when the preferred path fails. | runtime |
| 11. Runtime | Control-flow combinators | `early_stop` | Terminate search when coverage, confidence, marginal gain, or budget criteria are satisfied. | runtime |
| 11. Runtime | Control-flow combinators | `paginate` | Traverse paged results using offsets, cursors, tokens, or search-after keys. | runtime |
| 11. Runtime | Control-flow combinators | `stream` | Consume or emit results incrementally rather than materializing the complete collection. | runtime |
| 11. Runtime | Control-flow combinators | `sandbox_execute` | Run generated search code in an isolated, resource-limited execution environment. | runtime |
| 12. State and observability | State, provenance, and telemetry | `cache_query_result` | Store a query result under source-, version-, user-, and parameter-aware cache keys. | runtime |
| 12. State and observability | State, provenance, and telemetry | `cache_document` | Store fetched content while tracking freshness and source version. | runtime |
| 12. State and observability | State, provenance, and telemetry | `memoize_operation` | Reuse deterministic operation outputs for identical inputs. | runtime |
| 12. State and observability | State, provenance, and telemetry | `persist_state` | Serialize intermediate candidate sets, records, plans, or evidence across turns. | runtime |
| 12. State and observability | State, provenance, and telemetry | `load_state` | Restore explicitly persisted intermediate state. | runtime |
| 12. State and observability | State, provenance, and telemetry | `checkpoint` | Save a recoverable pipeline state at a defined execution boundary. | runtime |
| 12. State and observability | State, provenance, and telemetry | `trace_operation` | Record parent-child execution spans, parameters, timings, status, and resource use. | runtime |
| 12. State and observability | State, provenance, and telemetry | `log_query_activity` | Record what was searched, against which source, and with which parameters. | runtime |
| 12. State and observability | State, provenance, and telemetry | `record_lineage` | Track how each output was derived from source items and transformations. | runtime |
| 12. State and observability | State, provenance, and telemetry | `explain_score` | Expose the components and calculations that produced a candidate's score or rank. | runtime |
| 12. State and observability | State, provenance, and telemetry | `record_snapshot` | Record source versions, model versions, configuration, and time for reproducibility. | runtime |
| 12. State and observability | State, provenance, and telemetry | `audit_event` | Write governance-relevant access, filtering, policy, and output events to an audit trail. | runtime |
| 12. State and observability | State, provenance, and telemetry | `record_metrics` | Emit latency, throughput, cost, token, error, candidate, and quality measurements. | runtime |
| 12. State and observability | State, provenance, and telemetry | `capture_error` | Store structured failure context without losing successful branch outputs. | runtime |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_precision` | Measure the proportion of retrieved items that are relevant. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_recall` | Measure the proportion of relevant items that were retrieved. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_hit_rate` | Measure whether at least one relevant item appears within a cutoff. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_mrr` | Measure the reciprocal rank of the first relevant result, averaged across queries. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_map` | Measure average precision across relevant results and then average across queries. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_ndcg` | Measure ranked graded relevance with position discounting and ideal-list normalization. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_err` | Estimate user satisfaction under a cascade model with graded relevance. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_diversity` | Measure novelty, subtopic coverage, source diversity, or redundancy in a result set. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_context_precision` | Measure whether relevant retrieved contexts are ranked ahead of irrelevant contexts. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_context_recall` | Measure whether the retrieved context contains the information needed to answer. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_faithfulness` | Measure whether generated claims are supported by the supplied evidence. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_citation_quality` | Measure citation validity, relevance, correctness, completeness, and granularity. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `evaluate_latency_cost` | Measure end-to-end and stage-level latency, throughput, token use, and monetary cost. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `run_offline_evaluation` | Evaluate a pipeline against a fixed dataset, qrels, references, or human judgments. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `run_online_experiment` | Compare ranking or pipeline variants using A/B testing, interleaving, or controlled rollout. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `collect_relevance_feedback` | Capture explicit judgments, clicks, reformulations, dwell, or model-generated feedback. | evaluation |
| 13. Evaluation and learning | Evaluation and improvement | `train_or_tune_component` | Fit or optimize a rewriter, retriever, ranker, router, threshold, or fusion policy from evaluation data. | evaluation |
| 14. Composite macros | Non-atomic convenience pipelines | `keyword_search` | Convenience pipeline combining lexical parsing, candidate generation, lexical scoring, filtering, and ranking. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `semantic_search` | Convenience pipeline combining embedding, vector retrieval, filtering, and similarity ranking. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `hybrid_search` | Composite pipeline that runs two or more retrieval modes and fuses their candidate lists. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `multi_query_search` | Composite pipeline that generates query variants, retrieves for each, deduplicates, and fuses results. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `rag_fusion` | Multi-query retrieval followed by reciprocal-rank fusion and optional reranking. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `agentic_retrieval` | Planner-driven decomposition, multi-source parallel retrieval, semantic ranking, and structured cited output. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `graph_rag_local` | Entity-anchored graph retrieval that expands a local neighborhood or paths for focused questions. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `graph_rag_global` | Community- or hierarchy-based retrieval and aggregation for corpus-wide sensemaking questions. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `corrective_rag` | Retrieval followed by quality assessment, query correction or alternate retrieval, and renewed evidence selection. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `deep_research` | Iterative planning, broad source discovery, targeted backfilling, extraction, verification, and report rendering. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `answer_with_citations` | Evidence selection, answer synthesis, claim-evidence alignment, citation rendering, and citation validation. | composite |
| 14. Composite macros | Non-atomic convenience pipelines | `search_as_code_program` | A generated executable program that composes atomic search primitives and runtime combinators for one task. | composite |

## Recommended SDK boundary

An Agentic Search SDK should expose the **search-domain** operations and typed **contracts**. The host runtime supplies loops, branching, concurrency, joins, retries, caching, serialization, and dataframe-style aggregation. Composite macros are optional shorthand and must remain bypassable so generated code can reach lower-level operations. This is exactly the split the support matrix encodes as *database-layer* vs *harness-layer (not eligible)*.
