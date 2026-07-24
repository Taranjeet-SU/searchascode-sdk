# Database support matrix

How each vector database maps onto the [canonical primitive taxonomy](PRIMITIVES.md).
**Legend:** ✅ Supported · ⚪ Not eligible · ❌ Not supported · 🕒 Planned.

*"Not eligible"* means the class is **not a database concern** — it belongs to the
SDK harness/host runtime (query rewriting, planning, sandboxed execution, evidence
verification, output rendering, evaluation). That split is the point of the
project: the DB owns retrieval, the harness owns everything around it.

## Primitive class × database

| # · Primitive class | Layer | Mem | Qdr | Chr | pgv | OS | Pine | Weav | Milv | ES | Vsp | Mgo | Rds |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 · Data contracts | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 1 · Source & corpus | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 · Query processing | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 3 · Search planning | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 4 · Candidate generation | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 5 · Candidate manipulation | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 6 · Content materialization | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 7 · Scoring & ranking | **db** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8 · Aggregation & analysis | **db** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |
| 9 · Evidence & verification | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 10 · Context & output | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 11 · Runtime | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 12 · State & observability | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 13 · Evaluation & learning | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| 14 · Composite macros | harness | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |

## Capability drill-down (where the variation lives)

| Capability (class) | Mem | Qdr | Chr | pgv | OS | Pine | Weav | Milv | ES | Vsp | Mgo | Rds |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Source & schema introspection (1) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dense vector search (4) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Lexical / full-text BM25 (4) | ✅ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sparse-neural — SPLADE/ELSER/BM42 (4) | ❌ | ✅ | ❌ | 🕒 | 🕒 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Multi-vector / late-interaction — ColBERT (4) | ❌ | ✅ | ❌ | 🕒 | 🕒 | ✅ | 🕒 | ✅ | 🕒 | ✅ | ❌ | ❌ |
| Structured / scalar / SQL retrieval (4) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | 🕒 | 🕒 | ✅ | ✅ | ✅ | ✅ |
| Graph / traversal / community (4) | ❌ | ❌ | ❌ | 🕒 | ❌ | ❌ | 🕒 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Regex / exact-match (4) | ✅ | 🕒 | ✅ | ✅ | ✅ | ❌ | ❌ | 🕒 | ✅ | 🕒 | ✅ | 🕒 |
| Phrase / proximity search (4·7) | 🕒 | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | 🕒 |
| Fuzzy / wildcard / prefix (4) | 🕒 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | 🕒 | ✅ | 🕒 |
| Fielded / multi-field boost (4) | 🕒 | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | 🕒 |
| More-like-this / find-similar (4) | 🕒 | ✅ | ❌ | 🕒 | ✅ | ✅ | ✅ | 🕒 | ✅ | ✅ | ❌ | ❌ |
| Random sample / browse-scroll (4) | 🕒 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Geospatial / temporal retrieval (4·5) | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |
| Metadata filtering & governance (5) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Native indexing / upsert (6) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Server-side embedding (6) | ❌ | 🕒 | ✅ | ❌ | 🕒 | ✅ | ✅ | ✅ | ✅ | ✅ | 🕒 | ❌ |
| Scoring — BM25 / vector similarity (7) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hybrid fusion — RRF / score (7) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Native reranking — cross-encoder / multi-stage (7) | ❌ | ✅ | ❌ | ❌ | 🕒 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Diversification / MMR (7) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🕒 | ❌ | ❌ |
| Aggregation & analytics (8) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |
| Facet / cardinality / stats (8) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🕒 | ✅ | ✅ | ✅ | ✅ |

**A ❌ is "not native," not "unavailable to your agent."** Via capability
negotiation the SDK exposes hybrid, keyword, regex, rerank (pluggable), MMR,
dedup, freshness, and compression on **every** backend by emulating them
client-side — so agent code is portable regardless of the marks above. The matrix
reflects native DB capability as of the 2025–2026 research base
([RESEARCH.md](RESEARCH.md)) and will drift as vendors ship features.

**Columns:** Mem = in-memory reference · Qdr Qdrant · Chr Chroma · pgv pgvector ·
OS OpenSearch · Pine Pinecone · Weav Weaviate · Milv Milvus · ES Elasticsearch ·
Vsp Vespa · Mgo MongoDB Atlas · Rds Redis.

**Adapters shipped today:** `memory`, `qdrant`, `chroma`, `pgvector`, `opensearch`,
`faiss`, `sqlite` (plus `nmslib`, `milvus` references). The rest are
native-capability references pending adapters — each is a thin `VectorStore`
implementation ([adapters/base.py](../search_as_code/adapters/base.py)).

**New OpenSearch-native primitives** (`adapters/opensearch.py`, verified against a
live cluster — see `tests/test_opensearch.py`): `query_phrase` (phrase/proximity
with slop), `query_fielded` (multi-field boosts), `query_prefix`, `query_wildcard`,
`query_fuzzy`, `more_like_this` (recommendation/find-similar), `random_sample`
(seeded), `browse` (query-less enumeration via `search_after`), and analysis
helpers `facet`, `count_distinct`, `stats`. These map to taxonomy primitives
`phrase_search`·`proximity_search`·`fielded_search`·`field_boost`·`prefix_search`·
`wildcard_search`·`fuzzy_search`·`recommendation_search`·`random_sample`·
`browse_or_scroll`·`facet`·`count_distinct`·`aggregate_statistics`. Next up for
OpenSearch: `sparse_neural_search` (neural-sparse/ELSER), native `rerank`
(ml-commons), and `multi_vector_search` (ColBERT) — all need an ML model/plugin.
