# Introspection primitives — schema-first agentic retrieval

Let the agent discover the **data shape** before it writes retrieval code — "the first call tells
the agent what data exists and what queries are safe to construct" (see `research.md`: RushDB Schema
API, Google Agent Search auto-schema, MCP schema tools).

## Primitives
| primitive | returns |
|---|---|
| `store.describe_schema()` | `{count, fields:{name:type}, text_field, vector_field, sample_text}` |
| `store.sample(n=5)` | a few representative `Document`s |
| `content_type(text)` | `'table' \| 'fact-card' \| 'list' \| 'code' \| 'prose' \| 'short-fact' \| 'empty'` |
| `Session.describe(n_samples=4)` | LLM-ready corpus profile: schema + **content-type mix** + snippets |

## The pattern
1. The agent calls `session.describe()` **first**.
2. The profile (field names/types, whether the store holds prose vs tables vs curated fact-cards,
   sample rows) is injected into the agent prompt.
3. The agent picks primitives that fit the data shape:
   - **fact-cards / part numbers / IDs** → `keyword` / exact match / `regex`
   - **prose** → `dense` / `hyde_search`
   - **tables** → keyword + `compress`
   - mixed → `fuse` across strategies.

## Backend support
- **OpenSearch**: real — reads `_mapping` for field types, `random_score` sample.
- **memory**: metadata keys + sample.
- **others**: inherit the base default (`count` + `backend`) until overridden.

## Why
On a heterogeneous store (e.g. curated fact-cards in one index, prose docs in another), knowing the
shape up front lets the agent avoid dense search where exact match wins, and vice-versa — the biggest
lever we found for retrieval quality on domain data.
