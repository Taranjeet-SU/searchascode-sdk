# soul.md — how this repo's docs work (read me first)

The "constitution" for agents (human or LLM) working in this repo. It says **what each
Markdown file is for, what to put in it, and when to update it** — so knowledge is captured
consistently and nothing important gets lost in code or chat.

> This is our `CLAUDE.md`-style always-loaded file. Design principles borrowed from Anthropic's
> guidance (see `research.md`): **keep it < ~200 lines** (models reliably follow ~150–200
> instructions before "context rot"); **progressive disclosure** — this file stays small and
> *points to* the detailed docs by path, never inlines them; **just-in-time memory** — record
> learnings in files (`learnings_standard.md`, `research.md`) and read them back on demand;
> **skills are evaluation-driven** — find a capability gap first, then add the primitive/doc.

## Golden rules
1. **Standard vs internal.** Generalizable code/learnings go to the **standard SDK** (`search_as_code/`)
   and the public repo (`main` → GitHub). Client/customer-specific work (data, models, tokens,
   bespoke scripts) stays **internal** (e.g. the `test1` branch) and **gitignored** — never pushed.
2. **Improve the SDK, don't fork it.** Even custom scripts should *import and use* the standard SDK.
   If you find a generalizable fix in custom work, land it in `search_as_code/` and note it in
   `learnings_standard.md`.
3. **Capture as you go.** Every non-trivial finding, decision, or gotcha lands in the right doc below
   the same session it happens — not "later".
4. **Cite sources.** Any claim/technique that came from a paper/article goes into `research.md`.
5. **Honesty.** Report real numbers, including negative results. No cherry-picking.

## The docs and what goes in each
| file | purpose | what to put in it | update when |
|---|---|---|---|
| `README.md` | the pitch + quickstart | what SAC is, headline benchmark numbers, how to install/run, links to deep docs | a headline result or the public API changes |
| `soul.md` | **this file** — doc constitution | the rules above + this table | the doc system itself changes |
| `learnings_standard.md` | generalizable learnings → belong in the SDK | reusable fixes/patterns/insights (embedder quirks, adapter patterns, eval methodology, ops) | you learn something reusable in custom work |
| `research.md` | running research log | every paper/article/source we drew on, with a 1-line takeaway and how it maps to our code | you read/use any external source |
| `CHANGELOG.md` | running work log + status board | what was built/changed/measured, tasks (done/in-progress/planned), gotchas for future agents | every work session |
| `docs/CONCEPT.md` | the core idea | search-as-code thesis, how source articles map to code | the concept evolves |
| `docs/PRIMITIVES.md` | canonical primitive taxonomy | the full primitive catalog (implemented + planned) | a primitive is added/planned |
| `docs/DATABASES.md` | primitive × backend matrix | which primitive each backend supports/emulates | an adapter changes |
| `docs/SELECTION.md` | prompt surface + decision rules | how the SDK is shown to the LLM, when to call/chain each primitive | primitive selection logic changes |
| `docs/CACHING.md` | efficient LLM prompting | byte-stable prefix / prompt-cache strategy | the prompt surface changes |
| `docs/RESEARCH.md` | curated research base | the large curated source list (stable) | major curation pass (day-to-day goes in `research.md`) |
| `docs/<TOPIC>.md`, `phase*/RESULTS.md`, `*_REPORT.md` | topic plans + benchmark reports | design plans and measured results per phase/topic | a phase runs or a plan is made |

## How an LLM agent should use these
- **Before coding:** read `soul.md` → `CHANGELOG.md` (status) → the relevant `docs/*`.
- **While working:** prefer standard SDK primitives; if introspecting a new corpus, run the
  introspection primitives (see `docs/PRIMITIVES.md`) and feed the schema to yourself first.
- **After a finding:** update `CHANGELOG.md`; if generalizable → `learnings_standard.md`; if from a
  source → `research.md`; if it changes the primitive catalog → `docs/PRIMITIVES.md`.
- **Before pushing:** confirm nothing customer-specific/secret is staged (see rule 1).
