# Contributing

`README.md` documented a contribution workflow that existed nowhere as a file (issues.md STR-9).
This is that file. Flask, mem0 and LangChain all ship one; the conventions below are this repo's.

## Setup

```bash
git clone https://github.com/oro-jackson/searchascode-sdk.git
cd searchascode-sdk
make install                 # pip install -e '.[dev]'
pre-commit install           # file-hygiene + ruff + mypy on every commit
```

We develop in parallel using **git worktrees** (isolated checkouts per feature):

```bash
git worktree add ../sac-<feature> -b feat/<feature>
```

## Keeping it green

One command, defined once in the `Makefile` — CI runs the same target:

```bash
make check                   # lint + typecheck + test + guard + docs-links
```

| target | what it does |
|---|---|
| `make test` | unit tests, no services required |
| `make test-all` | adds the live-OpenSearch integration tests |
| `make conformance` | the adapter contract across every installed backend |
| `make wheel` | builds the wheel and runs the README quickstart in a clean venv |
| `make guard` | refuses customer/internal artifacts in the tracked tree |

## Adding a backend

Implement one `VectorStore` (`search_as_code/adapters/base.py`); `memory.py` is the executable
spec. **`tests/test_conformance.py` is the contract** — add your backend to its `BACKENDS` map
and the whole suite runs against it. A backend whose client library is absent is skipped, never
silently passed. Then update `docs/DATABASES.md`.

## Adding a primitive

Portable, model-free primitives go in `primitives.py`; native ones become a backend method with
an emulated fallback in the primitive layer. Update `docs/PRIMITIVES.md` and
`docs/DATABASES.md`.

## Reporting numbers

Any measured claim needs an interval, not a bare mean — use
`search_as_code.metrics.bootstrap_ci` / `compare`. A difference whose CI includes zero is not a
result. See `issues.md` DJ-2 for why this rule exists.

## Repo conventions

- **`issues.md` is the standing audit log.** Found a problem — approach, code, or redundancy?
  Append an entry with a `file:line` reference before you finish. Append only; never rewrite
  history. See `CLAUDE.md`.
- **Never `git add -A`.** Stage explicit paths. The guard hook is the backstop, not the rule.
- **Standard vs internal** (`soul.md` rule 1): generalizable code goes in `search_as_code/`;
  customer- or corpus-specific work never reaches the public repo.
- **Pull before you start and before you push** — several agents share a checkout.
