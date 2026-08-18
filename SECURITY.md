# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/Taranjeet-SU/searchascode-sdk/security/advisories/new)
rather than opening a public issue. We aim to acknowledge within a few working days.

## Scope and threat model

`search-as-code` **executes LLM-authored Python** (`search_as_code/sandbox.py`,
`harness/forge.py`) and **LLM-authored OpenSearch query bodies** (`harness/os_query.py`). Treat
both as untrusted input.

- The sandbox restricts builtins and imports. It is a **guard-rail against accidents, not a
  security boundary against an adversarial prompt.** Do not run untrusted user queries through
  it on a host you care about; use a container or a hardened backend (Docker/e2b are on the
  roadmap).
- `os_query` validates authored bodies against a read-only allowlist and rejects
  scripts/aggregations. This allowlist was dead code until 2026-08 (issues.md SDK-C1) — pin a
  version at or after that fix if you rely on it.
- Filters are validated at the boundary; an operator a backend cannot express raises rather
  than silently running unfiltered (SDK-C2, ADP-3).

## Supported versions

Pre-1.0: only the latest release receives fixes.

## Secrets

Never commit credentials. `scripts/check_no_customer_artifacts.py` runs in pre-commit and CI and
refuses private keys, `.pem`/`.env`/`.secrets` files, and customer-identifying paths.
