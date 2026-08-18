# Pinned requirements

Flask ships `uv.lock`, mem0 `poetry.lock`, LangChain a lock per lib. This repo had none, and
every dependency was an open-ended floor (issues.md STR-4). The cost is on record: transformers
5.x silently corrupted ReasonIR's rotary `inv_freq` and collapsed recall (BC-4), with the
workaround living in a hand-maintained venv *outside* the repo.

`pyproject.toml` keeps the permissive floors that a library should publish. These files pin what
*we* develop and measure against, so a run is reproducible.

```bash
pip install -r requirements/dev.txt          # lint/type/test tooling
pip install -r requirements/experiments.txt  # + the ML stack the experiments need
```

Regenerate with `uv pip compile pyproject.toml --extra dev -o requirements/dev.txt`
(or `pip-compile`) when a dependency changes.
