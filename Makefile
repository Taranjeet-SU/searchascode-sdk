# One place for the commands the docs describe (issues.md STR-10).
#
# `ruff check search_as_code && mypy search_as_code && pytest -q` was written out in
# CLAUDE.md:62 and README.md:222 and matched NEITHER — CI ran pytest with --cov. Three copies
# of a build command in two markdown files and a YAML is drift waiting to happen; this is the
# single definition all three now point at. mem0 and every LangChain lib ship one.

.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install lint typecheck test test-all conformance check guard docs-links wheel clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## editable install with dev tooling
	$(PY) -m pip install -e '.[dev]'

lint:  ## ruff
	$(PY) -m ruff check search_as_code

typecheck:  ## mypy
	$(PY) -m mypy search_as_code

test:  ## unit tests (no services needed)
	$(PY) -m pytest tests/ -q --ignore=tests/test_opensearch.py --ignore=tests/test_diagnostic_playbook.py

test-all:  ## every test, including live-backend integration
	$(PY) -m pytest tests/ -q

conformance:  ## the adapter contract, across every installed backend
	$(PY) -m pytest tests/test_conformance.py -q

guard:  ## refuse customer/internal artifacts in the tracked tree (GOV-1/2/3)
	$(PY) scripts/check_no_customer_artifacts.py --check-tree

docs-links:  ## check every relative link in the markdown resolves (DOC-6)
	$(PY) scripts/check_doc_links.py --public --staged

wheel:  ## build the wheel and smoke-test it in a clean venv (STR-1)
	$(PY) scripts/smoke_wheel.py

check: lint typecheck test guard docs-links  ## what CI runs — "keep it green"

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
