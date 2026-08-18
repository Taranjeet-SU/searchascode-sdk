# One definition of "green" — the same set CI runs.
.PHONY: check lint type test docs-links
check: lint type test docs-links
lint:
	python3 -m ruff check search_as_code
type:
	python3 -m mypy search_as_code
test:
	python3 -m pytest tests/ -q --ignore=tests/test_opensearch.py
docs-links:
	python3 scripts/check_doc_links.py
