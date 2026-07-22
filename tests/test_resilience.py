"""Retry/backoff and batching helpers used by the network adapters."""

import pytest

from search_as_code import _resilience as R
from search_as_code.errors import BackendError, GeneratorRequiredError, SacError

NO_SLEEP = lambda _s: None  # noqa: E731 - keep tests instant


def test_chunked_splits_evenly_and_remainder():
    assert list(R.chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(R.chunked([], 2)) == []
    assert list(R.chunked(range(3), 10)) == [[0, 1, 2]]


def test_chunked_rejects_bad_size():
    with pytest.raises(ValueError):
        list(R.chunked([1, 2], 0))


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    out = R.with_retry(flaky, attempts=3, backoff=0, sleep=NO_SLEEP)
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_to_backend_error_preserving_cause():
    def always_fail():
        raise ConnectionError("down")

    with pytest.raises(BackendError) as ei:
        R.with_retry(always_fail, attempts=2, backoff=0, sleep=NO_SLEEP, backend="opensearch", op="bulk")
    assert ei.value.code == "E_BACKEND"
    assert ei.value.context["backend"] == "opensearch"
    assert isinstance(ei.value.__cause__, ConnectionError)


def test_retry_never_retries_or_wraps_sacerror():
    calls = {"n": 0}

    def typed_error():
        calls["n"] += 1
        raise GeneratorRequiredError("needs an llm")

    with pytest.raises(SacError) as ei:
        R.with_retry(typed_error, attempts=5, backoff=0, sleep=NO_SLEEP)
    assert not isinstance(ei.value, BackendError)  # not re-wrapped
    assert calls["n"] == 1  # not retried


def test_retry_requires_at_least_one_attempt():
    with pytest.raises(ValueError):
        R.with_retry(lambda: 1, attempts=0)


def test_retry_decorator_form():
    calls = {"n": 0}

    @R.retry(attempts=2, backoff=0)
    def sometimes():
        calls["n"] += 1
        raise TimeoutError("slow")

    with pytest.raises(BackendError):
        sometimes()
    assert calls["n"] == 2
