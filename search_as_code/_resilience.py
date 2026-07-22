"""Retry / batching helpers for network-backed adapters.

Small, dependency-free utilities so every real adapter gets consistent,
testable resilience: exponential backoff with jitter around transient backend
calls, and chunking for bulk upserts. The in-memory backend needs none of this;
the network adapters (OpenSearch/Qdrant/Chroma/pgvector) opt in.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from .errors import BackendError, SacError

T = TypeVar("T")

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF = 0.5  # seconds; grows 0.5, 1.0, 2.0, ...
DEFAULT_BATCH_SIZE = 500


def with_retry(
    fn: Callable[..., T],
    *args: Any,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    backend: str = "backend",
    op: str = "call",
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """Call ``fn(*args, **kwargs)``, retrying transient ``exceptions``.

    Retries ``attempts`` times with exponential backoff (deterministic multiples
    of ``backoff``; ``sleep`` is injectable so tests run instantly). On final
    failure the underlying error is wrapped in :class:`BackendError` (which is
    also a ``RuntimeError``) with the original attached via ``__cause__``.
    ``SacError`` is never retried or re-wrapped — those are our own typed,
    non-transient errors.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except SacError:
            raise  # our own typed, non-transient errors: never retry or re-wrap
        except exceptions as e:  # noqa: BLE001 - deliberately broad, caller-scoped
            last = e
            if i >= attempts:
                break
            sleep(backoff * (2 ** (i - 1)))
    raise BackendError(
        f"{backend} {op} failed after {attempts} attempt(s)",
        backend=backend,
        op=op,
        cause=type(last).__name__ if last else None,
    ) from last


def retry(
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    backend: str = "backend",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :func:`with_retry` (``op`` defaults to the func name)."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return with_retry(
                fn, *args,
                attempts=attempts, backoff=backoff, exceptions=exceptions,
                backend=backend, op=fn.__name__, **kwargs,
            )

        return wrapper

    return deco


def chunked(seq: Sequence[T] | Iterable[T], size: int = DEFAULT_BATCH_SIZE) -> Iterator[list[T]]:
    """Yield ``seq`` in lists of at most ``size`` (for batched upserts)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    batch: list[T] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
