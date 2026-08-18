"""Typed exceptions with stable error codes.

Every error the SDK raises is a :class:`SacError` carrying a stable string
``.code`` (e.g. ``"E_INVALID_FILTER"``) for programmatic handling, logging, and
metrics. Each subclass *also* inherits from the built-in exception that best
matches its meaning (``ValueError`` / ``TypeError`` / ``RuntimeError`` /
``ImportError``), so existing ``except ValueError:`` handlers — and code written
before this hierarchy existed — keep working unchanged.

    try:
        sac.connect("nope")
    except sac.BackendNotFoundError as e:
        log.warning("bad backend", code=e.code, **e.context)
    except ValueError:
        ...   # still catches it (back-compat)

Codes are stable identifiers; the human message may change, the code should not.
"""

from __future__ import annotations

from typing import Any


class SacError(Exception):
    """Base class for every error raised by search-as-code.

    Carries a stable ``code`` and an optional structured ``context`` dict so
    callers can branch on the code and log the details without parsing strings.
    """

    code: str = "E_SAC"

    def __init__(self, message: str = "", **context: Any):
        self.context: dict[str, Any] = context
        if context:
            detail = ", ".join(f"{k}={v!r}" for k, v in context.items())
            message = f"{message} ({detail})" if message else detail
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:  # keep the code visible in tracebacks/logs
        base = super().__str__()
        return f"[{self.code}] {base}" if base else f"[{self.code}]"


# ---- configuration / wiring --------------------------------------------------
class ConfigurationError(SacError, ValueError):
    """The harness was set up incorrectly (bad backend, provider, options)."""

    code = "E_CONFIG"


class BackendNotFoundError(ConfigurationError):
    """``connect(backend)`` was given an unknown backend name."""

    code = "E_BACKEND_NOT_FOUND"


class MissingDependencyError(SacError, ImportError):
    """An optional dependency for a backend/provider is not installed."""

    code = "E_MISSING_DEPENDENCY"

    def __init__(self, package: str, extra: str | None = None, **context: Any):
        hint = f"pip install '{extra}'" if extra else f"pip install {package}"
        self.package = package
        self.extra = extra
        super().__init__(
            f"optional dependency {package!r} is not installed; {hint}", **context
        )


# ---- bad arguments -----------------------------------------------------------
class InvalidArgumentError(SacError, ValueError):
    """A user-supplied argument is out of range or the wrong shape."""

    code = "E_INVALID_ARGUMENT"


class InvalidModeError(InvalidArgumentError):
    """An unknown search ``mode`` was requested."""

    code = "E_INVALID_MODE"


class InvalidFilterError(InvalidArgumentError):
    """A metadata filter used an unknown operator or malformed structure."""

    code = "E_INVALID_FILTER"


class DimensionMismatchError(InvalidArgumentError):
    """A vector's dimensionality does not match the store/embedder's."""

    code = "E_DIMENSION_MISMATCH"


class InvalidEmbedderError(SacError, TypeError):
    """``embedder`` is neither an :class:`Embedder` nor a callable."""

    code = "E_INVALID_EMBEDDER"


# ---- missing pluggable callables ---------------------------------------------
class GeneratorRequiredError(SacError, RuntimeError):
    """A query-side primitive needs an LLM ``generator`` but none was supplied."""

    code = "E_GENERATOR_REQUIRED"


class ExtractorRequiredError(SacError, RuntimeError):
    """``extract()`` needs an ``extractor`` callable but none was supplied."""

    code = "E_EXTRACTOR_REQUIRED"


# ---- runtime / backend failures ----------------------------------------------
class EmbeddingError(SacError, RuntimeError):
    """The embedder failed or returned a malformed result."""

    code = "E_EMBEDDING"


class BackendError(SacError, RuntimeError):
    """A backend call failed (after any retries) — network, timeout, or server."""

    code = "E_BACKEND"


__all__ = [
    "SacError",
    "ConfigurationError",
    "BackendNotFoundError",
    "MissingDependencyError",
    "InvalidArgumentError",
    "InvalidModeError",
    "InvalidFilterError",
    "DimensionMismatchError",
    "InvalidEmbedderError",
    "GeneratorRequiredError",
    "ExtractorRequiredError",
    "EmbeddingError",
    "BackendError",
]
