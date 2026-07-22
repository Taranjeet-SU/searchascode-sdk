"""The typed error hierarchy: stable codes, structured context, and — crucially —
back-compat with the built-in exceptions callers may already be catching.
"""

import pytest

import search_as_code as sac
from search_as_code import errors
from search_as_code.filters import matches, validate
from search_as_code.primitives import extract
from search_as_code.types import ResultSet


def test_codes_are_stable():
    assert errors.SacError().code == "E_SAC"
    assert errors.InvalidFilterError().code == "E_INVALID_FILTER"
    assert errors.InvalidModeError().code == "E_INVALID_MODE"
    assert errors.BackendNotFoundError().code == "E_BACKEND_NOT_FOUND"
    assert errors.MissingDependencyError("x").code == "E_MISSING_DEPENDENCY"
    assert errors.DimensionMismatchError().code == "E_DIMENSION_MISMATCH"
    assert errors.BackendError().code == "E_BACKEND"


@pytest.mark.parametrize("cls,builtin", [
    (errors.ConfigurationError, ValueError),
    (errors.BackendNotFoundError, ValueError),
    (errors.InvalidArgumentError, ValueError),
    (errors.InvalidFilterError, ValueError),
    (errors.InvalidModeError, ValueError),
    (errors.DimensionMismatchError, ValueError),
    (errors.MissingDependencyError, ImportError),
    (errors.InvalidEmbedderError, TypeError),
    (errors.GeneratorRequiredError, RuntimeError),
    (errors.ExtractorRequiredError, RuntimeError),
    (errors.BackendError, RuntimeError),
    (errors.EmbeddingError, RuntimeError),
])
def test_backcompat_subclassing(cls, builtin):
    assert issubclass(cls, builtin)
    assert issubclass(cls, errors.SacError)


def test_str_shows_code_and_context():
    e = errors.InvalidModeError("bad mode", mode="fuzzy", allowed=["dense"])
    assert "[E_INVALID_MODE]" in str(e)
    assert e.context["mode"] == "fuzzy"
    assert "fuzzy" in str(e)


def test_missing_dependency_message_has_install_hint():
    e = errors.MissingDependencyError("opensearch-py", extra="search-as-code[opensearch]")
    assert e.package == "opensearch-py"
    assert "pip install 'search-as-code[opensearch]'" in str(e)


def test_connect_unknown_backend_is_typed_and_backcompat():
    with pytest.raises(errors.BackendNotFoundError) as ei:
        sac.connect("nope")
    assert ei.value.code == "E_BACKEND_NOT_FOUND"
    assert "nope" in str(ei.value)
    with pytest.raises(ValueError):  # back-compat
        sac.connect("nope")


def test_unknown_filter_operator_is_typed():
    with pytest.raises(errors.InvalidFilterError):
        matches({"a": 1}, {"a": {"$bogus": 1}})
    with pytest.raises(errors.InvalidFilterError):
        validate({"a": {"$bogus": 1}})
    with pytest.raises(ValueError):  # back-compat
        validate({"a": {"$bogus": 1}})


def test_generator_and_extractor_required():
    s = sac.Session("memory")
    s.add([{"id": "1", "text": "x"}])
    with pytest.raises(errors.GeneratorRequiredError):
        s.expand_search("x")
    with pytest.raises(RuntimeError):  # back-compat
        s.decompose_search("x")
    with pytest.raises(errors.ExtractorRequiredError):
        extract(ResultSet(), {}, "pull records")


def test_invalid_embedder_and_provider():
    with pytest.raises(errors.InvalidEmbedderError):
        sac.as_embedder(123)
    with pytest.raises(TypeError):  # back-compat
        sac.as_embedder(123)
    with pytest.raises(errors.ConfigurationError):
        sac.get_embedder("bogus")
