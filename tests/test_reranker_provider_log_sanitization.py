from __future__ import annotations

import traceback

import pytest

from rag_core.search.providers import reranker as reranker_module
from rag_core.search.providers.reranker import create_reranker

SECRET = "sk-test-secret"


class ProviderSecretError(RuntimeError):
    pass


def _fail_create(name: str, **kwargs: object) -> object:
    assert name == "cohere"
    raise ProviderSecretError(f"raw init detail with api key {SECRET}")


def test_reranker_init_error_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RERANKER_STRICT_PROVIDER", raising=False)
    monkeypatch.setattr(reranker_module.RERANKER_PROVIDERS, "create", _fail_create)

    with pytest.raises(ValueError) as exc_info:
        create_reranker(provider="cohere", api_key=SECRET)

    message = str(exc_info.value)
    assert "Failed to initialize reranker provider 'cohere'" in message
    assert "error_type=ProviderSecretError" in message
    assert "raw init detail" not in message
    assert SECRET not in message

    cause = exc_info.value.__cause__
    assert cause is not None
    assert type(cause).__name__ == "_SanitizedRerankerInitError"
    assert getattr(cause, "provider") == "cohere"
    assert getattr(cause, "error_type") == "ProviderSecretError"
    assert not isinstance(cause, ProviderSecretError)
    assert "raw init detail" not in str(cause)
    assert SECRET not in str(cause)

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "raw init detail" not in formatted
    assert SECRET not in formatted
    assert exc_info.value.__suppress_context__ is True
