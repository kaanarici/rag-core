from __future__ import annotations

import pytest

from rag_core.config import DEFAULT_RERANKER_PROVIDER
from rag_core.search.providers import reranker as reranker_module
from rag_core.search.providers.reranker import NoOpReranker, create_reranker
from rag_core.search.providers.reranker_resolution import resolve_reranker_provider


def test_invalid_strict_provider_env_preserves_fallback_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("RERANKER_STRICT_PROVIDER", "definitely")

    reranker = create_reranker(provider="cohere")

    assert isinstance(reranker, NoOpReranker)
    assert getattr(reranker, "_rag_core_provider_requested") == "cohere"
    assert getattr(reranker, "_rag_core_provider_effective") == DEFAULT_RERANKER_PROVIDER
    assert getattr(reranker, "_rag_core_fallback_reason") == "missing_cohere_api_key"


def test_whitespace_explicit_api_key_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "env-secret")

    effective, fallback_reason = resolve_reranker_provider(
        provider="cohere",
        api_key="   ",
    )
    assert effective == "cohere"
    assert fallback_reason is None


def test_cohere_resolution_accepts_standard_co_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "cohere-secret")

    effective, fallback_reason = resolve_reranker_provider(provider="cohere")

    assert effective == "cohere"
    assert fallback_reason is None


def test_create_reranker_passes_resolved_cohere_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create(name: str, **kwargs: object) -> NoOpReranker:
        captured["name"] = name
        captured.update(kwargs)
        return NoOpReranker()

    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "cohere-env-secret")
    monkeypatch.setattr(reranker_module.RERANKER_PROVIDERS, "create", fake_create)

    create_reranker(provider="cohere")

    assert captured["name"] == "cohere"
    assert captured["api_key"] == "cohere-env-secret"


@pytest.mark.parametrize("strict_value", ["true", "on", "1"])
def test_truthy_strict_provider_env_raises_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
    strict_value: str,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("RERANKER_STRICT_PROVIDER", strict_value)

    with pytest.raises(ValueError, match="missing_cohere_api_key"):
        create_reranker(provider="cohere")


def test_strict_argument_raises_on_missing_key_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER_STRICT_PROVIDER", raising=False)

    with pytest.raises(ValueError, match="missing_cohere_api_key"):
        create_reranker(provider="cohere", strict=True)


def test_default_lenient_degrades_to_noop_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER_STRICT_PROVIDER", raising=False)

    reranker = create_reranker(provider="cohere")

    assert isinstance(reranker, NoOpReranker)
    assert getattr(reranker, "_rag_core_fallback_reason") == "missing_cohere_api_key"
