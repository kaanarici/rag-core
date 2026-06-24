from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.config import RerankerConfig


def test_reranker_config_normalizes_provider_name() -> None:
    config = RerankerConfig(provider=" CoHere ")

    assert config.provider == "cohere"


@pytest.mark.parametrize("provider", ["", "   ", None, True, cast(Any, 123)])
def test_reranker_config_rejects_invalid_provider(provider: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        RerankerConfig(provider=cast(Any, provider))

    assert str(exc_info.value) == (
        "RerankerConfig.provider must be a non-empty string"
    )


def test_reranker_config_strict_provider_defaults_false() -> None:
    assert RerankerConfig().strict_provider is False


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_reranker_config_rejects_non_bool_strict_provider(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        RerankerConfig(strict_provider=cast(Any, value))

    assert str(exc_info.value) == (
        "RerankerConfig.strict_provider must be a boolean"
    )
