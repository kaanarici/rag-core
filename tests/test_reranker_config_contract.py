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
