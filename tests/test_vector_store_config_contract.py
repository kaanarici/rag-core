from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.config import VectorStoreConfig


def test_vector_store_config_normalizes_provider() -> None:
    assert VectorStoreConfig(provider=" Qdrant ").provider == "qdrant"


@pytest.mark.parametrize("provider", ["", "   ", None, True, cast(Any, 123)])
def test_vector_store_config_rejects_invalid_provider(provider: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        VectorStoreConfig(provider=cast(Any, provider))

    assert str(exc_info.value) == "VectorStoreConfig.provider must be a non-empty string"


def test_vector_store_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError) as exc_info:
        VectorStoreConfig(provider="unknown")

    assert str(exc_info.value) == "VectorStoreConfig.provider must be one of: qdrant"
