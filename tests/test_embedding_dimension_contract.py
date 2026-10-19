from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.config import EmbeddingConfig
from rag_core.config import LOCAL_EMBEDDING_DIMENSIONS, LOCAL_EMBEDDING_MODEL
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions


def test_embedding_config_accepts_positive_integer_dimensions() -> None:
    assert EmbeddingConfig(dimensions=1536).dimensions == 1536


@pytest.mark.parametrize("dimensions", [0, -1, True, cast(Any, 1.5), cast(Any, "1536")])
def test_embedding_config_rejects_invalid_dimensions(dimensions: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        EmbeddingConfig(dimensions=cast(Any, dimensions))

    assert str(exc_info.value) == (
        "EmbeddingConfig.dimensions must be a positive integer"
    )


def test_resolve_embedding_dimensions_accepts_positive_integer_dimensions() -> None:
    assert (
        resolve_embedding_dimensions(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=512,
        )
        == 512
    )


def test_resolve_local_embedding_model_default_dimensions() -> None:
    assert (
        resolve_embedding_dimensions(
            provider="local",
            model=LOCAL_EMBEDDING_MODEL,
            dimensions=None,
        )
        == LOCAL_EMBEDDING_DIMENSIONS
    )


def test_resolve_local_embedding_model_rejects_dimension_override() -> None:
    with pytest.raises(ValueError, match="only the default dimension 384"):
        resolve_embedding_dimensions(
            provider="local",
            model=LOCAL_EMBEDDING_MODEL,
            dimensions=768,
        )


@pytest.mark.parametrize("dimensions", [0, -1, True, cast(Any, 1.5), cast(Any, "1536")])
def test_resolve_embedding_dimensions_rejects_invalid_dimensions(
    dimensions: object,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_embedding_dimensions(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=cast(Any, dimensions),
        )

    assert str(exc_info.value) == "embedding dimensions must be a positive integer"
