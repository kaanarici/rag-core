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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("voyage-4", 1024),
        ("voyage-4-lite", 1024),
        ("voyage-4-large", 1024),
        ("voyage-4-nano", 1024),
        ("voyage-code-3", 1024),
        ("voyage-3-large", 1024),
        ("voyage-3.5", 1024),
        ("voyage-3.5-lite", 1024),
        ("voyage-3", 1024),
        ("voyage-3-lite", 512),
        ("voyage-multilingual-2", 1024),
        ("voyage-finance-2", 1024),
        ("voyage-law-2", 1024),
        ("voyage-large-2-instruct", 1024),
        ("voyage-large-2", 1536),
        ("voyage-code-2", 1536),
    ],
)
def test_resolve_voyage_current_model_default_dimensions(
    model: str,
    expected: int,
) -> None:
    assert (
        resolve_embedding_dimensions(
            provider="voyage",
            model=model,
            dimensions=None,
        )
        == expected
    )


@pytest.mark.parametrize(
    "model",
    [
        "voyage-4",
        "voyage-4-lite",
        "voyage-4-large",
        "voyage-4-nano",
        "voyage-code-3",
        "voyage-3-large",
        "voyage-3.5",
        "voyage-3.5-lite",
    ],
)
@pytest.mark.parametrize("dimensions", [256, 512, 1024, 2048])
def test_resolve_voyage_flexible_model_dimensions(
    model: str,
    dimensions: int,
) -> None:
    assert (
        resolve_embedding_dimensions(
            provider="voyage",
            model=model,
            dimensions=dimensions,
        )
        == dimensions
    )


def test_resolve_voyage_rejects_unsupported_flexible_dimension() -> None:
    with pytest.raises(ValueError, match="not supported for voyage/voyage-code-3"):
        resolve_embedding_dimensions(
            provider="voyage",
            model="voyage-code-3",
            dimensions=1536,
        )


def test_resolve_voyage_rejects_fixed_model_dimension_override() -> None:
    with pytest.raises(ValueError, match="only the default dimension 1536"):
        resolve_embedding_dimensions(
            provider="voyage",
            model="voyage-code-2",
            dimensions=2048,
        )


def test_resolve_cohere_current_model_default_dimensions() -> None:
    assert (
        resolve_embedding_dimensions(
            provider="cohere",
            model="embed-v4.0",
            dimensions=None,
        )
        == 1536
    )


@pytest.mark.parametrize("dimensions", [256, 512, 1024, 1536])
def test_resolve_cohere_flexible_model_dimensions(dimensions: int) -> None:
    assert (
        resolve_embedding_dimensions(
            provider="cohere",
            model="embed-v4.0",
            dimensions=dimensions,
        )
        == dimensions
    )


def test_resolve_cohere_rejects_unsupported_dimension() -> None:
    with pytest.raises(ValueError, match="not supported for cohere/embed-v4.0"):
        resolve_embedding_dimensions(
            provider="cohere",
            model="embed-v4.0",
            dimensions=2048,
        )
