from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.config import TurboPufferVectorStoreConfig, VectorStoreConfig


def test_vector_store_config_normalizes_provider() -> None:
    assert VectorStoreConfig(provider=" TurboPuffer ").provider == "turbopuffer"


@pytest.mark.parametrize("provider", ["", "   ", None, True, cast(Any, 123)])
def test_vector_store_config_rejects_invalid_provider(provider: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        VectorStoreConfig(provider=cast(Any, provider))

    assert str(exc_info.value) == "VectorStoreConfig.provider must be a non-empty string"


def test_vector_store_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError) as exc_info:
        VectorStoreConfig(provider="unknown")

    assert str(exc_info.value) == (
        "VectorStoreConfig.provider must be one of: qdrant, turbopuffer"
    )


def test_turbopuffer_vector_store_config_normalizes_values() -> None:
    config = TurboPufferVectorStoreConfig(
        namespace=" docs ",
        api_key=" key ",
        region=" aws-us-west-2 ",
        base_url=" https://example.invalid ",
        distance_metric=" cosine_distance ",
        delete_continuation_limit=25,
    )

    assert config.namespace == "docs"
    assert config.api_key == "key"
    assert config.region == "aws-us-west-2"
    assert config.base_url == "https://example.invalid"
    assert config.distance_metric == "cosine_distance"
    assert config.delete_continuation_limit == 25


@pytest.mark.parametrize(
    "namespace",
    [
        pytest.param("has/slash", id="slash"),
        pytest.param("has space", id="space"),
        pytest.param("has:colon", id="colon"),
        pytest.param("x" * 129, id="too-long"),
    ],
)
def test_turbopuffer_vector_store_config_rejects_invalid_namespace(
    namespace: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        TurboPufferVectorStoreConfig(namespace=namespace)

    assert str(exc_info.value) == (
        "TurboPufferVectorStoreConfig.namespace must match "
        "[A-Za-z0-9-_.]{1,128}"
    )


@pytest.mark.parametrize(
    "field_name",
    ["namespace", "api_key", "region", "base_url"],
)
def test_turbopuffer_vector_store_config_rejects_non_string_optional_values(
    field_name: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        TurboPufferVectorStoreConfig(**{field_name: cast(Any, True)})

    assert str(exc_info.value) == (
        f"TurboPufferVectorStoreConfig.{field_name} must be a string"
    )


@pytest.mark.parametrize("distance_metric", ["", "   ", None, True, cast(Any, 123)])
def test_turbopuffer_vector_store_config_rejects_invalid_distance_metric(
    distance_metric: object,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        TurboPufferVectorStoreConfig(distance_metric=cast(Any, distance_metric))

    assert str(exc_info.value) == (
        "TurboPufferVectorStoreConfig.distance_metric must be a non-empty string"
    )


def test_turbopuffer_vector_store_config_rejects_unknown_distance_metric() -> None:
    with pytest.raises(ValueError) as exc_info:
        TurboPufferVectorStoreConfig(distance_metric="dot_product")

    assert str(exc_info.value) == (
        "TurboPufferVectorStoreConfig.distance_metric must be one of: "
        "cosine_distance, euclidean_squared"
    )


@pytest.mark.parametrize("limit", [0, -1, None, True, "10"])
def test_turbopuffer_vector_store_config_rejects_invalid_delete_continuation_limit(
    limit: object,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        TurboPufferVectorStoreConfig(delete_continuation_limit=cast(Any, limit))

    assert str(exc_info.value) == (
        "TurboPufferVectorStoreConfig.delete_continuation_limit must be positive"
    )
