from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    provider: str
    model: str
    default_dimensions: int
    max_dimensions: int | None
    supports_dimensions_override: bool
    allowed_dimensions: tuple[int, ...] | None = None


_MODEL_SPECS = {
    ("openai", "text-embedding-3-large"): EmbeddingModelSpec(
        provider="openai",
        model="text-embedding-3-large",
        default_dimensions=3072,
        max_dimensions=3072,
        supports_dimensions_override=True,
    ),
    ("openai", "text-embedding-3-small"): EmbeddingModelSpec(
        provider="openai",
        model="text-embedding-3-small",
        default_dimensions=1536,
        max_dimensions=1536,
        supports_dimensions_override=True,
    ),
    ("openai", "text-embedding-ada-002"): EmbeddingModelSpec(
        provider="openai",
        model="text-embedding-ada-002",
        default_dimensions=1536,
        max_dimensions=1536,
        supports_dimensions_override=False,
    ),
    ("local", "BAAI/bge-small-en-v1.5"): EmbeddingModelSpec(
        provider="local",
        model="BAAI/bge-small-en-v1.5",
        default_dimensions=384,
        max_dimensions=384,
        supports_dimensions_override=False,
    ),
}


def get_embedding_model_spec(provider: str, model: str) -> EmbeddingModelSpec | None:
    key = ((provider or "").strip().lower(), (model or "").strip())
    return _MODEL_SPECS.get(key)


def resolve_embedding_dimensions(
    *,
    provider: str,
    model: str,
    dimensions: int | None,
) -> int:
    normalized_provider = (provider or "").strip().lower()
    if dimensions is not None:
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise ValueError("embedding dimensions must be a positive integer")
        spec = get_embedding_model_spec(normalized_provider, model)
        if (
            spec
            and not spec.supports_dimensions_override
            and dimensions != spec.default_dimensions
        ):
            raise ValueError(
                "embedding dimensions %d are not supported for %s/%s; "
                "only the default dimension %d is supported"
                % (dimensions, normalized_provider, model, spec.default_dimensions)
            )
        if (
            spec
            and spec.allowed_dimensions is not None
            and dimensions not in spec.allowed_dimensions
        ):
            raise ValueError(
                "embedding dimensions %d are not supported for %s/%s; choose one of %s"
                % (
                    dimensions,
                    normalized_provider,
                    model,
                    list(spec.allowed_dimensions),
                )
            )
        if (
            spec
            and spec.max_dimensions is not None
            and dimensions > spec.max_dimensions
        ):
            raise ValueError(
                "embedding dimensions %d exceed max %d for %s/%s"
                % (dimensions, spec.max_dimensions, normalized_provider, model)
            )
        return dimensions

    spec = get_embedding_model_spec(normalized_provider, model)
    if spec is not None:
        return spec.default_dimensions

    msg = "embedding dimensions are required for unknown provider/model pair: %s/%s"
    raise ValueError(msg % (normalized_provider or "unknown", model or "unknown"))
