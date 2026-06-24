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


_VOYAGE_FLEXIBLE_DIMENSIONS = (256, 512, 1024, 2048)
_COHERE_FLEXIBLE_DIMENSIONS = (256, 512, 1024, 1536)


def _voyage_flexible_model(model: str) -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        provider="voyage",
        model=model,
        default_dimensions=1024,
        max_dimensions=max(_VOYAGE_FLEXIBLE_DIMENSIONS),
        allowed_dimensions=_VOYAGE_FLEXIBLE_DIMENSIONS,
        supports_dimensions_override=True,
    )


def _voyage_fixed_model(model: str, dimensions: int) -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        provider="voyage",
        model=model,
        default_dimensions=dimensions,
        max_dimensions=dimensions,
        supports_dimensions_override=False,
    )


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
    ("cohere", "embed-v4.0"): EmbeddingModelSpec(
        provider="cohere",
        model="embed-v4.0",
        default_dimensions=1536,
        max_dimensions=max(_COHERE_FLEXIBLE_DIMENSIONS),
        allowed_dimensions=_COHERE_FLEXIBLE_DIMENSIONS,
        supports_dimensions_override=True,
    ),
    ("voyage", "voyage-4-lite"): _voyage_flexible_model("voyage-4-lite"),
    ("voyage", "voyage-4"): _voyage_flexible_model("voyage-4"),
    ("voyage", "voyage-4-large"): _voyage_flexible_model("voyage-4-large"),
    ("voyage", "voyage-code-3"): _voyage_flexible_model("voyage-code-3"),
    ("voyage", "voyage-3-large"): _voyage_flexible_model("voyage-3-large"),
    ("voyage", "voyage-3.5"): _voyage_flexible_model("voyage-3.5"),
    ("voyage", "voyage-3.5-lite"): _voyage_flexible_model("voyage-3.5-lite"),
    ("voyage", "voyage-4-nano"): _voyage_flexible_model("voyage-4-nano"),
    ("voyage", "voyage-3"): _voyage_fixed_model("voyage-3", 1024),
    ("voyage", "voyage-3-lite"): _voyage_fixed_model("voyage-3-lite", 512),
    ("voyage", "voyage-multilingual-2"): _voyage_fixed_model(
        "voyage-multilingual-2",
        1024,
    ),
    ("voyage", "voyage-finance-2"): _voyage_fixed_model("voyage-finance-2", 1024),
    ("voyage", "voyage-law-2"): _voyage_fixed_model("voyage-law-2", 1024),
    ("voyage", "voyage-large-2-instruct"): _voyage_fixed_model(
        "voyage-large-2-instruct",
        1024,
    ),
    ("voyage", "voyage-large-2"): _voyage_fixed_model("voyage-large-2", 1536),
    ("voyage", "voyage-code-2"): _voyage_fixed_model("voyage-code-2", 1536),
    ("zeroentropy", "zembed-1"): EmbeddingModelSpec(
        provider="zeroentropy",
        model="zembed-1",
        default_dimensions=2560,
        max_dimensions=2560,
        allowed_dimensions=(40, 80, 160, 320, 640, 1280, 2560),
        supports_dimensions_override=True,
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
