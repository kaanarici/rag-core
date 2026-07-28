from __future__ import annotations

from dataclasses import dataclass

from rag_core.fetch_security import FetchSecurityPolicy, validate_fetch_url

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEMO_EMBEDDING_PROVIDER = "demo"
LOCAL_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEMO_EMBEDDING_MODEL = "demo-dense-v1"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_EMBEDDING_DIMENSIONS = 384
DEFAULT_EMBEDDING_BATCH_SIZE = 50
EMBEDDING_BATCH_SIZE_ENV = "RAG_CORE_EMBEDDING_BATCH_SIZE"
EMBEDDING_DIMENSIONS_ENV = "RAG_CORE_EMBEDDING_DIMENSIONS"
EMBEDDING_MODEL_ENV = "RAG_CORE_EMBEDDING_MODEL"
EMBEDDING_PROVIDER_ENV = "RAG_CORE_EMBEDDING_PROVIDER"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedder configuration.

    ``base_url`` is validated at construction via ``validate_fetch_url`` so
    misconfiguration is rejected before the engine starts: http://, embedded
    credentials, and (without explicit opt-in) private-IP literals all raise
    ``ValueError`` here, not at the first embed call.

    ``region`` is an optional host-substring pin for the endpoint host (e.g.
    ``"us-east-1"``, ``"eu"``). When set, ``base_url``'s resolved host must
    contain the pin string. The check is intentionally a stable substring
    match: each operator is the source of truth for their own host naming
    convention, and the assembly seam refuses to construct an embedder whose
    base_url host does not satisfy the pin.
    """

    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    region: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("EmbeddingConfig.provider must be non-empty")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions <= 0
        ):
            raise ValueError("EmbeddingConfig.dimensions must be a positive integer")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("EmbeddingConfig.batch_size must be a positive integer")
        if self.base_url is not None:
            if not isinstance(self.base_url, str):
                raise ValueError("EmbeddingConfig.base_url must be a string")
            stripped = self.base_url.strip()
            if not stripped:
                # Treat empty/whitespace as absent rather than silently storing it.
                object.__setattr__(self, "base_url", None)
            else:
                try:
                    validated = validate_fetch_url(
                        stripped, policy=FetchSecurityPolicy()
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"EmbeddingConfig.base_url is not a safe https URL: {exc}"
                    ) from None
                object.__setattr__(self, "base_url", stripped)
                if self.region is not None:
                    region = self.region.strip()
                    if not region:
                        raise ValueError(
                            "EmbeddingConfig.region must be a non-empty string"
                        )
                    object.__setattr__(self, "region", region)
                    if region not in validated.host:
                        raise ValueError(
                            "EmbeddingConfig.base_url host does not match "
                            f"region pin {region!r}"
                        )
        elif self.region is not None:
            region = self.region.strip()
            if not region:
                raise ValueError(
                    "EmbeddingConfig.region must be a non-empty string"
                )
            # Region pin without a base_url is pointless and likely a config
            # bug; fail closed rather than silently accept it.
            raise ValueError(
                "EmbeddingConfig.region requires base_url to be set"
            )
