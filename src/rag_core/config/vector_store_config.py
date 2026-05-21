from __future__ import annotations

from dataclasses import dataclass

DEFAULT_VECTOR_STORE_PROVIDER = "qdrant"
SUPPORTED_VECTOR_STORE_PROVIDERS = ("qdrant",)


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str = DEFAULT_VECTOR_STORE_PROVIDER

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("VectorStoreConfig.provider must be a non-empty string")
        provider = self.provider.strip().lower()
        if provider not in SUPPORTED_VECTOR_STORE_PROVIDERS:
            known = ", ".join(SUPPORTED_VECTOR_STORE_PROVIDERS)
            raise ValueError(
                f"VectorStoreConfig.provider must be one of: {known}"
            )
        object.__setattr__(self, "provider", provider)
