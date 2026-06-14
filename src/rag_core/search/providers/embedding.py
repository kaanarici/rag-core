"""Dense embedding providers with dimension-aware defaults."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

from rag_core.config.embedding_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEMO_EMBEDDING_MODEL,
    DEMO_EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
)
from rag_core.search.provider_protocols import EmbeddingProvider, ProviderHealth
from rag_core.search.providers.registry import EMBEDDING_PROVIDERS

from .embedding_models import get_embedding_model_spec, resolve_embedding_dimensions
from .cohere import COHERE_PROVIDER, DEFAULT_COHERE_EMBEDDING_MODEL
from .openai_embedding import build_openai_client
from .openai_embedding import check_openai_embedding_health
from .openai_embedding import embed_openai_texts
from .openai_embedding import fingerprint_provider_config as _fingerprint_provider_config
from .openai_embedding import import_async_openai as _import_async_openai
from .voyage import DEFAULT_VOYAGE_EMBEDDING_MODEL, VOYAGE_PROVIDER
from .zeroentropy import DEFAULT_ZEROENTROPY_EMBEDDING_MODEL, ZEROENTROPY_PROVIDER

if TYPE_CHECKING:
    from .cohere import CohereEmbeddingProvider
    from .local_embedding import LocalEmbeddingProvider
    from .voyage import VoyageEmbeddingProvider
    from .zeroentropy import ZeroEntropyEmbeddingProvider

logger = logging.getLogger(__name__)


def _import_voyage_embedding_provider() -> type["VoyageEmbeddingProvider"]:
    from .voyage import VoyageEmbeddingProvider

    return VoyageEmbeddingProvider


def _import_cohere_embedding_provider() -> type["CohereEmbeddingProvider"]:
    from .cohere import CohereEmbeddingProvider

    return CohereEmbeddingProvider


def _import_zeroentropy_embedding_provider() -> type["ZeroEntropyEmbeddingProvider"]:
    from .zeroentropy import ZeroEntropyEmbeddingProvider

    return ZeroEntropyEmbeddingProvider


def _import_local_embedding_provider() -> type["LocalEmbeddingProvider"]:
    from .local_embedding import LocalEmbeddingProvider

    return LocalEmbeddingProvider


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with optional base URL override."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._provider = DEFAULT_EMBEDDING_PROVIDER
        self._model = model
        self._cache_identity = _fingerprint_provider_config(base_url=base_url)
        spec = get_embedding_model_spec(self._provider, model)
        self._dimensions = resolve_embedding_dimensions(
            provider=self._provider,
            model=model,
            dimensions=dimensions,
        )
        self._send_dimensions = (
            bool(spec.supports_dimensions_override) if spec is not None else True
        )
        self._client = build_openai_client(
            _import_async_openai,
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def cache_identity(self) -> str:
        return self._cache_identity

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, batching to avoid API limits."""
        return await embed_openai_texts(
            self._client,
            model=self._model,
            dimensions=self._dimensions,
            send_dimensions=self._send_dimensions,
            texts=texts,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]

    async def check_health(self) -> ProviderHealth:
        return await check_openai_embedding_health(
            self._client,
            model=self._model,
            dimensions=self._dimensions,
            send_dimensions=self._send_dimensions,
        )


def _build_openai_provider(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        model=model,
        dimensions=dimensions,
        api_key=api_key,
        base_url=base_url,
    )


def _build_voyage_provider(
    *,
    model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL,
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    **_: Any,
) -> "VoyageEmbeddingProvider":
    voyage_cls = _import_voyage_embedding_provider()
    return voyage_cls(
        model=model,
        dimensions=resolve_embedding_dimensions(
            provider=VOYAGE_PROVIDER,
            model=model,
            dimensions=dimensions,
        ),
        api_key=api_key,
    )


def _build_cohere_provider(
    *,
    model: str = DEFAULT_COHERE_EMBEDDING_MODEL,
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    **_: Any,
) -> "CohereEmbeddingProvider":
    cohere_cls = _import_cohere_embedding_provider()
    return cohere_cls(
        model=model,
        dimensions=resolve_embedding_dimensions(
            provider=COHERE_PROVIDER,
            model=model,
            dimensions=dimensions,
        ),
        api_key=api_key,
    )


def _build_demo_provider(
    *,
    model: str = DEMO_EMBEDDING_MODEL,
    dimensions: int | None = None,
    **_: Any,
) -> EmbeddingProvider:
    from rag_core.demo import DemoEmbeddingProvider, _DEMO_EMBEDDING_DIMENSIONS

    if model != DEMO_EMBEDDING_MODEL:
        raise ValueError(
            f"demo embedding provider only supports model {DEMO_EMBEDDING_MODEL!r}; "
            f"got {model!r}"
        )
    resolved = dimensions if dimensions is not None else _DEMO_EMBEDDING_DIMENSIONS
    return DemoEmbeddingProvider(dimensions=resolved)


def _build_local_provider(
    *,
    model: str = LOCAL_EMBEDDING_MODEL,
    dimensions: int | None = None,
    **_: Any,
) -> "LocalEmbeddingProvider":
    local_cls = _import_local_embedding_provider()
    return local_cls(
        model=model,
        dimensions=dimensions,
    )


def _build_zeroentropy_provider(
    *,
    model: str = DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    **_: Any,
) -> "ZeroEntropyEmbeddingProvider":
    zeroentropy_cls = _import_zeroentropy_embedding_provider()
    return zeroentropy_cls(
        model=model,
        dimensions=resolve_embedding_dimensions(
            provider=ZEROENTROPY_PROVIDER,
            model=model,
            dimensions=dimensions,
        ),
        api_key=api_key,
    )


def create_embedding_provider(
    *,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    **kwargs: Any,
) -> EmbeddingProvider:
    return EMBEDDING_PROVIDERS.create(
        provider or DEFAULT_EMBEDDING_PROVIDER,
        **kwargs,
    )


EMBEDDING_PROVIDERS.register(DEFAULT_EMBEDDING_PROVIDER, _build_openai_provider)
EMBEDDING_PROVIDERS.register(DEMO_EMBEDDING_PROVIDER, _build_demo_provider)
EMBEDDING_PROVIDERS.register(LOCAL_EMBEDDING_PROVIDER, _build_local_provider)
EMBEDDING_PROVIDERS.register(COHERE_PROVIDER, _build_cohere_provider)
EMBEDDING_PROVIDERS.register(VOYAGE_PROVIDER, _build_voyage_provider)
EMBEDDING_PROVIDERS.register(ZEROENTROPY_PROVIDER, _build_zeroentropy_provider)
