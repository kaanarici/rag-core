"""Dense embedding providers with dimension-aware defaults."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

from rag_core.search.providers.registry import EMBEDDING_PROVIDERS
from rag_core.search.types import EmbeddingProvider

from .embedding_models import get_embedding_model_spec, resolve_embedding_dimensions
from .openai_embedding import build_openai_client
from .openai_embedding import embed_openai_texts
from .openai_embedding import fingerprint_provider_config as _fingerprint_provider_config
from .openai_embedding import import_async_openai as _import_async_openai

if TYPE_CHECKING:
    from .voyage import VoyageEmbeddingProvider
    from .zeroentropy import ZeroEntropyEmbeddingProvider

logger = logging.getLogger(__name__)


def _import_voyage_embedding_provider() -> type["VoyageEmbeddingProvider"]:
    from .voyage import VoyageEmbeddingProvider

    return VoyageEmbeddingProvider


def _import_zeroentropy_embedding_provider() -> type["ZeroEntropyEmbeddingProvider"]:
    from .zeroentropy import ZeroEntropyEmbeddingProvider

    return ZeroEntropyEmbeddingProvider


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with optional base URL override."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._provider = "openai"
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


def _build_openai_provider(
    *,
    model: str = "text-embedding-3-large",
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
    model: str = "voyage-4",
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    **_: Any,
) -> "VoyageEmbeddingProvider":
    voyage_cls = _import_voyage_embedding_provider()
    return voyage_cls(
        model=model,
        dimensions=resolve_embedding_dimensions(
            provider="voyage",
            model=model,
            dimensions=dimensions,
        ),
        api_key=api_key,
    )


def _build_demo_provider(
    *,
    model: str = "demo-dense-v1",
    dimensions: int | None = None,
    **_: Any,
) -> EmbeddingProvider:
    from rag_core.demo import DemoEmbeddingProvider, _DEMO_EMBEDDING_DIMENSIONS

    if model != "demo-dense-v1":
        pass  # v1 demo provider uses a fixed model name
    resolved = dimensions if dimensions is not None else _DEMO_EMBEDDING_DIMENSIONS
    return DemoEmbeddingProvider(dimensions=resolved)


def _build_zeroentropy_provider(
    *,
    model: str = "zembed-1",
    dimensions: int | None = None,
    api_key: Optional[str] = None,
    **_: Any,
) -> "ZeroEntropyEmbeddingProvider":
    zeroentropy_cls = _import_zeroentropy_embedding_provider()
    return zeroentropy_cls(
        model=model,
        dimensions=resolve_embedding_dimensions(
            provider="zeroentropy",
            model=model,
            dimensions=dimensions,
        ),
        api_key=api_key,
    )


def create_embedding_provider(
    *,
    provider: str = "openai",
    **kwargs: Any,
) -> EmbeddingProvider:
    return EMBEDDING_PROVIDERS.create(provider or "openai", **kwargs)


EMBEDDING_PROVIDERS.register("openai", _build_openai_provider)
EMBEDDING_PROVIDERS.register("demo", _build_demo_provider)
EMBEDDING_PROVIDERS.register("voyage", _build_voyage_provider)
EMBEDDING_PROVIDERS.register("zeroentropy", _build_zeroentropy_provider)
