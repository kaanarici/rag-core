"""Dense embedding providers with dimension-aware defaults."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
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

from .embedding_models import get_embedding_model_spec, resolve_embedding_dimensions
from .openai_embedding import build_openai_client
from .openai_embedding import check_openai_embedding_health
from .openai_embedding import embed_openai_texts
from .openai_embedding import import_async_openai as _import_async_openai


logger = logging.getLogger(__name__)


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


def create_embedding_provider(
    *,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model: str | None = None,
    dimensions: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingProvider:
    provider = provider.strip().lower()
    if provider == DEFAULT_EMBEDDING_PROVIDER:
        return OpenAIEmbeddingProvider(
            model=model or DEFAULT_EMBEDDING_MODEL,
            dimensions=dimensions,
            api_key=api_key,
            base_url=base_url,
        )
    if provider == LOCAL_EMBEDDING_PROVIDER:
        from .local_embedding import LocalEmbeddingProvider

        return LocalEmbeddingProvider(
            model=model or LOCAL_EMBEDDING_MODEL, dimensions=dimensions
        )
    if provider == DEMO_EMBEDDING_PROVIDER:
        return _build_demo_provider(
            model=model or DEMO_EMBEDDING_MODEL, dimensions=dimensions
        )
    raise ValueError(
        "Unknown embedding provider; use local, openai, demo, or inject one into Engine"
    )
