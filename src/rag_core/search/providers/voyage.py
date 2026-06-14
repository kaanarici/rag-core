from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.provider_protocols import ProviderHealth
from rag_core.search.providers.embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
    EmbeddingInputType,
)
from rag_core.search.providers.embedding_results import safe_ordered_embedding_vectors
from rag_core.search.providers.embedding_models import get_embedding_model_spec
from rag_core.search.providers.provider_health import (
    PROVIDER_HEALTH_KIND_EMBEDDING,
    PROVIDER_HEALTH_KIND_RERANKER,
    build_healthy_provider_health,
    build_unhealthy_provider_health,
)
from rag_core.search.providers.provider_retry import is_transient_http_status
from rag_core.search.providers.provider_retry import matches_exception_type
from rag_core.search.providers.provider_retry import retry_provider_call
from rag_core.search.providers.rerank_results import safe_indexed_rerank_results
from rag_core.search.request_models import RerankResult

VOYAGE_PROVIDER = "voyage"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4"
DEFAULT_VOYAGE_RERANKER_MODEL = "rerank-2.5-lite"

if TYPE_CHECKING:
    import types


def _import_voyageai() -> "types.ModuleType":
    try:
        voyageai = importlib.import_module("voyageai")
    except ImportError as exc:
        raise ImportError(
            "voyageai package is required for Voyage providers. "
            "Install the rag-core extra with: pip install 'rag-core[voyage]'"
        ) from exc
    if not callable(getattr(voyageai, "Client", None)):
        raise ImportError(
            "voyageai package is installed but incompatible: missing callable "
            "voyageai.Client constructor. Install the rag-core extra with: "
            "pip install 'rag-core[voyage]'"
        )
    return voyageai


def _supports_output_dimension(model: str) -> bool:
    spec = get_embedding_model_spec(VOYAGE_PROVIDER, model)
    return bool(spec.supports_dimensions_override) if spec is not None else False


class VoyageEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL,
        dimensions: int = 1024,
        api_key: str | None = None,
    ) -> None:
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._send_dimensions = _supports_output_dimension(model)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return VOYAGE_PROVIDER

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        async def call_provider() -> list[list[float]]:
            return await asyncio.to_thread(
                self._embed_sync,
                texts,
                EMBEDDING_INPUT_DOCUMENT,
            )

        return await retry_provider_call(
            call_provider,
            classify=_is_transient_voyage_error,
            provider_name=VOYAGE_PROVIDER,
        )

    async def embed_query(self, query: str) -> list[float]:
        async def call_provider() -> list[list[float]]:
            return await asyncio.to_thread(
                self._embed_sync,
                [query],
                EMBEDDING_INPUT_QUERY,
            )

        rows = await retry_provider_call(
            call_provider,
            classify=_is_transient_voyage_error,
            provider_name=VOYAGE_PROVIDER,
        )
        return rows[0]

    async def check_health(self) -> ProviderHealth:
        async def call_provider() -> list[list[float]]:
            return await asyncio.to_thread(
                self._embed_sync,
                ["health"],
                EMBEDDING_INPUT_DOCUMENT,
            )

        try:
            await retry_provider_call(
                call_provider,
                classify=_is_transient_voyage_error,
                provider_name=VOYAGE_PROVIDER,
                attempts=1,
            )
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=VOYAGE_PROVIDER,
                kind=PROVIDER_HEALTH_KIND_EMBEDDING,
                model_name=self._model,
                dimensions=self._dimensions,
                exc=exc,
                transient=_is_transient_voyage_error(exc),
            )
        return build_healthy_provider_health(
            provider_name=VOYAGE_PROVIDER,
            kind=PROVIDER_HEALTH_KIND_EMBEDDING,
            model_name=self._model,
            dimensions=self._dimensions,
        )

    def _embed_sync(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        kwargs: dict[str, object] = {
            "model": self._model,
            "input_type": input_type,
        }
        if self._send_dimensions:
            kwargs["output_dimension"] = self._dimensions
        response = self._client.embed(texts, **kwargs)
        return safe_ordered_embedding_vectors(
            rows=list(getattr(response, "embeddings", []) or []),
            expected_count=len(texts),
            expected_dimensions=self._dimensions,
            provider_name="VoyageEmbeddingProvider",
        )


class VoyageReranker:
    def __init__(
        self,
        *,
        model: str = DEFAULT_VOYAGE_RERANKER_MODEL,
        api_key: str | None = None,
    ) -> None:
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return VOYAGE_PROVIDER

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        limit = min(top_k, len(documents))
        async def call_provider() -> object:
            return await asyncio.to_thread(
                self._client.rerank,
                query,
                documents,
                self._model,
                limit,
            )

        response = await retry_provider_call(
            call_provider,
            classify=_is_transient_voyage_error,
            provider_name=VOYAGE_PROVIDER,
        )
        reranked = safe_indexed_rerank_results(
            rows=[
                (getattr(row, "index", None), getattr(row, "relevance_score", None))
                for row in getattr(response, "results", []) or []
            ],
            documents=documents,
            provider_name="VoyageReranker",
        )
        del reranked[limit:]
        return reranked

    async def check_health(self) -> ProviderHealth:
        documents = ["health document", "other document"]

        async def call_provider() -> object:
            return await asyncio.to_thread(
                self._client.rerank,
                "health",
                documents,
                self._model,
                1,
            )

        try:
            response = await retry_provider_call(
                call_provider,
                classify=_is_transient_voyage_error,
                provider_name=VOYAGE_PROVIDER,
                attempts=1,
            )
            safe_indexed_rerank_results(
                rows=[
                    (getattr(row, "index", None), getattr(row, "relevance_score", None))
                    for row in getattr(response, "results", []) or []
                ],
                documents=documents,
                provider_name="VoyageReranker",
            )
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=VOYAGE_PROVIDER,
                kind=PROVIDER_HEALTH_KIND_RERANKER,
                model_name=self._model,
                exc=exc,
                transient=_is_transient_voyage_error(exc),
            )
        return build_healthy_provider_health(
            provider_name=VOYAGE_PROVIDER,
            kind=PROVIDER_HEALTH_KIND_RERANKER,
            model_name=self._model,
        )


def _is_transient_voyage_error(exc: Exception) -> bool:
    try:
        errors = importlib.import_module("voyageai.error")
    except ImportError:
        return False
    if matches_exception_type(exc, getattr(errors, "APIConnectionError", None)):
        return True
    if matches_exception_type(exc, getattr(errors, "Timeout", None)):
        return True
    if matches_exception_type(exc, getattr(errors, "APIError", None)):
        return is_transient_http_status(getattr(exc, "http_status", None))
    return False
