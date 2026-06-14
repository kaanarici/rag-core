from __future__ import annotations

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

COHERE_PROVIDER = "cohere"
DEFAULT_COHERE_EMBEDDING_MODEL = "embed-v4.0"
DEFAULT_COHERE_RERANKER_MODEL = "rerank-v3.5"
_EMBED_BATCH_SIZE = 96
_COHERE_EMBEDDING_FLOAT_TYPE = "float"
_COHERE_INPUT_BY_TYPE = {
    EMBEDDING_INPUT_DOCUMENT: "search_document",
    EMBEDDING_INPUT_QUERY: "search_query",
}

if TYPE_CHECKING:
    import types


def _import_cohere() -> "types.ModuleType":
    try:
        cohere = importlib.import_module("cohere")
    except ImportError as exc:
        raise ImportError(
            "cohere package is required for Cohere providers. "
            "Install it with: pip install 'rag-core[rerank]'"
        ) from exc
    if not callable(getattr(cohere, "AsyncClientV2", None)):
        raise ImportError(
            "cohere package with AsyncClientV2 is required for Cohere providers."
        )
    return cohere


class CohereEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_COHERE_EMBEDDING_MODEL,
        dimensions: int = 1536,
        api_key: str | None = None,
    ) -> None:
        cohere = _import_cohere()
        self._model = model
        self._dimensions = dimensions
        self._client = cohere.AsyncClientV2(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return COHERE_PROVIDER

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, EMBEDDING_INPUT_DOCUMENT)

    async def embed_query(self, query: str) -> list[float]:
        rows = await self._embed([query], EMBEDDING_INPUT_QUERY)
        return rows[0]

    async def check_health(self) -> ProviderHealth:
        try:
            await self._embed_batch_with_retry(
                ["health"],
                EMBEDDING_INPUT_DOCUMENT,
                attempts=1,
            )
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=COHERE_PROVIDER,
                kind=PROVIDER_HEALTH_KIND_EMBEDDING,
                model_name=self._model,
                dimensions=self._dimensions,
                exc=exc,
                transient=_is_transient_cohere_error(exc),
            )
        return build_healthy_provider_health(
            provider_name=COHERE_PROVIDER,
            kind=PROVIDER_HEALTH_KIND_EMBEDDING,
            model_name=self._model,
            dimensions=self._dimensions,
        )

    async def _embed(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(
                await self._embed_batch_with_retry(batch, input_type)
            )
        return vectors

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
        *,
        attempts: int = 3,
    ) -> list[list[float]]:
        async def call_provider() -> list[list[float]]:
            return await self._embed_batch(texts, input_type)

        return await retry_provider_call(
            call_provider,
            classify=_is_transient_cohere_error,
            provider_name=COHERE_PROVIDER,
            attempts=attempts,
        )

    async def _embed_batch(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        response = await self._client.embed(
            model=self._model,
            input_type=_COHERE_INPUT_BY_TYPE[input_type],
            texts=texts,
            output_dimension=self._dimensions,
            embedding_types=[_COHERE_EMBEDDING_FLOAT_TYPE],
        )
        return safe_ordered_embedding_vectors(
            rows=list(_float_embeddings(response)),
            expected_count=len(texts),
            expected_dimensions=self._dimensions,
            provider_name="CohereEmbeddingProvider",
        )


class CohereReranker:
    """Cohere reranker provider."""

    def __init__(
        self,
        model: str = DEFAULT_COHERE_RERANKER_MODEL,
        api_key: str | None = None,
    ) -> None:
        cohere = _import_cohere()
        self._model = model
        self._client = cohere.AsyncClientV2(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return COHERE_PROVIDER

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
            return await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=limit,
            )

        response = await retry_provider_call(
            call_provider,
            classify=_is_transient_cohere_error,
            provider_name=COHERE_PROVIDER,
        )
        reranked = safe_indexed_rerank_results(
            rows=[
                (getattr(result, "index", None), getattr(result, "relevance_score", None))
                for result in getattr(response, "results", []) or []
            ],
            documents=documents,
            provider_name="CohereReranker",
        )
        del reranked[limit:]
        return reranked

    async def check_health(self) -> ProviderHealth:
        documents = ["health document", "other document"]

        async def call_provider() -> object:
            return await self._client.rerank(
                model=self._model,
                query="health",
                documents=documents,
                top_n=1,
            )

        try:
            response = await retry_provider_call(
                call_provider,
                classify=_is_transient_cohere_error,
                provider_name=COHERE_PROVIDER,
                attempts=1,
            )
            safe_indexed_rerank_results(
                rows=[
                    (getattr(result, "index", None), getattr(result, "relevance_score", None))
                    for result in getattr(response, "results", []) or []
                ],
                documents=documents,
                provider_name="CohereReranker",
            )
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=COHERE_PROVIDER,
                kind=PROVIDER_HEALTH_KIND_RERANKER,
                model_name=self._model,
                exc=exc,
                transient=_is_transient_cohere_error(exc),
            )
        return build_healthy_provider_health(
            provider_name=COHERE_PROVIDER,
            kind=PROVIDER_HEALTH_KIND_RERANKER,
            model_name=self._model,
        )


def _float_embeddings(response: object) -> list[object]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        return []
    rows = getattr(embeddings, "float_", None) or getattr(embeddings, "float", None)
    return rows if isinstance(rows, list) else []


def _is_transient_cohere_error(exc: Exception) -> bool:
    try:
        errors = importlib.import_module("cohere.errors")
    except ImportError:
        return False
    if any(
        matches_exception_type(exc, getattr(errors, name, None))
        for name in (
            "TooManyRequestsError",
            "InternalServerError",
            "ServiceUnavailableError",
            "GatewayTimeoutError",
        )
    ):
        return True
    for module_name in ("cohere.errors", "cohere.core"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if matches_exception_type(exc, getattr(module, "ApiError", None)):
            return is_transient_http_status(getattr(exc, "status_code", None))
    return False
