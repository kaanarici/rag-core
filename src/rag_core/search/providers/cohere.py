from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from rag_core.search.providers.embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
    EmbeddingInputType,
)
from rag_core.search.providers.embedding_results import safe_ordered_embedding_vectors
from rag_core.search.providers.http_provider_base import (
    HTTPEmbeddingProvider,
    HTTPReranker,
    TransientErrorSpec,
    rerank_index_score_rows,
)
from rag_core.search.providers.provider_retry import retry_provider_call
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
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
_COHERE_TRANSIENT = TransientErrorSpec(
    module="cohere.errors",
    connection_types=(
        "TooManyRequestsError",
        "InternalServerError",
        "ServiceUnavailableError",
        "GatewayTimeoutError",
    ),
    status_types=("ApiError",),
    status_attr="status_code",
    status_modules=("cohere.errors", "cohere.core"),
)

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


class CohereEmbeddingProvider(HTTPEmbeddingProvider):
    _provider_name = COHERE_PROVIDER
    _validation_name = "CohereEmbeddingProvider"
    _transient_spec = _COHERE_TRANSIENT

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

    async def _run_embed(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
        *,
        attempts: int = 3,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(
                await self._embed_batch_with_retry(batch, input_type, attempts=attempts)
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
            classify=self._classify,
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


class CohereReranker(HTTPReranker):
    """Cohere reranker provider."""

    _provider_name = COHERE_PROVIDER
    _validation_name = "CohereReranker"
    _transient_spec = _COHERE_TRANSIENT

    def __init__(
        self,
        model: str = DEFAULT_COHERE_RERANKER_MODEL,
        api_key: str | None = None,
    ) -> None:
        cohere = _import_cohere()
        self._model = model
        self._client = cohere.AsyncClientV2(api_key=api_key)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RerankResult]:
        return await super().rerank(query, documents, top_k)

    async def _call_rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> object:
        return await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

    def _extract_rows(self, response: object) -> list[tuple[object, object]]:
        return rerank_index_score_rows(response)


def _float_embeddings(response: object) -> list[object]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        return []
    rows = getattr(embeddings, "float_", None) or getattr(embeddings, "float", None)
    return rows if isinstance(rows, list) else []
