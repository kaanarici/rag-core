from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

from rag_core.search.providers.embedding_input_types import EmbeddingInputType
from rag_core.search.providers.embedding_models import get_embedding_model_spec
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

VOYAGE_PROVIDER = "voyage"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4"
DEFAULT_VOYAGE_RERANKER_MODEL = "rerank-2.5-lite"
_VOYAGE_TRANSIENT = TransientErrorSpec(
    module="voyageai.error",
    connection_types=("APIConnectionError", "Timeout"),
    status_types=("APIError",),
    status_attr="http_status",
)

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


class VoyageEmbeddingProvider(HTTPEmbeddingProvider):
    _provider_name = VOYAGE_PROVIDER
    _validation_name = "VoyageEmbeddingProvider"
    _transient_spec = _VOYAGE_TRANSIENT

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

    async def _run_embed(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
        *,
        attempts: int = 3,
    ) -> list[list[float]]:
        async def call_provider() -> list[list[float]]:
            return await asyncio.to_thread(self._embed_sync, texts, input_type)

        return await retry_provider_call(
            call_provider,
            classify=self._classify,
            provider_name=VOYAGE_PROVIDER,
            attempts=attempts,
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


class VoyageReranker(HTTPReranker):
    _provider_name = VOYAGE_PROVIDER
    _validation_name = "VoyageReranker"
    _transient_spec = _VOYAGE_TRANSIENT

    def __init__(
        self,
        *,
        model: str = DEFAULT_VOYAGE_RERANKER_MODEL,
        api_key: str | None = None,
    ) -> None:
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

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
        return await asyncio.to_thread(
            self._client.rerank,
            query,
            documents,
            self._model,
            top_n,
        )

    def _extract_rows(self, response: object) -> list[tuple[object, object]]:
        return rerank_index_score_rows(response)
