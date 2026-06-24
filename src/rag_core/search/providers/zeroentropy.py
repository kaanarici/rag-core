from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

from rag_core.search.providers.embedding_input_types import EmbeddingInputType
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

ZEROENTROPY_PROVIDER = "zeroentropy"
DEFAULT_ZEROENTROPY_EMBEDDING_MODEL = "zembed-1"
DEFAULT_ZEROENTROPY_RERANKER_MODEL = "zerank-2"
_ZEROENTROPY_TRANSIENT = TransientErrorSpec(
    module="zeroentropy",
    connection_types=("APIConnectionError", "APITimeoutError"),
    status_types=("APIStatusError",),
    status_attr="status_code",
)

if TYPE_CHECKING:
    import types


def _import_zeroentropy() -> "types.ModuleType":
    try:
        module = importlib.import_module("zeroentropy")
    except ImportError as exc:
        raise ImportError(
            "zeroentropy package is required for ZeroEntropy providers. "
            "Install the rag-core extra with: pip install 'rag-core[zeroentropy]'"
        ) from exc
    if not callable(getattr(module, "ZeroEntropy", None)):
        raise ImportError(
            "zeroentropy package is installed but incompatible: missing callable "
            "zeroentropy.ZeroEntropy constructor. Install the rag-core extra with: "
            "pip install 'rag-core[zeroentropy]'"
        )
    return module


class ZeroEntropyEmbeddingProvider(HTTPEmbeddingProvider):
    _provider_name = ZEROENTROPY_PROVIDER
    _validation_name = "ZeroEntropyEmbeddingProvider"
    _transient_spec = _ZEROENTROPY_TRANSIENT

    def __init__(
        self,
        *,
        model: str = DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
        dimensions: int = 2560,
        api_key: str | None = None,
    ) -> None:
        zeroentropy = _import_zeroentropy()
        self._client = zeroentropy.ZeroEntropy(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

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
            provider_name=ZEROENTROPY_PROVIDER,
            attempts=attempts,
        )

    def _embed_sync(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        response = self._client.models.embed(
            model=self._model,
            input_type=input_type,
            input=texts,
            dimensions=self._dimensions,
        )
        results = getattr(response, "results", None) or []
        return safe_ordered_embedding_vectors(
            rows=[getattr(row, "embedding", None) for row in results],
            expected_count=len(texts),
            expected_dimensions=self._dimensions,
            provider_name="ZeroEntropyEmbeddingProvider",
        )


class ZeroEntropyReranker(HTTPReranker):
    _provider_name = ZEROENTROPY_PROVIDER
    _validation_name = "ZeroEntropyReranker"
    _transient_spec = _ZEROENTROPY_TRANSIENT

    def __init__(
        self,
        *,
        model: str = DEFAULT_ZEROENTROPY_RERANKER_MODEL,
        api_key: str | None = None,
    ) -> None:
        zeroentropy = _import_zeroentropy()
        self._client = zeroentropy.ZeroEntropy(api_key=api_key)
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
            self._client.models.rerank,
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

    def _extract_rows(self, response: object) -> list[tuple[object, object]]:
        return rerank_index_score_rows(response)
