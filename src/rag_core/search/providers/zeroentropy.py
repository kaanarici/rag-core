from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

from rag_core.search.providers.embedding_results import safe_ordered_embedding_vectors
from rag_core.search.providers.rerank_results import safe_indexed_rerank_results
from rag_core.search.types import RerankResult

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


class ZeroEntropyEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "zembed-1",
        dimensions: int = 2560,
        api_key: str | None = None,
    ) -> None:
        zeroentropy = _import_zeroentropy()
        self._client = zeroentropy.ZeroEntropy(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "zeroentropy"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts, "document")

    async def embed_query(self, query: str) -> list[float]:
        rows = await asyncio.to_thread(self._embed_sync, [query], "query")
        return rows[0]

    def _embed_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
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


class ZeroEntropyReranker:
    def __init__(
        self,
        *,
        model: str = "zerank-2",
        api_key: str | None = None,
    ) -> None:
        zeroentropy = _import_zeroentropy()
        self._client = zeroentropy.ZeroEntropy(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "zeroentropy"

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        if not documents:
            return []
        bounded_top_k = min(top_k, len(documents))
        if bounded_top_k <= 0:
            return []
        response = await asyncio.to_thread(
            self._client.models.rerank,
            model=self._model,
            query=query,
            documents=documents,
            top_n=bounded_top_k,
        )
        reranked = safe_indexed_rerank_results(
            rows=[
                (getattr(row, "index", None), getattr(row, "relevance_score", None))
                for row in getattr(response, "results", None) or []
            ],
            documents=documents,
            provider_name="ZeroEntropyReranker",
        )
        del reranked[bounded_top_k:]
        return reranked
