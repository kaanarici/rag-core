from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import cast

from rag_core.search.providers.embedding_results import safe_ordered_embedding_vectors
from rag_core.search.providers.embedding_models import get_embedding_model_spec
from rag_core.search.providers.rerank_results import safe_indexed_rerank_results
from rag_core.search.types import RerankResult

if TYPE_CHECKING:
    import types


def _import_voyageai() -> "types.ModuleType":
    try:
        import voyageai  # type: ignore[import-not-found]
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
    return cast("types.ModuleType", voyageai)


def _supports_output_dimension(model: str) -> bool:
    spec = get_embedding_model_spec("voyage", model)
    return bool(spec.supports_dimensions_override) if spec is not None else False


class VoyageEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "voyage-4",
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
        return "voyage"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts, "document")

    async def embed_query(self, query: str) -> list[float]:
        rows = await asyncio.to_thread(self._embed_sync, [query], "query")
        return rows[0]

    def _embed_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
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
        model: str = "rerank-2.5-lite",
        api_key: str | None = None,
    ) -> None:
        voyageai = _import_voyageai()
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "voyage"

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        limit = min(top_k, len(documents))
        response = await asyncio.to_thread(
            self._client.rerank,
            query,
            documents,
            self._model,
            limit,
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
