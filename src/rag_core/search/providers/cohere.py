from __future__ import annotations

from typing import Any

from rag_core.search.providers.rerank_results import safe_indexed_rerank_results
from rag_core.search.types import RerankResult


def _import_cohere() -> Any:
    try:
        import cohere
    except ImportError as exc:
        raise ImportError(
            "cohere package is required for CohereReranker. Install it with: pip install cohere"
        ) from exc
    if cohere is None or not hasattr(cohere, "AsyncClientV2"):
        raise ImportError("cohere package with AsyncClientV2 is required for CohereReranker.")
    return cohere


class CohereReranker:
    """Cohere rerank-v3.5 provider."""

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: str | None = None,
    ) -> None:
        cohere = _import_cohere()
        self._model = model
        self._client = cohere.AsyncClientV2(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "cohere"

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
        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=limit,
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
