from __future__ import annotations

from rag_core.search.providers.registry import RERANKER_PROVIDERS
from rag_core.search.providers.reranker import (
    create_reranker,
    resolve_reranker_provider,
)
from rag_core.search.types import RerankResult


class _CustomReranker:
    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or "custom-rerank"
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "custom-reranker"

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        return [
            RerankResult(index=index, score=1.0, text=document)
            for index, document in enumerate(documents[:top_k])
        ]


def test_registered_custom_reranker_resolves_without_builtin_api_key_policy() -> None:
    name = "custom-reranker-contract"
    RERANKER_PROVIDERS.register(name, lambda **kwargs: _CustomReranker(**kwargs))
    try:
        assert resolve_reranker_provider(name) == (name, None)

        reranker = create_reranker(provider=f" {name.upper()} ", model="rerank-x")

        assert isinstance(reranker, _CustomReranker)
        assert reranker.model_name == "rerank-x"
        assert getattr(reranker, "_rag_core_provider_requested") == name
        assert getattr(reranker, "_rag_core_provider_effective") == name
        assert getattr(reranker, "_rag_core_fallback_reason") is None
    finally:
        RERANKER_PROVIDERS.unregister(name)
