"""Shared scaffolding for first-party HTTP embedding and reranker adapters.

Cohere, Voyage, and ZeroEntropy share identical retry, health-probe, and
result-validation control flow. Each concrete provider supplies only the
client call, the response row accessor, and a declarative transient-error
spec; everything else lives here so the per-provider modules stay small.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.provider_protocols import ProviderHealth
from rag_core.search.providers.embedding_input_types import (
    EMBEDDING_INPUT_DOCUMENT,
    EMBEDDING_INPUT_QUERY,
    EmbeddingInputType,
)
from rag_core.search.providers.provider_health import (
    PROVIDER_HEALTH_KIND_EMBEDDING,
    PROVIDER_HEALTH_KIND_RERANKER,
    build_healthy_provider_health,
    build_unhealthy_provider_health,
)
from rag_core.search.providers.provider_retry import (
    is_transient_http_status,
    matches_exception_type,
    retry_provider_call,
)
from rag_core.search.providers.rerank_results import safe_indexed_rerank_results
from rag_core.search.request_models import RerankResult


@dataclass(frozen=True)
class TransientErrorSpec:
    """Declarative description of a provider's transient (retryable) errors.

    ``module`` is imported lazily; ``connection_types`` always retry when they
    match the exception; ``status_types`` retry only when the exception's
    ``status_attr`` holds a transient HTTP status. ``status_modules`` allows the
    status-bearing error type to live in a different module than the connection
    types (Cohere keeps ``ApiError`` under both ``cohere.errors`` and
    ``cohere.core``).
    """

    module: str
    connection_types: tuple[str, ...]
    status_types: tuple[str, ...]
    status_attr: str
    status_modules: tuple[str, ...] | None = None


def classify_transient_error(exc: Exception, spec: TransientErrorSpec) -> bool:
    try:
        module = importlib.import_module(spec.module)
    except ImportError:
        return False
    if any(
        matches_exception_type(exc, getattr(module, name, None))
        for name in spec.connection_types
    ):
        return True
    status_modules = spec.status_modules or (spec.module,)
    for module_name in status_modules:
        try:
            status_module = importlib.import_module(module_name)
        except ImportError:
            continue
        if any(
            matches_exception_type(exc, getattr(status_module, name, None))
            for name in spec.status_types
        ):
            return is_transient_http_status(getattr(exc, spec.status_attr, None))
    return False


class HTTPEmbeddingProvider:
    """Embedding adapter base: retry + health probe over a subclass embed hook.

    Subclasses set ``_provider_name``, ``_validation_name``, and
    ``_transient_spec`` and implement :meth:`_run_embed`, which performs the
    real (already retry-wrapped) embedding call for one input type.
    """

    _provider_name: str
    _validation_name: str
    _transient_spec: TransientErrorSpec
    _model: str
    _dimensions: int

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _classify(self, exc: Exception) -> bool:
        return classify_transient_error(exc, self._transient_spec)

    async def _run_embed(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
        *,
        attempts: int = 3,
    ) -> list[list[float]]:
        raise NotImplementedError

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._run_embed(texts, EMBEDDING_INPUT_DOCUMENT)

    async def embed_query(self, query: str) -> list[float]:
        rows = await self._run_embed([query], EMBEDDING_INPUT_QUERY)
        return rows[0]

    async def check_health(self) -> ProviderHealth:
        try:
            await self._run_embed(["health"], EMBEDDING_INPUT_DOCUMENT, attempts=1)
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=self._provider_name,
                kind=PROVIDER_HEALTH_KIND_EMBEDDING,
                model_name=self._model,
                dimensions=self._dimensions,
                exc=exc,
                transient=self._classify(exc),
            )
        return build_healthy_provider_health(
            provider_name=self._provider_name,
            kind=PROVIDER_HEALTH_KIND_EMBEDDING,
            model_name=self._model,
            dimensions=self._dimensions,
        )


class HTTPReranker:
    """Reranker adapter base: retry + health probe over a subclass rerank hook.

    Subclasses set ``_provider_name``, ``_validation_name``, and
    ``_transient_spec`` and implement :meth:`_call_rerank` (the raw client call)
    and :meth:`_extract_rows` (the per-result ``(index, score)`` accessor).
    """

    _provider_name: str
    _validation_name: str
    _transient_spec: TransientErrorSpec
    _model: str

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    def _classify(self, exc: Exception) -> bool:
        return classify_transient_error(exc, self._transient_spec)

    async def _call_rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> object:
        raise NotImplementedError

    def _extract_rows(self, response: object) -> list[tuple[object, object]]:
        raise NotImplementedError

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
            return await self._call_rerank(query, documents, limit)

        response = await retry_provider_call(
            call_provider,
            classify=self._classify,
            provider_name=self._provider_name,
        )
        reranked = safe_indexed_rerank_results(
            rows=self._extract_rows(response),
            documents=documents,
            provider_name=self._validation_name,
        )
        del reranked[limit:]
        return reranked

    async def check_health(self) -> ProviderHealth:
        documents = ["health document", "other document"]

        async def call_provider() -> object:
            return await self._call_rerank("health", documents, 1)

        try:
            response = await retry_provider_call(
                call_provider,
                classify=self._classify,
                provider_name=self._provider_name,
                attempts=1,
            )
            safe_indexed_rerank_results(
                rows=self._extract_rows(response),
                documents=documents,
                provider_name=self._validation_name,
            )
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=self._provider_name,
                kind=PROVIDER_HEALTH_KIND_RERANKER,
                model_name=self._model,
                exc=exc,
                transient=self._classify(exc),
            )
        return build_healthy_provider_health(
            provider_name=self._provider_name,
            kind=PROVIDER_HEALTH_KIND_RERANKER,
            model_name=self._model,
        )


def rerank_index_score_rows(
    response: object,
    *,
    results_attr: str = "results",
    index_attr: str = "index",
    score_attr: str = "relevance_score",
) -> list[tuple[object, object]]:
    results = getattr(response, results_attr, None) or []
    return [
        (getattr(row, index_attr, None), getattr(row, score_attr, None))
        for row in results
    ]


__all__ = [
    "HTTPEmbeddingProvider",
    "HTTPReranker",
    "TransientErrorSpec",
    "classify_transient_error",
    "rerank_index_score_rows",
]
