"""Pluggable reranking providers."""

from __future__ import annotations

import logging
from typing import Any, cast

from rag_core.config.env_access import get_env_bool
from rag_core.search.providers.cohere import CohereReranker
from rag_core.search.providers.reranker_resolution import (
    _SanitizedRerankerInitError,
    attach_runtime_metadata as _attach_runtime_metadata,
    resolve_reranker_provider,
)
from rag_core.search.providers.registry import RERANKER_PROVIDERS
from rag_core.search.types import RerankResult, RerankerProvider

logger = logging.getLogger(__name__)


class NoOpReranker:
    """Passthrough reranker that returns results in original order."""

    @property
    def provider_name(self) -> str:
        return "none"

    @property
    def model_name(self) -> str:
        return "none"

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        return [
            RerankResult(index=i, score=1.0 - (i * 0.01), text=doc)
            for i, doc in enumerate(documents[:top_k])
        ]


def _import_voyage_reranker() -> Any:
    from .voyage import VoyageReranker

    return VoyageReranker


def _import_zeroentropy_reranker() -> Any:
    from .zeroentropy import ZeroEntropyReranker

    return ZeroEntropyReranker


def _build_noop_reranker(**_: Any) -> NoOpReranker:
    return NoOpReranker()


def _build_cohere_reranker(
    *,
    model: str | None = None,
    api_key: str | None = None,
    **_: Any,
) -> CohereReranker:
    return CohereReranker(model=model or "rerank-v3.5", api_key=api_key)


def _build_voyage_reranker(
    *,
    model: str | None = None,
    api_key: str | None = None,
    **_: Any,
) -> RerankerProvider:
    voyage_cls = _import_voyage_reranker()
    return cast(
        RerankerProvider,
        voyage_cls(model=model or "rerank-2.5-lite", api_key=api_key),
    )


def _build_zeroentropy_reranker(
    *,
    model: str | None = None,
    api_key: str | None = None,
    **_: Any,
) -> RerankerProvider:
    zeroentropy_cls = _import_zeroentropy_reranker()
    return cast(
        RerankerProvider,
        zeroentropy_cls(model=model or "zerank-2", api_key=api_key),
    )


def create_reranker(
    provider: str = "none",
    model: str | None = None,
    api_key: str | None = None,
) -> RerankerProvider:
    """Factory function for creating reranker instances.

    Two concerns layered: policy (resolve_reranker_provider handles
    missing-API-key fallback) and lookup (the registry instantiates).
    """

    requested = (provider or "none").strip().lower()
    strict = get_env_bool("RERANKER_STRICT_PROVIDER", False)
    effective, fallback_reason = resolve_reranker_provider(requested, api_key=api_key)
    if effective == "invalid":
        raise ValueError(f"Unknown reranker provider: {requested}")

    if fallback_reason:
        message = (
            f"Reranker provider '{requested}' unavailable ({fallback_reason}); "
            "falling back to no-op reranker."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    try:
        reranker = RERANKER_PROVIDERS.create(effective, model=model, api_key=api_key)
    except Exception as exc:
        error_type = type(exc).__name__
        raise ValueError(
            "Failed to initialize reranker provider '%s' (error_type=%s)"
            % (effective, error_type)
        ) from _SanitizedRerankerInitError(
            provider=effective,
            error_type=error_type,
        )

    return _attach_runtime_metadata(
        reranker,
        requested=requested,
        effective=effective,
        fallback_reason=fallback_reason,
    )


RERANKER_PROVIDERS.register("none", _build_noop_reranker)
RERANKER_PROVIDERS.register("cohere", _build_cohere_reranker)
RERANKER_PROVIDERS.register("voyage", _build_voyage_reranker)
RERANKER_PROVIDERS.register("zeroentropy", _build_zeroentropy_reranker)
