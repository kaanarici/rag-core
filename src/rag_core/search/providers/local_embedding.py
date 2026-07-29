"""No-key dense embeddings via FastEmbed ONNX models."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from typing import Any, Callable, cast

from rag_core.config import (
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
)
from rag_core.search.provider_protocols import ProviderHealth
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions
from rag_core.search.providers.provider_health import (
    PROVIDER_HEALTH_KIND_EMBEDDING,
    build_healthy_provider_health,
    build_unhealthy_provider_health,
)
from rag_core.search.providers.provider_retry import retry_provider_call

TextEmbeddingLoader = Callable[[], type[Any]]


class LocalEmbeddingProvider:
    provider_name = LOCAL_EMBEDDING_PROVIDER

    def __init__(
        self,
        *,
        model: str = LOCAL_EMBEDDING_MODEL,
        dimensions: int | None = None,
        text_embedding_loader: TextEmbeddingLoader | None = None,
    ) -> None:
        self._model_name = model
        self._dimensions = resolve_embedding_dimensions(
            provider=LOCAL_EMBEDDING_PROVIDER,
            model=model,
            dimensions=dimensions,
        )
        self._text_embedding_loader = text_embedding_loader or _import_text_embedding
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_texts_sync, list(texts))

    async def embed_query(self, query: str) -> list[float]:
        vectors = await asyncio.to_thread(self._embed_query_sync, query)
        return vectors[0]

    async def check_health(self) -> ProviderHealth:
        async def load_model() -> object:
            return await asyncio.to_thread(self._ensure_model)

        try:
            await retry_provider_call(
                load_model,
                classify=lambda _exc: False,
                provider_name=LOCAL_EMBEDDING_PROVIDER,
                attempts=1,
            )
        except Exception as exc:
            return build_unhealthy_provider_health(
                provider_name=LOCAL_EMBEDDING_PROVIDER,
                kind=PROVIDER_HEALTH_KIND_EMBEDDING,
                model_name=self._model_name,
                dimensions=self._dimensions,
                exc=exc,
                transient=False,
            )
        return build_healthy_provider_health(
            provider_name=LOCAL_EMBEDDING_PROVIDER,
            kind=PROVIDER_HEALTH_KIND_EMBEDDING,
            model_name=self._model_name,
            dimensions=self._dimensions,
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        text_embedding = self._text_embedding_loader()
        self._model = text_embedding(
            model_name=self._model_name,
            cuda=False,
        )
        return self._model

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        return self._run_fastembed(
            lambda model: model.passage_embed(texts),
            len(texts),
        )

    def _embed_query_sync(self, query: str) -> list[list[float]]:
        return self._run_fastembed(lambda model: model.query_embed([query]), 1)

    def _run_fastembed(
        self,
        embed: Callable[[Any], object],
        expected_count: int,
    ) -> list[list[float]]:
        try:
            with self._lock:
                raw_vectors = list(
                    cast(Iterable[object], embed(self._ensure_model()))
                )
        except Exception as exc:
            raise RuntimeError(_download_error(self._model_name)) from exc
        vectors = [_vector_to_float_list(vector) for vector in raw_vectors]
        if len(vectors) != expected_count:
            raise ValueError(
                "LocalEmbeddingProvider provider contract violation: expected "
                f"{expected_count} vectors, got {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise ValueError(
                    "LocalEmbeddingProvider provider contract violation: expected "
                    f"{self._dimensions} dimensions, got {len(vector)}"
                )
        return vectors


def _vector_to_float_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        raw_values = vector.tolist()
    else:
        raw_values = list(cast(Iterable[object], vector))
    return [float(value) for value in raw_values]


def _download_error(model_name: str) -> str:
    return (
        "FastEmbed local embedding model failed to load: "
        f"{model_name!r}. The local provider downloads this model once on first use; "
        "check network/cache access, or use provider='demo' for offline/CI smoke."
    )


def _import_text_embedding() -> type[Any]:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            "fastembed is required for local embedding provider "
            f"{LOCAL_EMBEDDING_PROVIDER!r}"
        ) from exc
    return TextEmbedding


__all__ = ["LocalEmbeddingProvider"]
