"""Local BM25 sparse embedding via FastEmbed (no API calls)."""

from __future__ import annotations

import logging
import threading
from typing import Any, Final, Literal, TypeAlias

from rag_core.config.env_access import get_env as config_get_env
from rag_core.search.providers.registry import SPARSE_EMBEDDERS
from rag_core.search.sparse_channels import (
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
    single_sparse_channel,
)
from rag_core.search.provider_protocols import SparseEmbedder
from rag_core.search.vector_models import SparseVector

logger = logging.getLogger(__name__)

DEFAULT_SPARSE_EMBEDDER_PROVIDER = "fastembed"
SPARSE_EMBEDDER_PROVIDER_ORDER = (DEFAULT_SPARSE_EMBEDDER_PROVIDER,)
SparseLoadStatus: TypeAlias = Literal[
    "not_loaded",
    "disabled",
    "loaded",
    "load_failed",
    "not_checked_by_doctor",
    "unknown_until_sparse_embedding_runs",
]
SPARSE_LOAD_NOT_LOADED: Final[SparseLoadStatus] = "not_loaded"
SPARSE_LOAD_DISABLED: Final[SparseLoadStatus] = "disabled"
SPARSE_LOAD_LOADED: Final[SparseLoadStatus] = "loaded"
SPARSE_LOAD_FAILED: Final[SparseLoadStatus] = "load_failed"
SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR: Final[SparseLoadStatus] = "not_checked_by_doctor"
SPLADE_LOAD_UNKNOWN_UNTIL_RUN: Final[SparseLoadStatus] = (
    "unknown_until_sparse_embedding_runs"
)


def _model_env(name: str, default: str) -> str:
    value = config_get_env(name, default)
    if value is None:
        return default
    return value


_DEFAULT_BM25_MODEL_NAME = "Qdrant/bm25"
_DEFAULT_SPLADE_MODEL_NAME = "prithivida/Splade_PP_en_v1"


def _default_bm25_model_name() -> str:
    return _model_env(
        "SPARSE_EMBEDDING_MODEL_BM25",
        _model_env("SPARSE_EMBEDDING_MODEL", _DEFAULT_BM25_MODEL_NAME),
    )


def _default_splade_model_name() -> str:
    return _model_env(
        "SPARSE_EMBEDDING_MODEL_SPLADE",
        _DEFAULT_SPLADE_MODEL_NAME,
    )


class _SparseCardinalityError(ValueError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            "FastEmbedSparseEmbedder provider contract violation: "
            "expected %d sparse vectors, got %d" % (expected, actual)
        )


class FastEmbedSparseEmbedder:
    """Sparse embedder using FastEmbed with bm25 + optional SPLADE channels."""

    provider_name = DEFAULT_SPARSE_EMBEDDER_PROVIDER

    def __init__(
        self,
        bm25_model_name: str | None = None,
        splade_model_name: str | None = None,
        *,
        enable_splade: bool = True,
    ) -> None:
        bm25_model_name = bm25_model_name or _default_bm25_model_name()
        splade_model_name = splade_model_name or _default_splade_model_name()
        sparse_text_embedding = _import_sparse_text_embedding()
        self._bm25_model = sparse_text_embedding(bm25_model_name)
        self._bm25_model_name = bm25_model_name
        self._splade_model_name = splade_model_name
        self._splade_enabled = enable_splade
        self._splade_model: Any | None = None
        self._splade_load_status = (
            SPARSE_LOAD_NOT_LOADED if enable_splade else SPARSE_LOAD_DISABLED
        )
        self._lock = threading.Lock()

    def _embed_with_model(self, model: Any, texts: list[str]) -> list[SparseVector]:
        with self._lock:
            raw_results = list(model.embed(texts))
        if len(raw_results) != len(texts):
            raise _SparseCardinalityError(expected=len(texts), actual=len(raw_results))
        return [
            SparseVector(
                indices=list(result.indices),
                values=list(result.values),
            )
            for result in raw_results
        ]

    def _ensure_splade_model(self) -> Any | None:
        if not self._splade_enabled:
            self._splade_load_status = SPARSE_LOAD_DISABLED
            return None
        if self._splade_model is not None:
            self._splade_load_status = SPARSE_LOAD_LOADED
            return self._splade_model
        try:
            sparse_text_embedding = _import_sparse_text_embedding()
            self._splade_model = sparse_text_embedding(self._splade_model_name)
            self._splade_load_status = SPARSE_LOAD_LOADED
            logger.info(
                "Loaded sparse model: provider=%s channel=%s",
                DEFAULT_SPARSE_EMBEDDER_PROVIDER,
                SECONDARY_SPARSE_CHANNEL,
            )
            return self._splade_model
        except Exception as exc:
            self._splade_enabled = False
            self._splade_load_status = SPARSE_LOAD_FAILED
            logger.warning(
                "Failed to load sparse model; using fallback: "
                "provider=%s channel=%s fallback_channel=%s error_type=%s",
                DEFAULT_SPARSE_EMBEDDER_PROVIDER,
                SECONDARY_SPARSE_CHANNEL,
                PRIMARY_SPARSE_CHANNEL,
                type(exc).__name__,
            )
            return None

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Embed multiple texts as sparse BM25 vectors."""
        return self._embed_with_model(self._bm25_model, texts)

    def embed_texts_multi(self, texts: list[str]) -> list[dict[str, SparseVector]]:
        """Embed texts into multiple sparse channels (bm25 + splade)."""
        bm25_vectors = self.embed_texts(texts)
        merged = [single_sparse_channel(vector) for vector in bm25_vectors]

        splade_model = self._ensure_splade_model()
        if splade_model is None:
            return merged

        try:
            splade_vectors = self._embed_with_model(splade_model, texts)
        except _SparseCardinalityError as exc:
            logger.warning(
                "Sparse vector count mismatch; using fallback: "
                "provider=%s channel=%s fallback_channel=%s "
                "expected=%d actual=%d error_type=%s",
                DEFAULT_SPARSE_EMBEDDER_PROVIDER,
                SECONDARY_SPARSE_CHANNEL,
                PRIMARY_SPARSE_CHANNEL,
                exc.expected,
                exc.actual,
                type(exc).__name__,
            )
            return merged

        for idx, vector in enumerate(splade_vectors):
            merged[idx][SECONDARY_SPARSE_CHANNEL] = vector
        return merged

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": DEFAULT_SPARSE_EMBEDDER_PROVIDER,
            "bm25_enabled": True,
            "splade_enabled": self._splade_enabled,
            "splade_load_status": self._splade_load_status,
        }

    def embed_query(self, query: str) -> SparseVector:
        """Embed a single query as a sparse BM25 vector."""
        return self.embed_texts([query])[0]

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]:
        """Embed a single query into all available sparse channels."""
        return self.embed_texts_multi([query])[0]


def _build_fastembed_sparse_embedder(**kwargs: Any) -> FastEmbedSparseEmbedder:
    return FastEmbedSparseEmbedder(**kwargs)


def _import_sparse_text_embedding() -> Any:
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise ImportError(
            f"fastembed is required for sparse provider "
            f"{DEFAULT_SPARSE_EMBEDDER_PROVIDER!r}"
        ) from exc
    return SparseTextEmbedding


def create_sparse_embedder(
    *,
    provider: str = DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    **kwargs: Any,
) -> SparseEmbedder:
    return SPARSE_EMBEDDERS.create(
        provider or DEFAULT_SPARSE_EMBEDDER_PROVIDER,
        **kwargs,
    )


SPARSE_EMBEDDERS.register(
    DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    _build_fastembed_sparse_embedder,
)
