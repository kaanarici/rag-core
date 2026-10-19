"""Local BM25 sparse embeddings; TurboPuffer uses its own full-text index."""

from __future__ import annotations
import threading
from typing import Any
from os import getenv
from rag_core.search.provider_protocols import SparseEmbedder
from rag_core.search.vector_models import SparseVector

DEFAULT_SPARSE_EMBEDDER_PROVIDER = "fastembed"
SPARSE_LOAD_NOT_LOADED = "not_loaded"
SPARSE_LOAD_LOADED = "loaded"
SPARSE_LOAD_FAILED = "load_failed"


class _SparseCardinalityError(ValueError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            "FastEmbedSparseEmbedder provider contract violation: "
            "expected %d sparse vectors, got %d" % (expected, actual)
        )


class FastEmbedSparseEmbedder:
    provider_name = DEFAULT_SPARSE_EMBEDDER_PROVIDER

    def __init__(self, bm25_model_name: str | None = None) -> None:
        self._bm25_model_name = (
            bm25_model_name
            or getenv(
                "SPARSE_EMBEDDING_MODEL_BM25",
                getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25"),
            )
            or "Qdrant/bm25"
        )
        self._bm25_model: Any = None
        self._bm25_load_status = SPARSE_LOAD_NOT_LOADED
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._bm25_model_name

    def _embed(self, texts: list[str], *, query: bool) -> list[SparseVector]:
        with self._lock:
            if self._bm25_model is None:
                try:
                    self._bm25_model = _import_sparse_text_embedding()(
                        self._bm25_model_name
                    )
                except Exception:
                    self._bm25_load_status = SPARSE_LOAD_FAILED
                    raise
                self._bm25_load_status = SPARSE_LOAD_LOADED
            method = self._bm25_model.query_embed if query else self._bm25_model.embed
            raw = list(method(texts))
        if len(raw) != len(texts):
            raise _SparseCardinalityError(expected=len(texts), actual=len(raw))
        return [
            SparseVector(indices=list(result.indices), values=list(result.values))
            for result in raw
        ]

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        return self._embed(texts, query=False)

    def embed_query(self, query: str) -> SparseVector:
        return self._embed([query], query=True)[0]

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "bm25_enabled": True,
            "bm25_load_status": self._bm25_load_status,
        }


def _import_sparse_text_embedding() -> Any:
    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        raise ImportError(
            "fastembed is required for sparse provider 'fastembed'"
        ) from None
    return SparseTextEmbedding


def create_sparse_embedder(
    *, provider: str = DEFAULT_SPARSE_EMBEDDER_PROVIDER, **kwargs: Any
) -> SparseEmbedder:
    if provider != DEFAULT_SPARSE_EMBEDDER_PROVIDER:
        raise ValueError(
            "Only fastembed is built in; inject a SparseEmbedder into Engine"
        )
    return FastEmbedSparseEmbedder(**kwargs)
