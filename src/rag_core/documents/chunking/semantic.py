"""Semantic chunking using embedding similarity to find topic boundaries."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, cast

from rag_core.config.env_access import get_env as config_get_env, get_env_bool
from rag_core.core_models import PreparedChunk

from .protocol import ChunkConfig
from .semantic_chunk_builder import (
    build_chunks_from_segments,
    paragraph_heuristic_chunks,
    single_semantic_chunk,
)
from .semantic_segments import (
    segments_from_semantic_boundaries,
    split_sentences,
)

logger = logging.getLogger(__name__)

EmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


class _SentenceEmbeddingModel(Protocol):
    def encode(self, sentences: List[str], *, show_progress_bar: bool) -> Any: ...


class _LocalSemanticEmbedder:
    """Singleton local sentence embedder."""

    _instances: Dict[str, "_LocalSemanticEmbedder"] = {}

    @classmethod
    def get(cls, model_name: str) -> "_LocalSemanticEmbedder":
        instance = cls._instances.get(model_name)
        if instance is None:
            instance = cls(model_name)
            cls._instances[model_name] = instance
        return instance

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: _SentenceEmbeddingModel | None = None

    def _load_model(self) -> _SentenceEmbeddingModel:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        self._model = cast(_SentenceEmbeddingModel, SentenceTransformer(self._model_name))
        return self._model

    async def embed_many(self, sentences: List[str]) -> List[List[float]]:
        if not sentences:
            return []

        def _encode() -> List[List[float]]:
            model = self._load_model()
            vectors = model.encode(sentences, show_progress_bar=False)
            return [list(map(float, vector)) for vector in vectors]

        return await asyncio.to_thread(_encode)


class SemanticChunker:
    """Chunks text by finding semantic boundaries using embedding similarity."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.75,
        embed_fn: Optional[EmbedFn] = None,
        enable_local_model: Optional[bool] = None,
        local_model_name: Optional[str] = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._embed_fn = embed_fn
        self._enable_local_model = (
            get_env_bool("CHUNKING_ENABLE_LOCAL_SEMANTIC", False)
            if enable_local_model is None
            else enable_local_model
        )
        configured_model = (
            config_get_env("CHUNKING_SEMANTIC_LOCAL_MODEL", "") or ""
        ).strip()
        self._local_model_name = (
            local_model_name or configured_model or "sentence-transformers/all-MiniLM-L6-v2"
        )

    def _get_local_embed_fn(self) -> Optional[EmbedFn]:
        if not self._enable_local_model:
            return None

        try:
            embedder = _LocalSemanticEmbedder.get(self._local_model_name)
        except Exception as exc:
            logger.warning(
                "Local semantic embedder setup failed for model %r; using heuristic "
                "fallback (error_type=%s)",
                self._local_model_name,
                type(exc).__name__,
            )
            return None

        async def _embed(sentences: List[str]) -> List[List[float]]:
            return await embedder.embed_many(sentences)

        return _embed

    def _resolve_embed_fn(self) -> Optional[EmbedFn]:
        if self._embed_fn is not None:
            return self._embed_fn
        return self._get_local_embed_fn()

    def chunk(self, text: str, config: ChunkConfig) -> List[PreparedChunk]:
        """Synchronous path using heuristic boundaries only."""
        if not text:
            return []

        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [single_semantic_chunk(text)]

        return paragraph_heuristic_chunks(text, sentences, config)

    async def chunk_async(self, text: str, config: ChunkConfig) -> List[PreparedChunk]:
        """Async semantic chunking with embedding-based boundaries."""
        if not text:
            return []

        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [single_semantic_chunk(text)]

        using_local_model = self._embed_fn is None and self._enable_local_model
        embed_fn = self._resolve_embed_fn()
        if embed_fn is None:
            return paragraph_heuristic_chunks(text, sentences, config)

        try:
            embeddings = await embed_fn(sentences)
        except Exception as exc:
            if using_local_model:
                logger.warning(
                    "Semantic embedding failed for model %r; using heuristic fallback "
                    "(error_type=%s)",
                    self._local_model_name,
                    type(exc).__name__,
                )
            else:
                logger.warning(
                    "Semantic embedding failed, using heuristic fallback (error_type=%s)",
                    type(exc).__name__,
                )
            return paragraph_heuristic_chunks(text, sentences, config)

        if len(embeddings) != len(sentences):
            logger.warning(
                "Semantic embedding length mismatch (%d != %d), using heuristic fallback",
                len(embeddings),
                len(sentences),
            )
            return paragraph_heuristic_chunks(text, sentences, config)

        segments = segments_from_semantic_boundaries(
            sentences,
            embeddings,
            similarity_threshold=self._threshold,
        )
        return build_chunks_from_segments(
            text,
            segments,
            config,
            strategy_name="semantic",
        )
