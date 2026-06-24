"""Built-in adapters for :mod:`rag_core.documents.contextualizer`.

* ``AnthropicChunkContextualizer`` implements the published per-chunk recipe
  with prompt caching, requiring ``anthropic`` only at instantiation time.
* ``CachingContextualizer`` pairs a contextualizer with a cache so a reindex
  of an unchanged document re-pays nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .contextualizer import (
    ChunkContextRequest,
    ChunkContextualizer,
    validate_contextualizer_chunk_cap,
)
from .contextualizer_anthropic_runtime import (
    _ANTHROPIC_CONTEXT_PROMPT_VERSION,
    build_context_messages,
    create_anthropic_client,
    extract_anthropic_text,
)
from .contextualizer_provider_names import (
    ANTHROPIC_CONTEXTUALIZER_ID,
    DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL,
)

if TYPE_CHECKING:
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache

logger = logging.getLogger(__name__)


class AnthropicChunkContextualizer:
    """Per-chunk contextualizer using Anthropic Messages with prompt caching.

    Lazy-imports ``anthropic``; raises :class:`ImportError` if the SDK is not
    installed when an instance is created. The document text is marked with
    ``cache_control={"type": "ephemeral"}`` so subsequent chunks reuse the
    same cached prefix.
    """

    contextualizer_id_prefix = ANTHROPIC_CONTEXTUALIZER_ID

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_CONTEXTUALIZER_MODEL,
        *,
        max_tokens: int = 200,
        chunk_cap: int | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._chunk_cap = validate_contextualizer_chunk_cap(chunk_cap)
        self._last_call_failed = False
        if client is not None:
            self._client = client
            return
        self._client = create_anthropic_client(api_key=api_key)

    @property
    def contextualizer_id(self) -> str:
        return (
            f"{self.contextualizer_id_prefix}:{self._model}:"
            f"{_ANTHROPIC_CONTEXT_PROMPT_VERSION}:max_tokens={self._max_tokens}"
        )

    @property
    def contextualizer_chunk_cap(self) -> int | None:
        return self._chunk_cap

    async def contextualize(self, request: ChunkContextRequest) -> str:
        messages = build_context_messages(request)
        if messages is None:
            self._last_call_failed = False
            return ""
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
            )
        except Exception as exc:
            self._last_call_failed = True
            logger.debug(
                "Anthropic contextualization failed for model %s chunk %d with %s",
                self._model,
                request.chunk_index,
                type(exc).__name__,
            )
            raise
        self._last_call_failed = False
        return extract_anthropic_text(response).strip()

    def should_cache_context(self, context: str) -> bool:
        if self._last_call_failed and context == "":
            return False
        return True


class CachingContextualizer:
    """Wrap a contextualizer with a :class:`ChunkContextCache`.

    Misses are forwarded to the inner contextualizer and the result is stored
    in the cache. Hits short-circuit the inner call entirely.

    Hit/miss/write counts accumulate on a per-instance ``CacheCounters``
    snapshot so the engine can stamp them onto ``ContextualizeCompleted``
    without re-reading the cache state.
    """

    def __init__(
        self,
        inner: ChunkContextualizer,
        cache: "ChunkContextCache",
        *,
        document_sha256_resolver: Any | None = None,
    ) -> None:
        from rag_core.search.providers.embedding_cache_models import sha256_text

        self._inner = inner
        self._cache = cache
        self._document_sha256 = document_sha256_resolver or sha256_text
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_writes = 0

    @property
    def contextualizer_id(self) -> str:
        return self._inner.contextualizer_id

    @property
    def contextualizer_chunk_cap(self) -> int | None:
        return validate_contextualizer_chunk_cap(
            getattr(self._inner, "contextualizer_chunk_cap", None)
        )

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_writes(self) -> int:
        return self._cache_writes

    async def contextualize(self, request: ChunkContextRequest) -> str:
        from rag_core.search.providers.chunk_context_cache import ChunkContextKey
        from rag_core.search.providers.embedding_cache_models import sha256_text

        document_sha = self._document_sha256(request.document_markdown)
        key = ChunkContextKey(
            contextualizer_id=self._inner.contextualizer_id,
            document_sha256=document_sha,
            document_filename_sha256=sha256_text(request.document_filename),
            chunk_text_sha256=sha256_text(request.chunk_text),
            chunk_index=request.chunk_index,
            total_chunks=request.total_chunks,
            namespace=request.namespace,
            collection=request.collection,
            document_id=request.document_id,
        )
        cached = await self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        context = await self._inner.contextualize(request)
        if _should_cache_context(self._inner, context):
            await self._cache.put(key, context)
            self._cache_writes += 1
        return context


def _should_cache_context(inner: ChunkContextualizer, context: str) -> bool:
    should_cache = getattr(inner, "should_cache_context", None)
    if callable(should_cache):
        try:
            return bool(should_cache(context))
        except Exception:
            return context != ""
    return True

__all__ = [
    "ANTHROPIC_CONTEXTUALIZER_ID",
    "AnthropicChunkContextualizer",
    "CachingContextualizer",
]
