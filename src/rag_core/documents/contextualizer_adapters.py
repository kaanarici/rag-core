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
)
from .contextualizer_anthropic_runtime import (
    _ANTHROPIC_CONTEXT_PROMPT_VERSION,
    _DEFAULT_ANTHROPIC_MODEL,
    build_context_messages,
    create_anthropic_client,
    extract_anthropic_text,
)

if TYPE_CHECKING:
    from rag_core.search.providers.embedding_cache import ChunkContextCache

logger = logging.getLogger(__name__)


class AnthropicChunkContextualizer:
    """Per-chunk contextualizer using Anthropic Messages with prompt caching.

    Lazy-imports ``anthropic``; raises :class:`ImportError` if the SDK is not
    installed when an instance is created. The document text is marked with
    ``cache_control={"type": "ephemeral"}`` so subsequent chunks reuse the
    same cached prefix.
    """

    contextualizer_id_prefix = "anthropic"

    def __init__(
        self,
        model: str = _DEFAULT_ANTHROPIC_MODEL,
        *,
        max_tokens: int = 200,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
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
    """

    def __init__(
        self,
        inner: ChunkContextualizer,
        cache: "ChunkContextCache",
        *,
        document_sha256_resolver: Any | None = None,
    ) -> None:
        from rag_core.search.providers.embedding_cache import sha256_text

        self._inner = inner
        self._cache = cache
        self._document_sha256 = document_sha256_resolver or sha256_text

    @property
    def contextualizer_id(self) -> str:
        return self._inner.contextualizer_id

    async def contextualize(self, request: ChunkContextRequest) -> str:
        from rag_core.search.providers.embedding_cache import ChunkContextKey, sha256_text

        document_sha = self._document_sha256(request.document_markdown)
        key = ChunkContextKey(
            contextualizer_id=self._inner.contextualizer_id,
            document_sha256=document_sha,
            document_filename_sha256=sha256_text(request.document_filename),
            chunk_text_sha256=sha256_text(request.chunk_text),
            chunk_index=request.chunk_index,
            total_chunks=request.total_chunks,
        )
        cached = await self._cache.get(key)
        if cached is not None:
            return cached
        context = await self._inner.contextualize(request)
        if _should_cache_context(self._inner, context):
            await self._cache.put(key, context)
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
    "AnthropicChunkContextualizer",
    "CachingContextualizer",
]
