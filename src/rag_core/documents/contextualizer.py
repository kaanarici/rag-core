"""Pluggable per-chunk contextualizer seam.

The contextualizer enriches a chunk with a short context paragraph before it
is embedded or BM25-indexed. The canonical shape is per-chunk so that cache
keys are stable; batching is an internal concern of any given adapter.

The cache wrapper (see :mod:`rag_core.search.providers.embedding_cache`) is
paired with the contextualizer so reindexing an unchanged document re-pays
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_core.documents.contextualizer_provider_names import NOOP_CONTEXTUALIZER_ID


@dataclass(frozen=True)
class ChunkContextRequest:
    """One contextualization unit.

    The full document text is passed because most quality recipes (Anthropic,
    Voyage) need it to situate the chunk; adapters may truncate.
    """

    document_markdown: str
    document_filename: str
    chunk_text: str
    chunk_index: int
    total_chunks: int


@runtime_checkable
class ChunkContextualizer(Protocol):
    """Produce a short context string for a chunk.

    The returned string is prepended to ``chunk.text`` to form the embedding
    text. An empty string means "no contextualization for this chunk".

    Implementations may batch internally, but the per-chunk shape is the
    canonical seam used for cache lookup. Production paths route through the
    cache wrapper paired with this protocol.
    """

    @property
    def contextualizer_id(self) -> str:
        """Stable identity used as part of processing and cache keys."""

    async def contextualize(self, request: ChunkContextRequest) -> str: ...


class NoOpContextualizer:
    """Default contextualizer that returns an empty context for every chunk.

    Selected when contextualization is disabled. Costs nothing and is a no-op
    even when wrapped in a cache.
    """

    contextualizer_id: str = NOOP_CONTEXTUALIZER_ID

    async def contextualize(self, request: ChunkContextRequest) -> str:
        return ""


__all__ = [
    "ChunkContextRequest",
    "ChunkContextualizer",
    "NOOP_CONTEXTUALIZER_ID",
    "NoOpContextualizer",
]
