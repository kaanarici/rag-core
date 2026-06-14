from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.documents.prepare_chunks import override_embedding_texts
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import ContextualizeCompleted, ContextualizeStarted

from rag_core.core_models import PreparedChunk

if TYPE_CHECKING:
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.documents.contextualizer_adapters import CachingContextualizer
    from rag_core.events.sink import EventSink
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache


async def apply_contextualizer(
    *,
    chunks: list[PreparedChunk],
    markdown: str,
    filename: str,
    namespace: str = "",
    corpus_id: str = "",
    document_id: str = "",
    contextualizer: "ChunkContextualizer | None",
    chunk_context_cache: "ChunkContextCache | None",
    event_sink: "EventSink | None" = None,
) -> list[PreparedChunk]:
    from rag_core.documents.contextualizer import (
        ChunkContextRequest,
        NoOpContextualizer,
        contextualizer_chunk_cap,
    )

    if not chunks or contextualizer is None or isinstance(contextualizer, NoOpContextualizer):
        return list(chunks)

    resolved = contextualizer
    caching: "CachingContextualizer | None" = None
    if chunk_context_cache is not None:
        from rag_core.documents.contextualizer_adapters import CachingContextualizer
        from rag_core.search.providers.embedding_cache_models import sha256_text

        document_sha256 = sha256_text(markdown)
        caching = CachingContextualizer(
            resolved,
            chunk_context_cache,
            document_sha256_resolver=lambda _: document_sha256,
        )
        resolved = caching

    contextualizer_id = getattr(resolved, "contextualizer_id", type(resolved).__name__)
    total = len(chunks)
    chunk_cap = contextualizer_chunk_cap(contextualizer)
    contextualized_limit = total if chunk_cap is None else min(chunk_cap, total)
    skipped_count = total - contextualized_limit
    emit_event(
        event_sink,
        ContextualizeStarted(
            chunk_count=total,
            model=contextualizer_id,
            contextualized_chunk_count=contextualized_limit,
            skipped_chunk_count=skipped_count,
            chunk_cap=chunk_cap,
        ),
    )
    started_ms = now_ms()
    embedding_texts: list[str] = []
    contextualized_count = 0
    try:
        for position, chunk in enumerate(chunks):
            if position >= contextualized_limit:
                embedding_texts.append(chunk.text)
                continue
            context = await resolved.contextualize(
                ChunkContextRequest(
                    document_markdown=markdown,
                    document_filename=filename,
                    chunk_text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    total_chunks=total,
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=document_id,
                )
            )
            contextualized_count += 1
            embedding_texts.append(
                f"{context}\n\n{chunk.text}" if context else chunk.text
            )
    except Exception:
        emit_event(
            event_sink,
            ContextualizeCompleted(
                chunk_count=total,
                model=contextualizer_id,
                duration_ms=now_ms() - started_ms,
                succeeded=False,
                contextualized_chunk_count=contextualized_count,
                skipped_chunk_count=skipped_count,
                chunk_cap=chunk_cap,
                cache_hits=caching.cache_hits if caching else 0,
                cache_misses=caching.cache_misses if caching else 0,
                cache_writes=caching.cache_writes if caching else 0,
            ),
        )
        raise
    emit_event(
        event_sink,
        ContextualizeCompleted(
            chunk_count=total,
            model=contextualizer_id,
            duration_ms=now_ms() - started_ms,
            succeeded=True,
            contextualized_chunk_count=contextualized_count,
            skipped_chunk_count=skipped_count,
            chunk_cap=chunk_cap,
            cache_hits=caching.cache_hits if caching else 0,
            cache_misses=caching.cache_misses if caching else 0,
            cache_writes=caching.cache_writes if caching else 0,
        ),
    )
    return override_embedding_texts(chunks, embedding_texts)
