from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core._engine.core_prepare_chunks import override_embedding_texts
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import ContextualizeCompleted, ContextualizeStarted

from rag_core.core_models import PreparedChunk

if TYPE_CHECKING:
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.events.sink import EventSink
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache


async def apply_contextualizer(
    *,
    chunks: list[PreparedChunk],
    markdown: str,
    filename: str,
    contextualizer: "ChunkContextualizer | None",
    chunk_context_cache: "ChunkContextCache | None",
    event_sink: "EventSink | None" = None,
) -> list[PreparedChunk]:
    from rag_core.documents.contextualizer import ChunkContextRequest, NoOpContextualizer

    if not chunks or contextualizer is None or isinstance(contextualizer, NoOpContextualizer):
        return list(chunks)

    resolved = contextualizer
    if chunk_context_cache is not None:
        from rag_core.documents.contextualizer_adapters import CachingContextualizer
        from rag_core.search.providers.embedding_cache_models import sha256_text

        document_sha256 = sha256_text(markdown)
        resolved = CachingContextualizer(
            resolved,
            chunk_context_cache,
            document_sha256_resolver=lambda _: document_sha256,
        )

    contextualizer_id = getattr(resolved, "contextualizer_id", type(resolved).__name__)
    emit_event(
        event_sink,
        ContextualizeStarted(
            chunk_count=len(chunks),
            model=contextualizer_id,
        ),
    )
    started_ms = now_ms()
    embedding_texts: list[str] = []
    try:
        total = len(chunks)
        for chunk in chunks:
            context = await resolved.contextualize(
                ChunkContextRequest(
                    document_markdown=markdown,
                    document_filename=filename,
                    chunk_text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    total_chunks=total,
                )
            )
            embedding_texts.append(
                f"{context}\n\n{chunk.text}" if context else chunk.text
            )
    except Exception:
        emit_event(
            event_sink,
            ContextualizeCompleted(
                chunk_count=len(chunks),
                model=contextualizer_id,
                duration_ms=now_ms() - started_ms,
                succeeded=False,
            ),
        )
        raise
    emit_event(
        event_sink,
        ContextualizeCompleted(
            chunk_count=len(chunks),
            model=contextualizer_id,
            duration_ms=now_ms() - started_ms,
            succeeded=True,
        ),
    )
    return override_embedding_texts(chunks, embedding_texts)
