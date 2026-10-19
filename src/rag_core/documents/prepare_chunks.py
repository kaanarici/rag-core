from __future__ import annotations

from rag_core._async import run_sync

from collections.abc import Mapping
from dataclasses import replace
from typing import Sequence

from rag_core.config import ChunkingConfig
from rag_core.config import PRECHUNKED_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk
from rag_core.documents.chunking.protocol import ChunkConfig
from rag_core.documents.chunking.router import chunk_text, chunk_text_async
from rag_core.documents.pdf_page_locators import (
    with_pdf_page_locators as _with_pdf_page_locators,
)


def prepare_text_chunks(
    text: str,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
    embedding_texts: Sequence[str] | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> list[PreparedChunk]:
    return _finalize_text_chunks(
        text=text,
        chunks=chunk_text(
            text,
            config=_chunk_config(chunking_config),
            mime_type=mime_type,
            filename=filename,
        ),
        mime_type=mime_type,
        filename=filename,
        embedding_texts=embedding_texts,
    )


async def prepare_text_chunks_async(
    text: str,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
    embedding_texts: Sequence[str] | None = None,
    chunking_config: ChunkingConfig | None = None,
    slide_ranges: Sequence[Mapping[str, int]] | None = None,
) -> list[PreparedChunk]:
    if slide_ranges is None:
        chunks = await chunk_text_async(
            text,
            config=_chunk_config(chunking_config),
            mime_type=mime_type,
            filename=filename,
        )
    else:
        chunks = await run_sync(
            _chunk_slides,
            text,
            slide_ranges,
            chunking_config=chunking_config,
            mime_type=mime_type,
            filename=filename,
        )
    return _finalize_text_chunks(
        text=text,
        chunks=chunks,
        mime_type=mime_type,
        filename=filename,
        embedding_texts=embedding_texts,
    )


def _chunk_slides(
    text: str,
    slide_ranges: Sequence[Mapping[str, int]],
    *,
    chunking_config: ChunkingConfig | None,
    mime_type: str | None,
    filename: str | None,
) -> list[PreparedChunk]:
    """Chunk converter-owned slide slices so source headings cannot move boundaries."""
    chunks: list[PreparedChunk] = []
    for slide in slide_ranges:
        start, end = slide["start_char"], slide["end_char"]
        for chunk in chunk_text(
            text[start:end],
            config=_chunk_config(chunking_config),
            mime_type=mime_type,
            filename=filename,
        ):
            chunks.append(
                replace(
                    chunk,
                    chunk_index=len(chunks),
                    start_char=start + chunk.start_char
                    if chunk.start_char is not None
                    else None,
                    end_char=start + chunk.end_char
                    if chunk.end_char is not None
                    else None,
                    metadata={**chunk.metadata, "slide_number": slide["slide_number"]},
                )
            )
    return chunks


def prepare_pre_chunked_texts(
    texts: Sequence[str],
    *,
    embedding_texts: Sequence[str] | None = None,
    chunk_metadata: Sequence[Mapping[str, object]] | None = None,
    chunking_strategy: str = PRECHUNKED_CHUNKING_STRATEGY,
) -> list[PreparedChunk]:
    resolved = resolve_embedding_texts(texts, embedding_texts)
    resolved_metadata = _resolve_chunk_metadata(texts, chunk_metadata)
    return [
        PreparedChunk(
            chunk_index=index,
            text=text,
            embedding_text=resolved[index],
            word_count=len(text.split()),
            start_char=None,
            end_char=None,
            token_count=0,
            chunking_strategy=chunking_strategy,
            metadata={
                **dict(resolved_metadata[index]),
                "chunking_strategy": chunking_strategy,
                "offset_reconstruction": "unreliable",
            },
            sparse_text=resolved[index] if embedding_texts is not None else None,
        )
        for index, text in enumerate(texts)
    ]


def override_embedding_texts(
    chunks: Sequence[PreparedChunk],
    embedding_texts: Sequence[str],
) -> list[PreparedChunk]:
    resolved = resolve_embedding_texts(
        [chunk.text for chunk in chunks], embedding_texts
    )
    return [
        replace(chunk, embedding_text=resolved[index], sparse_text=resolved[index])
        for index, chunk in enumerate(chunks)
    ]


def resolve_embedding_texts(
    texts: Sequence[str],
    embedding_texts: Sequence[str] | None,
) -> list[str]:
    if embedding_texts is not None:
        if len(embedding_texts) != len(texts):
            raise ValueError(
                "embedding_texts length mismatch: expected %d got %d"
                % (len(texts), len(embedding_texts))
            )
        return list(embedding_texts)
    return list(texts)


def _resolve_chunk_metadata(
    texts: Sequence[str],
    chunk_metadata: Sequence[Mapping[str, object]] | None,
) -> list[Mapping[str, object]]:
    if chunk_metadata is None:
        return [{} for _ in texts]
    if len(chunk_metadata) != len(texts):
        raise ValueError(
            "chunk_metadata length mismatch: expected %d got %d"
            % (len(texts), len(chunk_metadata))
        )
    return list(chunk_metadata)


def _chunk_config(config: ChunkingConfig | None) -> ChunkConfig:
    if config is None:
        return ChunkConfig()
    return ChunkConfig(
        strategy=config.strategy,
        max_chars=config.max_chars,
        overlap=config.overlap,
    )


def _finalize_text_chunks(
    *,
    text: str,
    chunks: Sequence[PreparedChunk],
    mime_type: str | None,
    filename: str | None,
    embedding_texts: Sequence[str] | None,
) -> list[PreparedChunk]:
    annotated = _with_pdf_page_locators(
        text=text,
        chunks=chunks,
        mime_type=mime_type,
        filename=filename,
    )
    if embedding_texts is None:
        return annotated
    return override_embedding_texts(annotated, embedding_texts)


__all__ = [
    "override_embedding_texts",
    "prepare_pre_chunked_texts",
    "prepare_text_chunks_async",
    "prepare_text_chunks",
    "resolve_embedding_texts",
]
