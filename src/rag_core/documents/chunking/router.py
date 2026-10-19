"""Deterministic Markdown chunking, with row-aware XLSX handling."""

from __future__ import annotations

from rag_core._async import run_sync
from dataclasses import replace
from .code_detection import is_code_content as is_code_content
from .markdown import MarkdownChunker
from .protocol import ChunkConfig, ChunkingStrategy
from .xlsx import XlsxChunker
from rag_core.core_models import PreparedChunk


def create_chunking_strategy(name: str, **kwargs: object) -> ChunkingStrategy:
    if name == "markdown":
        return MarkdownChunker()
    if name == "xlsx":
        return XlsxChunker()
    raise ValueError(
        f"Unknown chunking strategy: {name}; pass prepared chunks for specialized chunking"
    )


def chunk_text(
    text: str,
    *,
    config: ChunkConfig | None = None,
    mime_type: str | None = None,
    filename: str | None = None,
) -> list[PreparedChunk]:
    config = config or ChunkConfig()
    strategy = config.strategy
    if strategy == "auto":
        strategy = (
            "xlsx"
            if (filename or "").lower().endswith(".xlsx")
            or (mime_type or "").split(";", 1)[0]
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else "markdown"
        )
    chunks = create_chunking_strategy(strategy).chunk(text, config)
    if is_code_content(mime_type=mime_type, filename=filename):
        for index, chunk in enumerate(chunks):
            if chunk.start_char is None or chunk.end_char is None:
                continue
            chunks[index] = replace(
                chunk,
                metadata={
                    **chunk.metadata,
                    "line_start": text.count("\n", 0, chunk.start_char) + 1,
                    "line_end": text.count("\n", 0, chunk.end_char)
                    + int(bool(chunk.end_char and text[chunk.end_char - 1] != "\n")),
                },
            )
    return chunks


async def chunk_text_async(
    text: str,
    *,
    config: ChunkConfig | None = None,
    mime_type: str | None = None,
    filename: str | None = None,
) -> list[PreparedChunk]:
    return await run_sync(
        chunk_text, text, config=config, mime_type=mime_type, filename=filename
    )
