from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rag_core.config.chunking_config import CODE_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk, estimate_token_count

from .protocol import ChunkConfig
from .spans import resolve_chunk_bounds, split_text_span


def mask_non_code_regions(text: str) -> str:
    def _spaces(match: re.Match[str]) -> str:
        return " " * len(match.group(0))

    masked = text
    masked = re.sub(r'"""[\s\S]*?"""', _spaces, masked)
    masked = re.sub(r"'''[\s\S]*?'''", _spaces, masked)
    masked = re.sub(r"(?m)^\s*#.*$", _spaces, masked)
    masked = re.sub(r"(?m)^\s*//.*$", _spaces, masked)
    masked = re.sub(r"/\*[\s\S]*?\*/", _spaces, masked)
    masked = re.sub(r'"(?:\\.|[^"\\])*"', _spaces, masked)
    masked = re.sub(r"'(?:\\.|[^'\\])*'", _spaces, masked)
    return masked


def segments_from_boundaries(text: str, boundaries: Sequence[int]) -> list[str]:
    segments: list[str] = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def build_code_chunk_metadata(
    *,
    chunking_engine: str,
    resolved_language: str | None,
) -> dict[str, str]:
    metadata = {
        "chunking_strategy": CODE_CHUNKING_STRATEGY,
        "chunking_engine": chunking_engine,
    }
    if resolved_language:
        metadata["language"] = resolved_language
    return metadata


def assemble_code_chunks(
    *,
    text: str,
    segments: Sequence[str],
    config: ChunkConfig,
    metadata: Mapping[str, str],
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    buffer: list[str] = []
    buffer_len = 0
    chunk_idx = 0
    search_start = 0

    for segment in segments:
        if buffer_len + len(segment) > config.max_chars and buffer:
            chunk_idx, search_start = _flush_buffer(
                text=text,
                chunks=chunks,
                buffer=buffer,
                index=chunk_idx,
                search_start=search_start,
                metadata=metadata,
                joiner="\n\n",
            )
            buffer, buffer_len = _retain_overlap(buffer, config.overlap)

        if len(segment) > config.max_chars:
            for line in segment.split("\n"):
                if len(line) > config.max_chars:
                    if buffer:
                        chunk_idx, search_start = _flush_buffer(
                            text=text,
                            chunks=chunks,
                            buffer=buffer,
                            index=chunk_idx,
                            search_start=search_start,
                            metadata=metadata,
                            joiner="\n",
                        )
                        buffer = []
                        buffer_len = 0
                    for span in split_text_span(
                        line,
                        0,
                        len(line),
                        max_chars=config.max_chars,
                        overlap=config.overlap,
                    ):
                        chunk_idx, search_start = _flush_buffer(
                            text=text,
                            chunks=chunks,
                            buffer=[line[span.start : span.end]],
                            index=chunk_idx,
                            search_start=search_start,
                            metadata=metadata,
                            joiner="\n",
                        )
                    continue
                if buffer_len + len(line) + 1 > config.max_chars and buffer:
                    chunk_idx, search_start = _flush_buffer(
                        text=text,
                        chunks=chunks,
                        buffer=buffer,
                        index=chunk_idx,
                        search_start=search_start,
                        metadata=metadata,
                        joiner="\n",
                    )
                    buffer = []
                    buffer_len = 0

                buffer.append(line)
                buffer_len += len(line) + 1
            continue

        buffer.append(segment)
        buffer_len += len(segment) + 2

    if buffer:
        _flush_buffer(
            text=text,
            chunks=chunks,
            buffer=buffer,
            index=chunk_idx,
            search_start=search_start,
            metadata=metadata,
            joiner="\n\n",
        )

    return chunks


def _flush_buffer(
    *,
    text: str,
    chunks: list[PreparedChunk],
    buffer: Sequence[str],
    index: int,
    search_start: int,
    metadata: Mapping[str, str],
    joiner: str,
) -> tuple[int, int]:
    chunk_text = joiner.join(buffer).strip()
    if not chunk_text:
        return index, search_start

    start_char, end_char, reliable = resolve_chunk_bounds(
        text,
        chunk_text,
        search_start=search_start,
    )
    chunk_metadata: dict[str, object] = dict(metadata)
    if reliable:
        line_start, line_end = _line_range(text, start=start_char, end=end_char)
        chunk_metadata["line_start"] = line_start
        chunk_metadata["line_end"] = line_end
    else:
        # Chunk text not found verbatim: line_start/line_end and span are
        # derived from the running cursor. Flag so EvidenceSpan resolvers
        # refuse the span.
        chunk_metadata["offset_reconstruction"] = "unreliable"
    chunks.append(
        PreparedChunk(
            chunk_index=index,
            text=chunk_text,
            embedding_text=chunk_text,
            word_count=len(chunk_text.split()),
            start_char=start_char,
            end_char=end_char,
            token_count=estimate_token_count(chunk_text),
            chunking_strategy=CODE_CHUNKING_STRATEGY,
            metadata=chunk_metadata,
        )
    )
    return index + 1, end_char


def _retain_overlap(buffer: Sequence[str], overlap: int) -> tuple[list[str], int]:
    if overlap <= 0 or not buffer:
        return [], 0

    last = buffer[-1]
    tail = last[-overlap:] if len(last) > overlap else last
    return [tail], len(tail)


def _line_range(full_text: str, *, start: int, end: int) -> tuple[int, int]:
    line_start = full_text.count("\n", 0, start) + 1
    inclusive_end = max(start, end - 1)
    line_end = full_text.count("\n", 0, inclusive_end) + 1
    return line_start, line_end
