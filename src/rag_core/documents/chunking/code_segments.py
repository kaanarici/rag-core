from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rag_core.config.chunking_config import CODE_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk, estimate_token_count

from .budget import fitting_overlap_tail, fits_chunk_budget
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
    chunk_idx = 0
    search_start = 0

    for segment in segments:
        candidate = "\n\n".join([*buffer, segment]).strip()
        if buffer and not fits_chunk_budget(
            candidate,
            max_chars=config.max_chars,
        ):
            chunk_idx, search_start = _flush_buffer(
                text=text,
                chunks=chunks,
                buffer=buffer,
                index=chunk_idx,
                search_start=search_start,
                metadata=metadata,
                joiner="\n\n",
            )
            buffer = _retain_overlap(
                buffer,
                following=segment,
                joiner="\n\n",
                config=config,
            )
            if buffer:
                search_start = max(0, search_start - len(buffer[0]))

        if not fits_chunk_budget(segment, max_chars=config.max_chars):
            buffer = []
            for line in segment.split("\n"):
                if not fits_chunk_budget(line, max_chars=config.max_chars):
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
                    chunk_idx, search_start = _append_split_line_chunks(
                        text=text,
                        line=line,
                        chunks=chunks,
                        index=chunk_idx,
                        search_start=search_start,
                        metadata=metadata,
                        config=config,
                    )
                    continue
                line_candidate = "\n".join([*buffer, line]).strip()
                if buffer and not fits_chunk_budget(
                    line_candidate,
                    max_chars=config.max_chars,
                ):
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

                buffer.append(line)
            continue

        buffer.append(segment)

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


def _append_split_line_chunks(
    *,
    text: str,
    line: str,
    chunks: list[PreparedChunk],
    index: int,
    search_start: int,
    metadata: Mapping[str, str],
    config: ChunkConfig,
) -> tuple[int, int]:
    spans = split_text_span(
        line,
        0,
        len(line),
        max_chars=config.max_chars,
        overlap=config.overlap,
    )
    line_start = text.find(line, search_start)
    if line_start < 0:
        for span in spans:
            index, search_start = _flush_buffer(
                text=text,
                chunks=chunks,
                buffer=[line[span.start : span.end]],
                index=index,
                search_start=search_start,
                metadata=metadata,
                joiner="\n",
            )
        return index, search_start

    for span in spans:
        chunk_text = line[span.start : span.end]
        start_char = line_start + span.start
        end_char = line_start + span.end
        chunk_metadata: dict[str, object] = dict(metadata)
        line_number_start, line_number_end = _line_range(
            text,
            start=start_char,
            end=end_char,
        )
        chunk_metadata["line_start"] = line_number_start
        chunk_metadata["line_end"] = line_number_end
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
        index += 1
    return index, line_start + len(line)


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


def _retain_overlap(
    buffer: Sequence[str],
    *,
    following: str,
    joiner: str,
    config: ChunkConfig,
) -> list[str]:
    if not buffer:
        return []

    tail = fitting_overlap_tail(
        buffer[-1],
        following,
        separator=joiner,
        overlap=config.overlap,
        max_chars=config.max_chars,
    )
    if not tail:
        return []
    return [tail]


def _line_range(full_text: str, *, start: int, end: int) -> tuple[int, int]:
    line_start = full_text.count("\n", 0, start) + 1
    inclusive_end = max(start, end - 1)
    line_end = full_text.count("\n", 0, inclusive_end) + 1
    return line_start, line_end
