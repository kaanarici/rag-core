"""Prepared chunk assembly helpers for semantic chunking."""

from __future__ import annotations

from rag_core.core_models import PreparedChunk, estimate_token_count
from rag_core.documents.chunking.budget import (
    fitting_overlap_tail,
    fits_chunk_budget,
)
from rag_core.documents.chunking.protocol import ChunkConfig
from rag_core.documents.chunking.spans import resolve_chunk_bounds, split_text_span


def build_chunks_from_segments(
    text: str,
    segments: list[str],
    config: ChunkConfig,
    *,
    strategy_name: str,
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    chunk_idx = 0
    search_start = 0

    for segment in segments:
        for piece in _segment_to_chunks(segment, config=config):
            start_char, end_char, reliable = resolve_chunk_bounds(
                text,
                piece,
                search_start=max(0, search_start - config.overlap),
            )
            search_start = end_char
            chunks.append(
                _prepared_chunk(
                    chunk_idx,
                    piece,
                    start_char,
                    end_char,
                    strategy_name,
                    offset_reliable=reliable,
                )
            )
            chunk_idx += 1

    return chunks


def paragraph_heuristic_chunks(
    full_text: str,
    sentences: list[str],
    config: ChunkConfig,
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    buffer: list[str] = []
    chunk_idx = 0
    search_start = 0

    for sentence in sentences:
        candidate = " ".join([*buffer, sentence]).strip()
        if buffer and not fits_chunk_budget(
            candidate,
            max_chars=config.max_chars,
        ):
            chunk_text = " ".join(buffer).strip()
            chunk_idx, search_start = _append_semantic_pieces(
                chunks=chunks,
                full_text=full_text,
                chunk_text=chunk_text,
                config=config,
                chunk_idx=chunk_idx,
                search_start=search_start,
                strategy_name="semantic_heuristic",
            )
            buffer = _next_buffer(
                chunk_text,
                following=sentence,
                config=config,
            )
            if buffer:
                search_start = max(0, search_start - len(buffer[0]))

        buffer.append(sentence)

    if buffer:
        chunk_text = " ".join(buffer).strip()
        if chunk_text:
            _append_semantic_pieces(
                chunks=chunks,
                full_text=full_text,
                chunk_text=chunk_text,
                config=config,
                chunk_idx=chunk_idx,
                search_start=search_start,
                strategy_name="semantic_heuristic",
            )

    return chunks


def _segment_to_chunks(segment: str, *, config: ChunkConfig) -> list[str]:
    if fits_chunk_budget(segment, max_chars=config.max_chars):
        return [segment]

    return [
        segment[span.start : span.end]
        for span in split_text_span(
            segment,
            0,
            len(segment),
            max_chars=config.max_chars,
            overlap=config.overlap,
        )
    ]


def _append_semantic_pieces(
    *,
    chunks: list[PreparedChunk],
    full_text: str,
    chunk_text: str,
    config: ChunkConfig,
    chunk_idx: int,
    search_start: int,
    strategy_name: str,
) -> tuple[int, int]:
    for piece in _segment_to_chunks(chunk_text, config=config):
        start_char, end_char, reliable = resolve_chunk_bounds(
            full_text,
            piece,
            search_start=max(0, search_start - config.overlap),
        )
        search_start = end_char
        chunks.append(
            _prepared_chunk(
                chunk_idx,
                piece,
                start_char,
                end_char,
                strategy_name,
                offset_reliable=reliable,
            )
        )
        chunk_idx += 1
    return chunk_idx, search_start


def _next_buffer(
    chunk_text: str,
    *,
    following: str,
    config: ChunkConfig,
) -> list[str]:
    overlap_text = fitting_overlap_tail(
        chunk_text,
        following,
        separator=" ",
        overlap=config.overlap,
        max_chars=config.max_chars,
    )
    if not overlap_text:
        return []
    return [overlap_text]


def _prepared_chunk(
    chunk_idx: int,
    text: str,
    start_char: int,
    end_char: int,
    strategy_name: str,
    *,
    offset_reliable: bool = True,
) -> PreparedChunk:
    metadata: dict[str, object] = {"chunking_strategy": strategy_name}
    if not offset_reliable:
        # Chunk text was not found verbatim in the source; downstream
        # EvidenceSpan resolvers must refuse this span rather than render
        # whatever sat under the running cursor.
        metadata["offset_reconstruction"] = "unreliable"
    return PreparedChunk(
        chunk_index=chunk_idx,
        text=text,
        embedding_text=text,
        word_count=len(text.split()),
        start_char=start_char,
        end_char=end_char,
        token_count=estimate_token_count(text),
        chunking_strategy=strategy_name,
        metadata=metadata,
    )
