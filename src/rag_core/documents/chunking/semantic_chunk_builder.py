"""Prepared chunk assembly helpers for semantic chunking."""

from __future__ import annotations

from rag_core.config.chunking_config import SEMANTIC_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk, estimate_token_count
from rag_core.documents.chunking.protocol import ChunkConfig
from rag_core.documents.chunking.spans import resolve_chunk_bounds


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
            start_char, end_char, reliable = _resolve_bounds(
                text,
                piece,
                search_start=search_start,
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
    buffer_len = 0
    chunk_idx = 0
    search_start = 0

    for sentence in sentences:
        if buffer_len + len(sentence) + 1 > config.max_chars and buffer:
            chunk_text = " ".join(buffer).strip()
            start_char, end_char, reliable = _resolve_bounds(
                full_text,
                chunk_text,
                search_start=search_start,
            )
            search_start = end_char
            chunks.append(
                _prepared_chunk(
                    chunk_idx,
                    chunk_text,
                    start_char,
                    end_char,
                    "semantic_heuristic",
                    offset_reliable=reliable,
                )
            )
            chunk_idx += 1
            buffer, buffer_len = _next_buffer(chunk_text, overlap=config.overlap)

        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    if buffer:
        chunk_text = " ".join(buffer).strip()
        if chunk_text:
            start_char, end_char, reliable = _resolve_bounds(
                full_text,
                chunk_text,
                search_start=search_start,
            )
            chunks.append(
                _prepared_chunk(
                    chunk_idx,
                    chunk_text,
                    start_char,
                    end_char,
                    "semantic_heuristic",
                    offset_reliable=reliable,
                )
            )

    return chunks


def single_semantic_chunk(text: str) -> PreparedChunk:
    return PreparedChunk(
        chunk_index=0,
        text=text,
        embedding_text=text,
        word_count=len(text.split()),
        start_char=0,
        end_char=len(text),
        token_count=estimate_token_count(text),
        chunking_strategy=SEMANTIC_CHUNKING_STRATEGY,
        metadata={"chunking_strategy": SEMANTIC_CHUNKING_STRATEGY},
    )


def _segment_to_chunks(segment: str, *, config: ChunkConfig) -> list[str]:
    if len(segment) <= config.max_chars:
        return [segment]

    pieces: list[str] = []
    step = max(1, config.max_chars - max(0, config.overlap))
    index = 0
    while index < len(segment):
        piece = segment[index : index + config.max_chars].strip()
        if piece:
            pieces.append(piece)
        index += step
    return pieces


def _resolve_bounds(
    full_text: str,
    chunk_text: str,
    *,
    search_start: int,
) -> tuple[int, int, bool]:
    return resolve_chunk_bounds(full_text, chunk_text, search_start=search_start)


def _next_buffer(chunk_text: str, *, overlap: int) -> tuple[list[str], int]:
    if overlap <= 0:
        return [], 0
    overlap_text = chunk_text[-overlap:]
    return [overlap_text], len(overlap_text)


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
