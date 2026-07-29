"""Shared char-span resolution for assembled chunk text.

Chunkers that synthesize chunk text (joined sentences, overlap tails) cannot
derive offsets positionally, so they locate the chunk in the source. The
contract here is exact-or-flagged: a span is reliable only when the full
chunk text occurs verbatim at the resolved offset, so
``source[start:end] == chunk_text`` holds. Anything else returns
``reliable=False`` and the caller must flag the chunk
``offset_reconstruction='unreliable'`` for EvidenceSpan resolvers to refuse.
"""

from __future__ import annotations

from typing import NamedTuple


class TextSpan(NamedTuple):
    start: int
    end: int


def resolve_chunk_bounds(
    full_text: str,
    chunk_text: str,
    *,
    search_start: int,
) -> tuple[int, int, bool]:
    if not chunk_text:
        return search_start, search_start, True

    start = full_text.find(chunk_text, search_start)
    if start >= 0:
        return start, start + len(chunk_text), True
    return search_start, min(len(full_text), search_start + len(chunk_text)), False


def split_text_span(
    text: str,
    start: int,
    end: int,
    *,
    max_chars: int,
    overlap: int,
) -> list[TextSpan]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if end <= start:
        return []
    if end - start <= max_chars:
        piece_start, piece_end = _trim_span(text, start, end)
        return [TextSpan(piece_start, piece_end)] if piece_start < piece_end else []

    spans: list[TextSpan] = []
    cursor = start
    resolved_overlap = max(0, min(overlap, max_chars - 1))
    while cursor < end:
        hard_end = min(end, cursor + max_chars)
        window_end = (
            hard_end
            if hard_end >= end
            else _safe_window_end(text, cursor, hard_end, max_chars=max_chars)
        )
        piece_start, piece_end = _trim_span(text, cursor, window_end)
        if piece_start < piece_end:
            spans.append(TextSpan(piece_start, piece_end))
        if window_end >= end:
            break
        next_cursor = window_end - resolved_overlap if resolved_overlap else window_end
        cursor = next_cursor if next_cursor > cursor else window_end
    return spans


def _safe_window_end(
    text: str,
    start: int,
    hard_end: int,
    *,
    max_chars: int,
) -> int:
    min_boundary = start + max(1, (max_chars * 2) // 3)
    for index in range(hard_end - 1, min_boundary - 1, -1):
        if text[index].isspace():
            return index
    return hard_end


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


__all__ = ["TextSpan", "resolve_chunk_bounds", "split_text_span"]
