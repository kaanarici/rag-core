"""Markdown-aware chunking strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from rag_core.config.chunking_config import MARKDOWN_CHUNKING_STRATEGY
from rag_core.core_models import PreparedChunk, estimate_token_count

from .protocol import ChunkConfig
from .spans import TextSpan, split_text_span

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SHEET_TITLE_RE = re.compile(
    r"^Sheet:\s+(.+?)(?:\s+\(Rows\s+([1-9]\d*)-([1-9]\d*)\))?$",
    re.IGNORECASE,
)
_SLIDE_TITLE_RE = re.compile(r"^Slide\s+([1-9]\d*)$", re.IGNORECASE)


@dataclass(frozen=True)
class _Section:
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _RawChunk:
    text: str
    metadata: dict[str, object]
    start: int
    end: int
    reliable: bool


class MarkdownChunker:
    """Chunks markdown by headers, then paragraphs, as exact source slices.

    Chunk ``text`` is a verbatim slice of the input, so ``(start_char,
    end_char)`` reproduces it exactly. Including for overlapped chunks. When
    a section cannot be located verbatim in the source (e.g. ``\\r\\n`` line
    endings were normalized during section splitting), that section falls
    back to normalized chunk text with cursor-estimated offsets flagged
    ``offset_reconstruction='unreliable'`` so EvidenceSpan resolvers refuse
    them.
    """

    def chunk(self, text: str, config: ChunkConfig) -> List[PreparedChunk]:
        if not text:
            return []

        raw_chunks: List[_RawChunk] = []
        cursor = 0
        for section in _split_sections(text):
            if not section.text:
                continue
            section_start = text.find(section.text, cursor)
            if section_start == -1:
                raw_chunks.extend(_normalized_section_chunks(section, config))
                continue
            section_end = section_start + len(section.text)
            cursor = section_end
            for span in _section_chunk_spans(
                text, section_start, section_end, config
            ):
                raw_chunks.append(
                    _RawChunk(
                        text=text[span.start : span.end],
                        metadata=section.metadata,
                        start=span.start,
                        end=span.end,
                        reliable=True,
                    )
                )

        chunks: List[PreparedChunk] = []
        prev_end = 0
        for idx, raw in enumerate(raw_chunks):
            if raw.reliable:
                start, end = raw.start, raw.end
            else:
                start, end = prev_end, prev_end + len(raw.text)
            prev_end = end
            metadata = dict(raw.metadata)
            if not raw.reliable:
                metadata["offset_reconstruction"] = "unreliable"
            chunks.append(
                PreparedChunk(
                    chunk_index=idx,
                    text=raw.text,
                    embedding_text=raw.text,
                    word_count=len(raw.text.split()),
                    start_char=start,
                    end_char=end,
                    token_count=estimate_token_count(raw.text),
                    chunking_strategy=MARKDOWN_CHUNKING_STRATEGY,
                    metadata=metadata,
                )
            )

        return chunks


def _section_chunk_spans(
    text: str,
    section_start: int,
    section_end: int,
    config: ChunkConfig,
) -> List[TextSpan]:
    if section_end - section_start <= config.max_chars:
        return [TextSpan(section_start, section_end)]

    spans: List[TextSpan] = []
    start: int | None = None
    end = section_start
    for part in _part_spans(text, section_start, section_end):
        if part.end - part.start > config.max_chars:
            if start is not None and end > start:
                spans.append(TextSpan(start, end))
            spans.extend(
                split_text_span(
                    text,
                    part.start,
                    part.end,
                    max_chars=config.max_chars,
                    overlap=config.overlap,
                )
            )
            start = (
                max(part.start, part.end - config.overlap)
                if config.overlap > 0
                else None
            )
            end = part.end
            continue
        if start is not None and part.end - start > config.max_chars and end > start:
            spans.append(TextSpan(start, end))
            # Overlap is a verbatim tail of the flushed chunk, clamped so a
            # short chunk never pushes the next start before its own start.
            start = max(start, end - config.overlap) if config.overlap > 0 else None
        if start is None:
            start = part.start
        end = part.end
    if start is not None and end > start:
        spans.append(TextSpan(start, end))
    return spans


def _part_spans(text: str, section_start: int, section_end: int) -> List[TextSpan]:
    spans: List[TextSpan] = []
    cursor = section_start
    for raw_part in text[section_start:section_end].split("\n\n"):
        part_start = cursor
        cursor += len(raw_part) + 2
        stripped = raw_part.strip()
        if not stripped:
            continue
        lead = len(raw_part) - len(raw_part.lstrip())
        spans.append(TextSpan(part_start + lead, part_start + lead + len(stripped)))
    return spans


def _normalized_section_chunks(
    section: _Section,
    config: ChunkConfig,
) -> List[_RawChunk]:
    """Legacy normalized assembly for sections not found verbatim in source."""
    raw_chunks: List[_RawChunk] = []

    def _append(chunk_text: str) -> None:
        if chunk_text:
            raw_chunks.append(
                _RawChunk(
                    text=chunk_text,
                    metadata=section.metadata,
                    start=0,
                    end=0,
                    reliable=False,
                )
            )

    if len(section.text) <= config.max_chars:
        _append(section.text)
        return raw_chunks

    parts = [p.strip() for p in section.text.split("\n\n") if p.strip()]
    buffer: List[str] = []
    buffer_len = 0
    for part in parts:
        if len(part) > config.max_chars:
            if buffer:
                _append("\n\n".join(buffer).strip())
                buffer = []
                buffer_len = 0
            for span in split_text_span(
                part,
                0,
                len(part),
                max_chars=config.max_chars,
                overlap=config.overlap,
            ):
                _append(part[span.start : span.end])
            continue
        if buffer_len + len(part) + 2 > config.max_chars and buffer:
            chunk_text = "\n\n".join(buffer).strip()
            _append(chunk_text)
            if config.overlap > 0 and chunk_text:
                overlap_text = chunk_text[-config.overlap :]
                buffer = [overlap_text]
                buffer_len = len(overlap_text)
            else:
                buffer = []
                buffer_len = 0
        buffer.append(part)
        buffer_len += len(part) + 2
    if buffer:
        _append("\n\n".join(buffer).strip())
    return raw_chunks


def _split_sections(text: str) -> List[_Section]:
    lines = text.splitlines()
    sections: List[_Section] = []
    current: List[str] = []
    current_metadata: dict[str, object] = {}
    heading_stack: list[tuple[int, str]] = []

    for line in lines:
        heading = _parse_heading(line)
        if heading is not None:
            if current:
                sections.append(_Section("\n".join(current).strip(), current_metadata))
            level, title = heading
            heading_stack = [
                (existing_level, existing_title)
                for existing_level, existing_title in heading_stack
                if existing_level < level
            ]
            heading_stack.append((level, title))
            current_metadata = _section_metadata(heading_stack)
            current = [line]
            continue
        current.append(line)

    if current:
        sections.append(_Section("\n".join(current).strip(), current_metadata))

    return sections


def _parse_heading(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line.strip())
    if match is None:
        return None
    title = match.group(2).strip()
    if not title:
        return None
    return (len(match.group(1)), title)


def _section_metadata(heading_stack: list[tuple[int, str]]) -> dict[str, object]:
    if not heading_stack:
        return {}
    titles = [title for _, title in heading_stack]
    metadata: dict[str, object] = {
        "section_path": " > ".join(titles),
        "section_title": titles[-1],
    }
    for title in titles:
        slide_match = _SLIDE_TITLE_RE.match(title)
        if slide_match is not None:
            metadata["slide_number"] = int(slide_match.group(1))
            break
    for title in titles:
        sheet_match = _SHEET_TITLE_RE.match(title)
        if sheet_match is None:
            continue
        metadata["sheet_name"] = sheet_match.group(1).strip()
        start_row = sheet_match.group(2)
        end_row = sheet_match.group(3)
        if start_row is not None and end_row is not None:
            metadata["row_range"] = "%s-%s" % (start_row, end_row)
        break
    return metadata
