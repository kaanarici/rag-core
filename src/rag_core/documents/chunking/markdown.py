"""Markdown-aware chunking strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from rag_core.core_models import PreparedChunk

from .protocol import ChunkConfig

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
    metadata: dict[str, object] = field(default_factory=dict)


class MarkdownChunker:
    """Chunks markdown by splitting on headers, then paragraphs."""

    def chunk(self, text: str, config: ChunkConfig) -> List[PreparedChunk]:
        if not text:
            return []

        max_chars = config.max_chars
        overlap = config.overlap

        sections = _split_sections(text)
        raw_chunks: List[_RawChunk] = []
        for section in sections:
            if len(section.text) <= max_chars:
                raw_chunks.append(_RawChunk(section.text, section.metadata))
                continue

            parts = [p.strip() for p in section.text.split("\n\n") if p.strip()]
            buffer: List[str] = []
            buffer_len = 0
            for part in parts:
                if buffer_len + len(part) + 2 > max_chars and buffer:
                    chunk_text = "\n\n".join(buffer).strip()
                    raw_chunks.append(_RawChunk(chunk_text, section.metadata))
                    if overlap > 0 and chunk_text:
                        overlap_text = chunk_text[-overlap:]
                        buffer = [overlap_text]
                        buffer_len = len(overlap_text)
                    else:
                        buffer = []
                        buffer_len = 0
                buffer.append(part)
                buffer_len += len(part) + 2
            if buffer:
                raw_chunks.append(_RawChunk("\n\n".join(buffer).strip(), section.metadata))

        chunks: List[PreparedChunk] = []
        char_pos = 0
        for idx, raw_chunk in enumerate(raw_chunks):
            raw = raw_chunk.text
            if not raw:
                continue
            start = text.find(raw[:50], char_pos)
            if start == -1:
                start = char_pos
            word_count = len(raw.split())
            chunks.append(
                PreparedChunk(
                    chunk_index=idx,
                    text=raw,
                    embedding_text=raw,
                    word_count=word_count,
                    start_char=start,
                    end_char=start + len(raw),
                    token_count=word_count,
                    chunking_strategy="markdown",
                    metadata=dict(raw_chunk.metadata),
                )
            )
            char_pos = start + len(raw)

        return chunks


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
