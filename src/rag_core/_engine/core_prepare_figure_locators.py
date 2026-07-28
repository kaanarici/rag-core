from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import re
from typing import NamedTuple

from rag_core.core_models import PreparedChunk

_FIGURE_METADATA_KEYS = (
    "bbox",
    "figure_bbox",
    "figure_thumbnail_url",
    "image_url",
    "page_index",
    "page_height",
    "page_number",
    "page_width",
    "paragraph_index",
    "row_range",
    "sheet_name",
    "slide_number",
    "thumbnail_url",
    "text_anchor_end_char",
    "text_anchor_start_char",
)


class _FigureLocator(NamedTuple):
    figure_id: str
    label: str
    description: str
    metadata: dict[str, object]


def with_figure_locators(
    *,
    chunks: Sequence[PreparedChunk],
    metadata: Mapping[str, object],
) -> list[PreparedChunk]:
    figures = _figure_locators(metadata.get("figure_items"))
    if not figures:
        return list(chunks)

    assignments: dict[int, list[_FigureLocator]] = {}
    for figure in figures:
        chunk_index = _matching_chunk_index(figure=figure, chunks=chunks)
        if chunk_index is not None:
            assignments.setdefault(chunk_index, []).append(figure)

    annotated: list[PreparedChunk] = []
    for index, chunk in enumerate(chunks):
        matches = assignments.get(index, [])
        if len(matches) != 1:
            annotated.append(chunk)
            continue
        figure = matches[0]
        chunk_metadata = dict(chunk.metadata)
        chunk_metadata["figure_id"] = figure.figure_id
        if figure.description:
            chunk_metadata["figure_caption"] = figure.description
        for key, value in figure.metadata.items():
            if value is not None and key not in chunk_metadata:
                chunk_metadata[key] = value
        annotated.append(replace(chunk, metadata=chunk_metadata))
    return annotated


def _matching_chunk_index(
    *,
    figure: _FigureLocator,
    chunks: Sequence[PreparedChunk],
) -> int | None:
    scoped = [
        (index, chunk)
        for index, chunk in enumerate(chunks)
        if _figure_scope_matches(
            figure_metadata=figure.metadata,
            chunk_metadata=chunk.metadata,
        )
    ]
    anchor = _int_value(figure.metadata.get("text_anchor_start_char"))
    if anchor is not None:
        anchored = [
            (index, chunk)
            for index, chunk in scoped
            if chunk.metadata.get("offset_reconstruction") != "unreliable"
            if chunk.start_char is not None
            and chunk.end_char is not None
            and chunk.start_char <= anchor < chunk.end_char
        ]
        if anchored:
            return max(
                anchored,
                key=lambda entry: (
                    entry[1].start_char if entry[1].start_char is not None else -1,
                    -entry[0],
                ),
            )[0]

    phrase_matches = [
        index
        for index, chunk in scoped
        if _figure_phrase_matches_chunk(figure=figure, chunk=chunk)
    ]
    return phrase_matches[0] if len(phrase_matches) == 1 else None


def _figure_locators(raw_items: object) -> list[_FigureLocator]:
    if not isinstance(raw_items, list):
        return []

    figures: list[_FigureLocator] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        figure_id = _non_blank_str(item.get("figure_id"))
        if figure_id is None:
            continue
        figures.append(
            _FigureLocator(
                figure_id=figure_id,
                label=_non_blank_str(item.get("label")) or figure_id,
                description=_non_blank_str(item.get("description")) or "",
                metadata=_figure_metadata(item),
            )
        )
    return figures


def _figure_metadata(item: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    raw_nested = item.get("metadata")
    if isinstance(raw_nested, Mapping):
        for key in _FIGURE_METADATA_KEYS:
            value = raw_nested.get(key)
            if value is not None:
                metadata[key] = value
    for key in (
        "bbox",
        "figure_bbox",
        "figure_thumbnail_url",
        "image_url",
        "page_index",
        "thumbnail_url",
    ):
        value = item.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _figure_phrase_matches_chunk(
    *,
    figure: _FigureLocator,
    chunk: PreparedChunk,
) -> bool:
    text = chunk.text.casefold()
    probes = (figure.figure_id, figure.label, figure.description)
    return any(_contains_phrase(text, probe) for probe in probes)


def _contains_phrase(normalized_text: str, probe: str) -> bool:
    normalized_probe = " ".join(probe.casefold().split())
    if not normalized_probe:
        return False
    pattern = rf"(?<![\w:]){re.escape(normalized_probe)}(?![\w:])"
    return re.search(pattern, normalized_text) is not None


def _figure_scope_matches(
    *,
    figure_metadata: Mapping[str, object],
    chunk_metadata: Mapping[str, object],
) -> bool:
    for key in ("slide_number", "sheet_name"):
        figure_value = figure_metadata.get(key)
        chunk_value = chunk_metadata.get(key)
        if (
            figure_value is not None
            and chunk_value is not None
            and figure_value != chunk_value
        ):
            return False
    return True


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _non_blank_str(value: object) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved or None
