from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class SourceForHeader(Protocol):
    @property
    def title(self) -> str | None: ...

    @property
    def document_key(self) -> str | None: ...

    @property
    def document_id(self) -> str | None: ...

    @property
    def source_id(self) -> str: ...

    @property
    def result_id(self) -> str: ...


class LocatorForLabel(Protocol):
    @property
    def chunk_index(self) -> int | None: ...

    @property
    def section_path(self) -> str | None: ...

    @property
    def page_number(self) -> int | None: ...

    @property
    def slide_number(self) -> int | None: ...

    @property
    def sheet_name(self) -> str | None: ...

    @property
    def row_range(self) -> str | None: ...


class RenderableSnippet(Protocol):
    def as_text(self) -> str: ...


def format_header(
    citation_id: str,
    source: SourceForHeader,
    locator: LocatorForLabel,
    *,
    model_safe: bool = False,
) -> str:
    title = model_safe_source_title(source) if model_safe else source_title(source)
    location = format_location(locator)
    suffix = f" {location}" if location else ""
    return f"[{citation_id}] {title}{suffix}"


def source_title(source: SourceForHeader) -> str:
    return source.title or source.document_key or source.document_id or source.result_id


def model_safe_source_title(source: SourceForHeader) -> str:
    return source.title or source.source_id or source.result_id


def render_snippets(
    snippets: Iterable[RenderableSnippet], *, max_chars: int | None
) -> str:
    rendered = "\n\n".join(snippet.as_text() for snippet in snippets)
    if max_chars is not None:
        return rendered[:max_chars]
    return rendered


def format_location(locator: LocatorForLabel) -> str:
    parts: list[str] = []
    if locator.section_path:
        parts.append(locator.section_path)
    elif locator.sheet_name:
        parts.append(f"sheet {locator.sheet_name}")
    if locator.slide_number is not None and not _contains_location(
        locator.section_path,
        "Slide %d" % locator.slide_number,
    ):
        parts.append(f"slide {locator.slide_number}")
    if locator.page_number is not None and not _contains_location(
        locator.section_path,
        "Page %d" % locator.page_number,
    ):
        parts.append(f"page {locator.page_number}")
    if locator.row_range and not _contains_location(
        locator.section_path,
        "Rows %s" % locator.row_range,
    ):
        parts.append(f"rows {locator.row_range}")
    if locator.chunk_index is not None:
        parts.append(f"chunk {locator.chunk_index}")
    return ", ".join(parts)


def _contains_location(section_path: str | None, value: str) -> bool:
    if not section_path:
        return False
    return value.lower() in section_path.lower()


__all__ = [
    "format_header",
    "format_location",
    "model_safe_source_title",
    "render_snippets",
    "source_title",
]
