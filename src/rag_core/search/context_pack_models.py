"""Context-pack dataclasses and text projection for retrieval results.

Citation naming:
- ``to_payload`` / ``as_text`` — stable source ids for app and trace consumers.
- ``to_prompt_payload`` / ``as_prompt_text`` — compact ``S{rank}`` citations for prompt-safe text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from rag_core.search.context_pack_helpers import drop_none as _drop_none


@dataclass(frozen=True)
class SourceReference:
    """Stable source identity for a retrieved chunk."""

    source_id: str
    result_id: str
    document_id: str | None = None
    corpus_id: str | None = None
    document_key: str | None = None
    title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    chunk_index: int | None = None
    content_sha256: str | None = None
    source_type: str | None = None
    result_type: str | None = None

    def to_payload(self) -> dict[str, object]:
        return _drop_none(asdict(self))

    def to_prompt_payload(self, *, citation_id: str | None = None) -> dict[str, object]:
        return _drop_none(
            {
                "citation_id": citation_id,
                "title": self.title,
                "section_title": self.section_title,
                "section_path": self.section_path,
                "chunk_index": self.chunk_index,
                "source_type": self.source_type,
                "result_type": self.result_type,
            }
        )


@dataclass(frozen=True)
class SourceLocator:
    """Optional location hints an app can use for citations or previews."""

    chunk_index: int | None = None
    section_path: str | None = None
    source_hash: str | None = None
    page_number: int | None = None
    page_index: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    row_range: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    figure_id: str | None = None
    figure_caption: str | None = None
    figure_thumbnail_url: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        return payload

    def to_prompt_payload(self) -> dict[str, object]:
        payload = self.to_payload()
        payload.pop("source_hash", None)
        return payload


@dataclass(frozen=True)
class SourcePreview:
    """Compact app-facing preview for citation lists and UI source chips."""

    citation_id: str
    title: str
    locator_label: str | None = None
    document_id: str | None = None
    corpus_id: str | None = None
    document_key: str | None = None
    source_hash: str | None = None
    source_type: str | None = None
    result_type: str | None = None
    truncated: bool = False

    def as_text(self) -> str:
        location = f" ({self.locator_label})" if self.locator_label else ""
        return f"[{self.citation_id}] {self.title}{location}"

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "title": self.title,
            "locator_label": self.locator_label,
            "source_type": self.source_type,
            "result_type": self.result_type,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ContextSnippet:
    """One retrieved context snippet with app and prompt projections."""

    citation_id: str
    rank: int
    text: str
    score: float
    source: SourceReference
    locator: SourceLocator
    token_estimate: int
    char_count: int
    retrieval_metadata: dict[str, object] | None = None
    truncated: bool = False

    @property
    def header(self) -> str:
        return format_snippet_header(self.citation_id, self.source, self.locator)

    @property
    def prompt_header(self) -> str:
        return format_snippet_header(
            self.prompt_citation_id,
            self.source,
            self.locator,
            prompt_safe=True,
        )

    @property
    def prompt_citation_id(self) -> str:
        return f"S{self.rank}"

    def as_text(self) -> str:
        return f"{self.header}\n{self.text}"

    def as_prompt_text(self) -> str:
        return f"{self.prompt_header}\n{self.text}"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "citation_id": self.citation_id,
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "source": self.source.to_payload(),
            "locator": self.locator.to_payload(),
            "token_estimate": self.token_estimate,
            "char_count": self.char_count,
            "truncated": self.truncated,
        }
        if self.retrieval_metadata is not None:
            payload["retrieval_metadata"] = self.retrieval_metadata
        return payload

    def to_prompt_payload(self) -> dict[str, object]:
        rendered_text = self.as_prompt_text()
        payload: dict[str, object] = {
            "citation_id": self.prompt_citation_id,
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "source": self.source.to_prompt_payload(citation_id=self.prompt_citation_id),
            "locator": self.locator.to_prompt_payload(),
            "token_estimate": self.token_estimate,
            "char_count": len(rendered_text),
            "truncated": self.truncated,
        }
        if self.retrieval_metadata is not None:
            payload["retrieval_metadata"] = self.retrieval_metadata
        return payload


@dataclass(frozen=True)
class ContextPack:
    """Deterministic retrieval context with separate app-facing and prompt views."""

    query: str
    snippets: tuple[ContextSnippet, ...]
    dropped_count: int
    max_snippets: int
    max_chars: int | None = None
    max_tokens: int | None = None
    token_estimate: int = 0
    char_count: int = 0
    truncated: bool = False

    @property
    def citations(self) -> tuple[SourceReference, ...]:
        return tuple(snippet.source for snippet in self.snippets)

    @property
    def source_previews(self) -> tuple[SourcePreview, ...]:
        return tuple(source_preview_from_snippet(snippet) for snippet in self.snippets)

    @property
    def prompt_source_previews(self) -> tuple[SourcePreview, ...]:
        return tuple(prompt_source_preview_from_snippet(snippet) for snippet in self.snippets)

    @property
    def citation_summary(self) -> str:
        return "\n".join(preview.as_text() for preview in self.source_previews)

    @property
    def prompt_citation_summary(self) -> str:
        return "\n".join(preview.as_text() for preview in self.prompt_source_previews)

    def as_text(self) -> str:
        """Return app-facing text with stable source ids for traces and UI/debug views."""
        return render_context_snippets(self.snippets, max_chars=self.max_chars)

    def as_prompt_text(self) -> str:
        """Return prompt-safe text with rank-local citation ids for model input."""
        rendered = "\n\n".join(snippet.as_prompt_text() for snippet in self.snippets)
        if self.max_chars is not None:
            return rendered[: self.max_chars]
        return rendered

    def to_payload(self) -> dict[str, object]:
        """Return the app-facing structured payload with stable source identity."""
        rendered_text = self.as_text()
        return {
            "query": self.query,
            "snippets": [snippet.to_payload() for snippet in self.snippets],
            "citations": [source.to_payload() for source in self.citations],
            "source_previews": [preview.to_payload() for preview in self.source_previews],
            "citation_summary": self.citation_summary,
            "dropped_count": self.dropped_count,
            "max_snippets": self.max_snippets,
            "max_chars": self.max_chars,
            "max_tokens": self.max_tokens,
            "token_estimate": self.token_estimate,
            "char_count": len(rendered_text),
            "truncated": self.truncated,
        }

    def to_prompt_payload(self) -> dict[str, object]:
        """Return the prompt-safe structured payload for model tool responses."""
        rendered_text = self.as_prompt_text()
        return {
            "query": self.query,
            "snippets": [snippet.to_prompt_payload() for snippet in self.snippets],
            "citations": [
                snippet.source.to_prompt_payload(citation_id=snippet.prompt_citation_id)
                for snippet in self.snippets
            ],
            "source_previews": [
                preview.to_prompt_payload() for preview in self.prompt_source_previews
            ],
            "citation_summary": self.prompt_citation_summary,
            "dropped_count": self.dropped_count,
            "max_snippets": self.max_snippets,
            "max_chars": self.max_chars,
            "max_tokens": self.max_tokens,
            "token_estimate": self.token_estimate,
            "char_count": len(rendered_text),
            "truncated": self.truncated,
        }


def source_preview_from_snippet(snippet: ContextSnippet) -> SourcePreview:
    source = snippet.source
    return SourcePreview(
        citation_id=snippet.citation_id,
        title=_source_title(source),
        locator_label=_format_location(snippet.locator) or None,
        document_id=source.document_id,
        corpus_id=source.corpus_id,
        document_key=source.document_key,
        source_hash=source.content_sha256,
        source_type=source.source_type,
        result_type=source.result_type,
        truncated=snippet.truncated,
    )


def prompt_source_preview_from_snippet(snippet: ContextSnippet) -> SourcePreview:
    source = snippet.source
    return SourcePreview(
        citation_id=snippet.prompt_citation_id,
        title=_prompt_safe_source_title(source),
        locator_label=_format_location(snippet.locator) or None,
        source_type=source.source_type,
        result_type=source.result_type,
        truncated=snippet.truncated,
    )


def format_snippet_header(
    citation_id: str,
    source: SourceReference,
    locator: SourceLocator,
    *,
    prompt_safe: bool = False,
) -> str:
    title = _prompt_safe_source_title(source) if prompt_safe else _source_title(source)
    location = _format_location(locator)
    suffix = f" {location}" if location else ""
    return f"[{citation_id}] {title}{suffix}"


def _source_title(source: SourceReference) -> str:
    return source.title or source.document_key or source.document_id or source.result_id


def _prompt_safe_source_title(source: SourceReference) -> str:
    return source.title or source.source_type or source.result_type or "source"


def render_context_snippets(
    snippets: Iterable[ContextSnippet], *, max_chars: int | None
) -> str:
    rendered = "\n\n".join(snippet.as_text() for snippet in snippets)
    if max_chars is not None:
        return rendered[:max_chars]
    return rendered


def _format_location(locator: SourceLocator) -> str:
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
    line_label = _line_label(locator)
    if line_label and not _contains_location(locator.section_path, line_label):
        parts.append(line_label)
    if locator.chunk_index is not None:
        parts.append(f"chunk {locator.chunk_index}")
    return ", ".join(parts)


def _line_label(locator: SourceLocator) -> str | None:
    if locator.line_start is None:
        return None
    if locator.line_end is None or locator.line_end == locator.line_start:
        return f"line {locator.line_start}"
    return f"lines {locator.line_start}-{locator.line_end}"


def _contains_location(section_path: str | None, value: str) -> bool:
    if not section_path:
        return False
    return value.lower() in section_path.lower()
