from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_core.search.context_pack_helpers import drop_none as _drop_none
from rag_core.search.context_pack_rendering import (
    format_header as _format_header,
    format_location as _format_location,
    model_safe_source_title as _model_safe_source_title,
    render_snippets as _render_snippets,
    source_title as _source_title,
)


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

    def to_model_payload(self) -> dict[str, object]:
        return _drop_none(
            {
                "source_id": self.source_id,
                "result_id": self.result_id,
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
    bbox: tuple[float, float, float, float] | None = None
    figure_id: str | None = None
    figure_caption: str | None = None
    figure_thumbnail_url: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
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

    def to_model_payload(self) -> dict[str, object]:
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
    """One model-ready context block with its source reference."""

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
        return _format_header(self.citation_id, self.source, self.locator)

    @property
    def model_header(self) -> str:
        return _format_header(
            self.citation_id,
            self.source,
            self.locator,
            model_safe=True,
        )

    def as_text(self) -> str:
        return f"{self.header}\n{self.text}"

    def as_model_text(self) -> str:
        return f"{self.model_header}\n{self.text}"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "citation_id": self.citation_id,
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "source": _drop_none(asdict(self.source)),
            "locator": self.locator.to_payload(),
            "token_estimate": self.token_estimate,
            "char_count": self.char_count,
            "truncated": self.truncated,
        }
        if self.retrieval_metadata is not None:
            payload["retrieval_metadata"] = self.retrieval_metadata
        return payload

    def to_model_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "citation_id": self.citation_id,
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "source": self.source.to_model_payload(),
            "locator": self.locator.to_payload(),
            "token_estimate": self.token_estimate,
            "char_count": self.char_count,
            "truncated": self.truncated,
        }
        if self.retrieval_metadata is not None:
            payload["retrieval_metadata"] = self.retrieval_metadata
        return payload


@dataclass(frozen=True)
class ModelContextPack:
    """Deterministic retrieval context ready for a caller-owned model request."""

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
    def model_source_previews(self) -> tuple[SourcePreview, ...]:
        return tuple(model_source_preview_from_snippet(snippet) for snippet in self.snippets)

    @property
    def citation_summary(self) -> str:
        return "\n".join(preview.as_text() for preview in self.source_previews)

    @property
    def model_citation_summary(self) -> str:
        return "\n".join(preview.as_text() for preview in self.model_source_previews)

    def as_text(self) -> str:
        return _render_snippets(self.snippets, max_chars=self.max_chars)

    def as_model_text(self) -> str:
        rendered = "\n\n".join(snippet.as_model_text() for snippet in self.snippets)
        if self.max_chars is not None:
            return rendered[: self.max_chars]
        return rendered

    def to_payload(self) -> dict[str, object]:
        rendered_text = self.as_text()
        return {
            "query": self.query,
            "snippets": [snippet.to_payload() for snippet in self.snippets],
            "citations": [_drop_none(asdict(source)) for source in self.citations],
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

    def to_model_payload(self) -> dict[str, object]:
        rendered_text = self.as_model_text()
        return {
            "query": self.query,
            "snippets": [snippet.to_model_payload() for snippet in self.snippets],
            "citations": [source.to_model_payload() for source in self.citations],
            "source_previews": [
                preview.to_model_payload() for preview in self.model_source_previews
            ],
            "citation_summary": self.model_citation_summary,
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


def model_source_preview_from_snippet(snippet: ContextSnippet) -> SourcePreview:
    source = snippet.source
    return SourcePreview(
        citation_id=snippet.citation_id,
        title=_model_safe_source_title(source),
        locator_label=_format_location(snippet.locator) or None,
        source_type=source.source_type,
        result_type=source.result_type,
        truncated=snippet.truncated,
    )
