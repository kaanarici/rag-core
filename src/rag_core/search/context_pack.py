from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Final, Iterable

from rag_core.retrieval_defaults import DEFAULT_CONTEXT_LIMIT
from rag_core.search.context_pack_helpers import (
    SourceDedupeKey as _SourceDedupeKey,
    base_source_id as _base_source_id,
    estimate_tokens as _estimate_tokens,
    resolve_char_budget as _resolve_char_budget,
    source_dedupe_key as _source_dedupe_key,
    retrieval_metadata_from_result as _retrieval_metadata_from_result,
)
from rag_core.search.context_pack_models import (
    CONTEXT_ORDER_VALUES,
    Citation,
    Context,
    ContextSnippet,
    ContextOrder,
    SourceLocator,
    SourcePreview,
    format_snippet_header,
    render_context_snippets,
    source_preview_from_snippet,
    validate_context_order,
)
from rag_core.search.context_pack_sources import (
    source_locator_from_result,
    source_reference_from_result,
)
from rag_core.search.result_scores import finite_score_or_zero
from rag_core.search.vector_models import SearchResult

CONTEXT_EXPANSION_BEFORE_METADATA_KEY: Final[str] = "context_expansion_before"
CONTEXT_EXPANSION_AFTER_METADATA_KEY: Final[str] = "context_expansion_after"
_CONTEXT_EXPANSION_SEPARATOR: Final[str] = "\n\n"


@dataclass(frozen=True)
class EvidenceSpan:
    """App-facing citation address resolving back to a chunk's bytes.

    Names mirror the consumer contract: ``artifact_id`` is rag-core's
    ``document_id``, and ``start_offset`` / ``end_offset`` are char offsets in
    the converted markdown the gateway persists as a derived Artifact.
    """

    artifact_id: str
    start_offset: int
    end_offset: int
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def evidence_span_from_result(result: SearchResult) -> EvidenceSpan | None:
    """Return an EvidenceSpan iff the result has a resolvable char-span and document.

    Returns ``None`` when the chunker could not commit reliable offsets
    (``offset_reconstruction='unreliable'`` in the stored chunk metadata).
    Callers must not synthesize an EvidenceSpan from running cursors.
    """
    if not result.document_id:
        return None
    if result.start_char is None or result.end_char is None:
        return None
    if result.end_char < result.start_char:
        return None
    if result.metadata.get("offset_reconstruction") == "unreliable":
        return None
    page = _safe_int(result.metadata.get("page_number"))
    slide = _safe_int(result.metadata.get("slide_number"))
    sheet_raw = result.metadata.get("sheet_name")
    sheet = sheet_raw if isinstance(sheet_raw, str) and sheet_raw.strip() else None
    return EvidenceSpan(
        artifact_id=result.document_id,
        start_offset=result.start_char,
        end_offset=result.end_char,
        page=page,
        slide=slide,
        sheet=sheet,
    )


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def context_pack_response_payload(
    pack: Context,
    *,
    context_order: ContextOrder = "rank",
) -> dict[str, object]:
    """Return app-facing payload plus prompt-safe text for CLI/HTTP."""
    return {
        **pack.to_payload(),
        "context_text": pack.as_prompt_text(context_order=context_order),
    }


def build_context_pack(
    results: Iterable[SearchResult],
    *,
    query: str,
    max_snippets: int = DEFAULT_CONTEXT_LIMIT,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    chars_per_token: int = 4,
) -> Context:
    """Convert ranked search hits into a deterministic Context."""

    if max_snippets <= 0:
        raise ValueError("max_snippets must be positive")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be positive when set")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive when set")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")

    all_results = list(results)
    colliding_source_ids = _colliding_source_ids(_deduped_results(all_results))
    char_budget = _resolve_char_budget(
        max_chars=max_chars,
        max_tokens=max_tokens,
        chars_per_token=chars_per_token,
    )
    remaining_chars = char_budget
    snippets: list[ContextSnippet] = []
    used_source_ids: set[str] = set()
    used_source_keys: set[_SourceDedupeKey] = set()
    dropped_count = 0
    truncated = False

    for index, result in enumerate(all_results):
        if len(snippets) >= max_snippets:
            dropped_count += len(all_results) - index
            truncated = True
            break
        source_key = _source_dedupe_key(result)
        if source_key in used_source_keys:
            dropped_count += 1
            continue
        if remaining_chars is not None and remaining_chars <= 0:
            dropped_count += len(all_results) - index
            truncated = True
            break
        if remaining_chars is not None:
            separator_chars = 2 if snippets else 0
            if remaining_chars <= separator_chars:
                dropped_count += len(all_results) - index
                truncated = True
                break
            remaining_chars -= separator_chars
        source = source_reference_from_result(
            result,
            used_source_ids=used_source_ids,
            require_stable_suffix=_base_source_id(result) in colliding_source_ids,
        )
        locator = source_locator_from_result(result)
        header = format_snippet_header(source.source_id, source, locator)
        text_budget = (
            None if remaining_chars is None else remaining_chars - len(header) - 1
        )
        if text_budget is not None and text_budget <= 0:
            dropped_count += len(all_results) - index
            truncated = True
            break
        text, snippet_truncated, locator_tracks_text = _context_text_for_result(
            result,
            text_budget,
        )
        if snippet_truncated:
            truncated = True
        # Narrow end_offset to the kept slice so EvidenceSpan resolves back to
        # the truncated text actually rendered.
        if (
            snippet_truncated
            and locator_tracks_text
            and locator.start_offset is not None
            and locator.end_offset is not None
        ):
            adjusted_end = locator.start_offset + len(text)
            if adjusted_end < locator.end_offset:
                locator = replace(locator, end_offset=adjusted_end)
        rendered_char_count = len(header) + 1 + len(text)
        token_estimate = _estimate_tokens(
            f"{header}\n{text}",
            chars_per_token=chars_per_token,
        )
        used_source_keys.add(source_key)
        snippets.append(
            ContextSnippet(
                citation_id=source.source_id,
                rank=len(snippets) + 1,
                text=text,
                score=finite_score_or_zero(result.score),
                source=source,
                locator=locator,
                token_estimate=token_estimate,
                char_count=rendered_char_count,
                retrieval_metadata=_retrieval_metadata_from_result(result),
                truncated=snippet_truncated,
            )
        )
        if remaining_chars is not None:
            remaining_chars -= rendered_char_count

    return Context(
        query=query,
        snippets=tuple(snippets),
        dropped_count=max(0, dropped_count),
        max_snippets=max_snippets,
        max_chars=char_budget,
        max_tokens=max_tokens,
        token_estimate=sum(snippet.token_estimate for snippet in snippets),
        char_count=len(render_context_snippets(snippets, max_chars=char_budget)),
        truncated=truncated,
    )


def _colliding_source_ids(results: list[SearchResult]) -> set[str]:
    counts: dict[str, int] = {}
    for result in results:
        source_id = _base_source_id(result)
        counts[source_id] = counts.get(source_id, 0) + 1
    return {source_id for source_id, count in counts.items() if count > 1}


def _deduped_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    used_source_keys: set[_SourceDedupeKey] = set()
    for result in results:
        source_key = _source_dedupe_key(result)
        if source_key in used_source_keys:
            continue
        used_source_keys.add(source_key)
        deduped.append(result)
    return deduped


def _context_text_for_result(
    result: SearchResult,
    max_chars: int | None,
) -> tuple[str, bool, bool]:
    before = _metadata_text_parts(
        result.metadata.get(CONTEXT_EXPANSION_BEFORE_METADATA_KEY)
    )
    after = _metadata_text_parts(
        result.metadata.get(CONTEXT_EXPANSION_AFTER_METADATA_KEY)
    )
    if not before and not after:
        return _fit_plain_text(result.text, max_chars)

    before_text = _CONTEXT_EXPANSION_SEPARATOR.join(before)
    after_text = _CONTEXT_EXPANSION_SEPARATOR.join(after)
    full_text = _join_context_parts((before_text, result.text, after_text))
    if max_chars is None or len(full_text) <= max_chars:
        return full_text, False, False
    return _fit_expanded_text(
        before_text=before_text,
        body_text=result.text,
        after_text=after_text,
        max_chars=max_chars,
    )


def _fit_plain_text(text: str, max_chars: int | None) -> tuple[str, bool, bool]:
    if max_chars is None or len(text) <= max_chars:
        return text, False, True
    return text[:max_chars].rstrip(), True, True


def _fit_expanded_text(
    *,
    before_text: str,
    body_text: str,
    after_text: str,
    max_chars: int,
) -> tuple[str, bool, bool]:
    body, body_truncated, locator_tracks_text = _fit_plain_text(body_text, max_chars)
    if body_truncated or len(body) >= max_chars:
        return body, True, locator_tracks_text

    remaining = max_chars - len(body)
    before_segment = ""
    after_segment = ""
    if before_text and after_text:
        neighbor_budget = max(0, remaining - (2 * len(_CONTEXT_EXPANSION_SEPARATOR)))
        before_keep = min(len(before_text), neighbor_budget // 2)
        after_keep = min(len(after_text), neighbor_budget - before_keep)
        extra = neighbor_budget - before_keep - after_keep
        if extra and before_keep < len(before_text):
            added = min(extra, len(before_text) - before_keep)
            before_keep += added
            extra -= added
        if extra and after_keep < len(after_text):
            after_keep += min(extra, len(after_text) - after_keep)
        before_segment = _tail_text(before_text, before_keep)
        after_segment = _head_text(after_text, after_keep)
    elif before_text:
        before_segment = _tail_text(
            before_text,
            max(0, remaining - len(_CONTEXT_EXPANSION_SEPARATOR)),
        )
    elif after_text:
        after_segment = _head_text(
            after_text,
            max(0, remaining - len(_CONTEXT_EXPANSION_SEPARATOR)),
        )

    expanded = _join_context_parts((before_segment, body, after_segment))
    if expanded == body:
        return body, True, True
    return expanded, True, False


def _metadata_text_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    parts: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            parts.append(text)
    return tuple(parts)


def _join_context_parts(parts: Iterable[str]) -> str:
    return _CONTEXT_EXPANSION_SEPARATOR.join(part for part in parts if part)


def _tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[-max_chars:].lstrip()


def _head_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[:max_chars].rstrip()


__all__ = [
    "CONTEXT_ORDER_VALUES",
    "Citation",
    "Context",
    "ContextSnippet",
    "ContextOrder",
    "CONTEXT_EXPANSION_AFTER_METADATA_KEY",
    "CONTEXT_EXPANSION_BEFORE_METADATA_KEY",
    "EvidenceSpan",
    "SourceLocator",
    "SourcePreview",
    "build_context_pack",
    "context_pack_response_payload",
    "evidence_span_from_result",
    "source_locator_from_result",
    "source_preview_from_snippet",
    "source_reference_from_result",
    "validate_context_order",
]
