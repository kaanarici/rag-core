from __future__ import annotations

from typing import Iterable

from rag_core.search.context_pack_helpers import (
    SourceDedupeKey as _SourceDedupeKey,
    base_source_id as _base_source_id,
    estimate_tokens as _estimate_tokens,
    resolve_char_budget as _resolve_char_budget,
    source_dedupe_key as _source_dedupe_key,
    retrieval_metadata_from_result as _retrieval_metadata_from_result,
)
from rag_core.search.context_pack_models import (
    ContextSnippet,
    ModelContextPack,
    SourceLocator,
    SourcePreview,
    SourceReference,
    source_preview_from_snippet,
)
from rag_core.search.context_pack_rendering import (
    format_header as _format_header,
    render_snippets as _render_snippets,
)
from rag_core.search.context_pack_sources import (
    source_locator_from_result,
    source_reference_from_result,
)
from rag_core.search.result_scores import finite_score_or_zero
from rag_core.search.types import SearchResult


def build_context_pack(
    results: Iterable[SearchResult],
    *,
    query: str,
    max_snippets: int = 8,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    chars_per_token: int = 4,
) -> ModelContextPack:
    """Convert ranked search hits into a deterministic model context pack."""

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
        header = _format_header(source.source_id, source, locator)
        text_budget = (
            None if remaining_chars is None else remaining_chars - len(header) - 1
        )
        if text_budget is not None and text_budget <= 0:
            dropped_count += len(all_results) - index
            truncated = True
            break
        text = result.text
        snippet_truncated = False
        if text_budget is not None and len(text) > text_budget:
            text = text[:text_budget].rstrip()
            snippet_truncated = True
            truncated = True
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

    return ModelContextPack(
        query=query,
        snippets=tuple(snippets),
        dropped_count=max(0, dropped_count),
        max_snippets=max_snippets,
        max_chars=char_budget,
        max_tokens=max_tokens,
        token_estimate=sum(snippet.token_estimate for snippet in snippets),
        char_count=len(_render_snippets(snippets, max_chars=char_budget)),
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


__all__ = [
    "ContextSnippet",
    "ModelContextPack",
    "SourceLocator",
    "SourcePreview",
    "SourceReference",
    "build_context_pack",
    "source_locator_from_result",
    "source_preview_from_snippet",
    "source_reference_from_result",
]
