"""Portable lexical sidecar matching helpers."""

from __future__ import annotations

from dataclasses import replace

from rag_core.search.result_filters import result_matches_sidecar_query
from rag_core.search.request_models import SearchSidecarQuery
from rag_core.search.vector_models import SearchResult

LexicalMatch = tuple[float, SearchResult]
_BestFieldMatch = tuple[float, str, str, str]


def normalized_lexical_query(value: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def match_lexical_result(
    *,
    record_namespace: str,
    result: SearchResult,
    query: SearchSidecarQuery,
    needle: str,
    trigram_threshold: float,
) -> LexicalMatch | None:
    if record_namespace != query.namespace:
        return None
    result = _with_record_namespace(result, record_namespace)
    if not result_matches_sidecar_query(result, query):
        return None

    best = _best_match(needle, _searchable_fields(result), trigram_threshold)
    if best is None:
        return None

    score, strategy, field_name, matched_value = best
    return score, _annotate_result(
        result,
        score=score,
        strategy=strategy,
        field_name=field_name,
        matched_value=matched_value,
    )


def _searchable_fields(result: SearchResult) -> list[tuple[str, str]]:
    raw_values = [
        ("title", result.title),
        ("section_title", result.section_title),
        ("section_path", result.section_path),
        ("document_path", result.document_path),
        ("text", result.text),
    ]
    return [(name, value) for name, value in raw_values if value]


def _with_record_namespace(result: SearchResult, namespace: str) -> SearchResult:
    if result.namespace:
        return result
    return replace(result, namespace=namespace)


def _best_match(
    needle: str,
    searchable_fields: list[tuple[str, str]],
    trigram_threshold: float,
) -> _BestFieldMatch | None:
    best: _BestFieldMatch | None = None
    for field_name, raw_value in searchable_fields:
        candidate = normalized_lexical_query(raw_value)
        if not candidate:
            continue

        if candidate == needle:
            return (1.0, "exact", field_name, raw_value)
        if f" {needle} " in f" {candidate} ":
            score = 0.9
            if best is None or score > best[0]:
                best = (score, "exact", field_name, raw_value)
            continue

        score = _trigram_score(needle, candidate)
        if score < trigram_threshold:
            continue
        if best is None or score > best[0]:
            best = (score, "trigram", field_name, raw_value)
    return best


def _annotate_result(
    result: SearchResult,
    *,
    score: float,
    strategy: str,
    field_name: str,
    matched_value: str,
) -> SearchResult:
    metadata = dict(result.metadata)
    metadata["search_sidecar"] = {
        "score": score,
        "field": field_name,
        "strategy": strategy,
        "matched_value": matched_value,
    }
    return replace(result, score=score, metadata=metadata)


def _trigram_score(left: str, right: str) -> float:
    left_trigrams = _trigrams(left)
    right_trigrams = _trigrams(right)
    if not left_trigrams or not right_trigrams:
        return 0.0
    overlap = len(left_trigrams & right_trigrams)
    total = len(left_trigrams | right_trigrams)
    if total == 0:
        return 0.0
    return overlap / total


def _trigrams(value: str) -> set[str]:
    padded = f"  {value}  "
    if len(padded) < 3:
        return set()
    return {padded[index : index + 3] for index in range(len(padded) - 2)}
