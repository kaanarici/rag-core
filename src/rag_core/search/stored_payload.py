from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.result_scores import finite_score_or_zero
from rag_core.search.stored_payload_fields import (
    optional_payload_int,
    optional_payload_str,
    required_payload_str,
    result_metadata,
)
from rag_core.search.vector_models import (
    SEARCH_RESULT_TYPE_TEXT,
    ContentType,
    SearchResult,
)

SECTION_PAYLOAD_KEYS = (
    "result_type",
    "figure_id",
    "image_url",
    "thumbnail_url",
    "figure_thumbnail_url",
    "figure_caption",
    "figure_bbox",
    "page_index",
    "page_number",
    "bbox",
    "page_width",
    "page_height",
    "slide_number",
    "paragraph_index",
    "sheet_name",
    "row_range",
    "line_start",
    "line_end",
    "is_full_page",
    "anchor_chunk_index",
)

_FIXED_PAYLOAD_KEYS = frozenset(
    {
        "mime_type",
        "document_path",
        "chunk_word_count",
        "chunk_token_estimate",
        "chunker_strategy",
        "embedding_model",
        "section_id",
        "section_path",
        "section_title",
        *SECTION_PAYLOAD_KEYS,
    }
)


def build_stored_payload(
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    document_key: str | None,
    content_sha256: str | None,
    processing_version: str | None,
    filename: str,
    mime_type: str,
    source_type: str,
    document_path: str | None,
    chunk_index: int,
    chunk_text: str,
    chunk_token_count: int,
    payload_text: str,
    content_type: ContentType,
    embedding_model: str | None,
    chunker_strategy: str,
    title: str | None,
    filter_metadata: Mapping[Any, object] | None = None,
    section_info: dict[str, object] | None = None,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    chunk_word_count = len(chunk_text.split())
    chunk_token_estimate = int(chunk_token_count or max(1, round(chunk_word_count * 1.3)))
    payload: dict[str, object] = {
        policy.namespace_field: namespace,
        policy.corpus_id_field: corpus_id,
        policy.document_id_field: document_id,
        policy.document_key_field: document_key,
        policy.content_sha256_field: content_sha256,
        policy.content_type_field: content_type,
        policy.chunk_index_field: chunk_index,
        policy.text_field: payload_text,
        policy.title_field: title or filename,
        policy.source_type_field: source_type,
        "mime_type": mime_type,
        "document_path": document_path,
        "chunk_word_count": chunk_word_count,
        "chunk_token_estimate": chunk_token_estimate,
        "chunker_strategy": chunker_strategy,
        "result_type": SEARCH_RESULT_TYPE_TEXT,
    }
    if embedding_model:
        payload["embedding_model"] = embedding_model
    if processing_version:
        payload[policy.processing_version_field] = processing_version
    payload.update(
        _filterable_metadata(
            filter_metadata or {},
            policy=policy,
        )
    )
    if not section_info:
        return payload

    for key in ("section_id", "section_path", "section_title"):
        value = section_info.get(key)
        if value is not None:
            payload[key] = value
    for key in SECTION_PAYLOAD_KEYS:
        value = section_info.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _filterable_metadata(
    metadata: Mapping[Any, object],
    *,
    policy: VectorStorePolicy,
) -> dict[str, object]:
    reserved = _FIXED_PAYLOAD_KEYS | {
        policy.namespace_field,
        policy.corpus_id_field,
        policy.document_id_field,
        policy.document_key_field,
        policy.content_sha256_field,
        policy.processing_version_field,
        policy.content_type_field,
        policy.source_type_field,
        policy.chunk_index_field,
        policy.text_field,
        policy.title_field,
    }
    filterable: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if key in reserved:
            continue
        if not key.strip():
            continue
        if key == "quality" and isinstance(value, Mapping):
            filterable.update(_quality_filter_fields(value))
            continue
        if _is_filterable_value(value):
            filterable[key] = value
    return filterable


def _quality_filter_fields(quality: Mapping[Any, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for source_key, target_key in (
        ("verdict", "quality_verdict"),
        ("details", "quality_details"),
    ):
        value = quality.get(source_key)
        if isinstance(value, str) and value.strip():
            fields[target_key] = value
    for source_key, target_key in (
        ("char_count", "quality_char_count"),
        ("page_count", "quality_page_count"),
    ):
        value = quality.get(source_key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            fields[target_key] = value
    for source_key, target_key in (
        ("meaningful_ratio", "quality_meaningful_ratio"),
        ("mojibake_ratio", "quality_mojibake_ratio"),
        ("text_to_page_ratio", "quality_text_to_page_ratio"),
    ):
        value = _finite_plain_float(quality.get(source_key))
        if value is not None:
            fields[target_key] = value
    return fields


def _is_filterable_value(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if _is_geo_point_value(value):
        return True
    return isinstance(value, (str, int, float))


def _is_geo_point_value(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if set(value) != {"lat", "lon"}:
        return False
    lat = _finite_plain_float(value.get("lat"))
    lon = _finite_plain_float(value.get("lon"))
    return (
        lat is not None
        and lon is not None
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )


def _is_finite_plain_number(value: object) -> bool:
    return _finite_plain_float(value) is not None


def _finite_plain_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def payload_to_result(
    *,
    point_id: str,
    payload: dict[str, object],
    score: float,
    policy: VectorStorePolicy = DEFAULT_POLICY,
) -> SearchResult:
    return SearchResult(
        id=point_id,
        text=required_payload_str(payload, policy.text_field),
        score=finite_score_or_zero(score),
        content_type=required_payload_str(payload, policy.content_type_field),
        source_type=optional_payload_str(payload, policy.source_type_field) or "",
        namespace=optional_payload_str(payload, policy.namespace_field),
        document_id=optional_payload_str(payload, policy.document_id_field),
        corpus_id=optional_payload_str(payload, policy.corpus_id_field),
        document_key=optional_payload_str(payload, policy.document_key_field),
        content_sha256=optional_payload_str(payload, policy.content_sha256_field),
        title=optional_payload_str(payload, policy.title_field),
        section_id=optional_payload_str(payload, "section_id"),
        section_title=optional_payload_str(payload, "section_title"),
        section_path=optional_payload_str(payload, "section_path"),
        document_path=optional_payload_str(payload, "document_path"),
        chunk_index=optional_payload_int(payload, policy.chunk_index_field),
        chunk_word_count=optional_payload_int(payload, "chunk_word_count"),
        chunk_token_estimate=optional_payload_int(payload, "chunk_token_estimate"),
        embedding_model=optional_payload_str(payload, "embedding_model"),
        chunker_strategy=optional_payload_str(payload, "chunker_strategy"),
        result_type=optional_payload_str(payload, "result_type"),
        figure_id=optional_payload_str(payload, "figure_id"),
        figure_thumbnail_url=(
            optional_payload_str(payload, "thumbnail_url")
            or optional_payload_str(payload, "figure_thumbnail_url")
        ),
        metadata=result_metadata(payload, policy=policy),
    )


def merge_duplicate_result(
    preferred: SearchResult,
    fallback: SearchResult,
) -> SearchResult:
    return SearchResult(
        id=preferred.id,
        text=preferred.text,
        score=max(
            finite_score_or_zero(preferred.score),
            finite_score_or_zero(fallback.score),
        ),
        content_type=preferred.content_type,
        source_type=preferred.source_type,
        namespace=preferred.namespace or fallback.namespace,
        document_id=preferred.document_id or fallback.document_id,
        corpus_id=preferred.corpus_id or fallback.corpus_id,
        document_key=preferred.document_key or fallback.document_key,
        content_sha256=preferred.content_sha256 or fallback.content_sha256,
        title=preferred.title or fallback.title,
        section_id=preferred.section_id or fallback.section_id,
        section_title=preferred.section_title or fallback.section_title,
        section_path=preferred.section_path or fallback.section_path,
        document_path=preferred.document_path or fallback.document_path,
        chunk_index=preferred.chunk_index if preferred.chunk_index is not None else fallback.chunk_index,
        chunk_word_count=(
            preferred.chunk_word_count
            if preferred.chunk_word_count is not None
            else fallback.chunk_word_count
        ),
        chunk_token_estimate=(
            preferred.chunk_token_estimate
            if preferred.chunk_token_estimate is not None
            else fallback.chunk_token_estimate
        ),
        embedding_model=preferred.embedding_model or fallback.embedding_model,
        chunker_strategy=preferred.chunker_strategy or fallback.chunker_strategy,
        result_type=preferred.result_type or fallback.result_type,
        figure_id=preferred.figure_id or fallback.figure_id,
        figure_thumbnail_url=preferred.figure_thumbnail_url or fallback.figure_thumbnail_url,
        metadata={**fallback.metadata, **preferred.metadata},
    )
