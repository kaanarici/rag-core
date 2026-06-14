"""Field coercion helpers for stored vector payloads."""

from __future__ import annotations

from enum import Enum
from typing import Final, Optional

from rag_core.search.policy import VectorStorePolicy

SEARCH_RESULT_STORED_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "section_id",
    "section_title",
    "section_path",
    "document_path",
    "chunk_word_count",
    "chunk_token_estimate",
    "start_char",
    "end_char",
    "embedding_model",
    "chunker_strategy",
    "result_type",
    "figure_id",
    "figure_thumbnail_url",
)

SEARCH_RESULT_FILTER_FIELDS: Final[tuple[str, ...]] = (
    "namespace",
    "content_type",
    "source_type",
    "document_id",
    "corpus_id",
    "document_key",
    "content_sha256",
    "title",
    *SEARCH_RESULT_STORED_METADATA_FIELDS,
)

SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD: Final[str] = "thumbnail_url"

# Keys consumed into top-level ``SearchResult`` fields and excluded from
# metadata. Policy-specific keys are added at runtime so renamed fields are
# still excluded.
_FIXED_RESULT_KEYS = frozenset(
    (*SEARCH_RESULT_STORED_METADATA_FIELDS, SEARCH_RESULT_LEGACY_THUMBNAIL_FIELD)
)


def result_metadata(
    payload: dict[str, object],
    *,
    policy: VectorStorePolicy,
) -> dict[str, object]:
    result_keys = _FIXED_RESULT_KEYS | _policy_result_keys(policy)
    return {key: value for key, value in payload.items() if key not in result_keys}


def required_payload_str(payload: dict[str, object], key: str) -> str:
    if key not in payload:
        raise ValueError("search payload missing required field: %s" % key)
    return payload_str_value(payload[key], key=key)


def optional_payload_str(payload: dict[str, object], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    return payload_str_value(value, key=key)


def optional_payload_int(payload: dict[str, object], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if "." in stripped:
            return None
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _policy_result_keys(policy: VectorStorePolicy) -> frozenset[str]:
    return frozenset(
        {
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
    )


def payload_str_value(value: object, *, key: str) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        return value
    raise ValueError("search payload field must be a string: %s" % key)
