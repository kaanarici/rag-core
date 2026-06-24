from __future__ import annotations

import hashlib
import math
import re
from math import ceil

from rag_core.search.vector_models import SEARCH_RESULT_TYPE_TEXT, SearchResult

SourceDedupeKey = tuple[str, ...]


def resolve_char_budget(
    *,
    max_chars: int | None,
    max_tokens: int | None,
    chars_per_token: int,
) -> int | None:
    token_budget = max_tokens * chars_per_token if max_tokens is not None else None
    if max_chars is None:
        return token_budget
    if token_budget is None:
        return max_chars
    return min(max_chars, token_budget)


def base_source_id(result: SearchResult) -> str:
    qualifiers: list[str] = []
    if result.chunk_index is not None:
        qualifiers.append(f"chunk-{result.chunk_index}")
    if result.figure_id and _has_reliable_positional_metadata(result):
        qualifiers.append(f"figure-{stable_source_fragment(result.figure_id)}")
    elif result.result_type and result.result_type != SEARCH_RESULT_TYPE_TEXT:
        qualifiers.append(f"type-{stable_source_fragment(result.result_type)}")
    elif result.chunk_index is None:
        if result.section_id:
            qualifiers.append(f"section-{stable_source_fragment(result.section_id)}")
        elif result.section_path:
            qualifiers.append(f"section-{_short_hash(result.section_path)}")
    suffix = "".join(f"#{qualifier}" for qualifier in qualifiers)
    if result.document_id:
        return f"{_scope_prefix(result)}{result.document_id}{suffix}"
    if result.document_key:
        return f"{_scope_prefix(result)}{result.document_key}{suffix}"
    return result.id


def source_dedupe_key(result: SearchResult) -> SourceDedupeKey:
    document_key = _document_dedupe_key(result)
    if document_key is None:
        return ("result_id", result.id)
    qualifier = _result_dedupe_qualifier(result)
    if result.chunk_index is not None:
        return (*document_key, "chunk", str(result.chunk_index), *qualifier)
    if result.section_id:
        return (*document_key, "section_id", result.section_id, *qualifier)
    if result.section_path:
        return (*document_key, "section_path", result.section_path, *qualifier)
    return ("result_id", result.id)


def _result_dedupe_qualifier(result: SearchResult) -> SourceDedupeKey:
    if result.figure_id and _has_reliable_positional_metadata(result):
        return ("figure", result.figure_id)
    if result.result_type and result.result_type != SEARCH_RESULT_TYPE_TEXT:
        return ("result_type", result.result_type)
    return ()


def _has_reliable_positional_metadata(result: SearchResult) -> bool:
    return result.metadata.get("offset_reconstruction") != "unreliable"


def unique_source_id(
    source_id: str,
    used_source_ids: set[str] | None,
    *,
    stable_suffix: str | None = None,
    require_stable_suffix: bool = False,
) -> str:
    if used_source_ids is None:
        return source_id
    if require_stable_suffix:
        if not stable_suffix:
            raise ValueError("stable_suffix is required for colliding source ids")
        unique = f"{source_id}-{_short_hash(stable_suffix)}"
        used_source_ids.add(unique)
        return unique
    if source_id not in used_source_ids:
        used_source_ids.add(source_id)
        return source_id
    if stable_suffix:
        unique = f"{source_id}-{_short_hash(stable_suffix)}"
        if unique not in used_source_ids:
            used_source_ids.add(unique)
            return unique
    index = 2
    while f"{source_id}-{index}" in used_source_ids:
        index += 1
    unique = f"{source_id}-{index}"
    used_source_ids.add(unique)
    return unique


def metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def metadata_str(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved or None


def metadata_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    bbox: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return None
        if not _finite_number(item):
            return None
        bbox.append(float(item))
    return (bbox[0], bbox[1], bbox[2], bbox[3])


def estimate_tokens(text: str, *, chars_per_token: int) -> int:
    if not text:
        return 0
    return ceil(len(text) / chars_per_token)


def retrieval_metadata_from_result(result: SearchResult) -> dict[str, object] | None:
    metadata: dict[str, object] = {}
    quality = _quality_metadata_from_result(result)
    if quality is not None:
        metadata["quality"] = quality
    rerank = result.metadata.get("rerank")
    if not isinstance(rerank, dict):
        return metadata or None
    payload: dict[str, object] = {}
    for key in ("provider", "model"):
        value = rerank.get(key)
        if isinstance(value, str) and value.strip():
            payload[key] = value
    for key in ("provider_score", "search_score"):
        value = rerank.get(key)
        if (
            not isinstance(value, bool)
            and isinstance(value, int | float)
            and math.isfinite(float(value))
        ):
            payload[key] = float(value)
    for key in ("original_rank", "rerank_rank", "rank_delta"):
        value = rerank.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            payload[key] = value
    if payload:
        metadata["rerank"] = payload
    return metadata or None


def _quality_metadata_from_result(result: SearchResult) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    for key in ("verdict", "details"):
        value = result.metadata.get(f"quality_{key}")
        if isinstance(value, str) and value.strip():
            payload[key] = value
    for key in ("char_count", "page_count"):
        value = result.metadata.get(f"quality_{key}")
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            payload[key] = value
    for key in ("meaningful_ratio", "mojibake_ratio", "text_to_page_ratio"):
        value = result.metadata.get(f"quality_{key}")
        if (
            not isinstance(value, bool)
            and isinstance(value, int | float)
            and math.isfinite(float(value))
        ):
            payload[key] = float(value)
    return payload or None


def stable_source_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return fragment[:64] or _short_hash(value)


def drop_none(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if value is not None}


def _document_dedupe_key(result: SearchResult) -> tuple[str, str, str, str] | None:
    namespace = result.namespace or ""
    if result.document_id:
        return ("document_id", namespace, result.collection or "", result.document_id)
    if result.document_key and result.collection:
        return ("document_key", namespace, result.collection, result.document_key)
    return None


def _scope_prefix(result: SearchResult) -> str:
    parts: list[str] = []
    if result.namespace:
        parts.append(result.namespace)
    if result.collection:
        parts.append(result.collection)
    if not parts:
        return ""
    return ":".join(parts) + ":"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _finite_number(value: object) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return (
        not isinstance(value, bool)
        and isinstance(value, float)
        and math.isfinite(value)
    )
