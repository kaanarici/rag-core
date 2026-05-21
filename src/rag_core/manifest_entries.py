from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_core.core_models import CorpusManifestEntry
from rag_core.manifest_paths import manifest_segment

_ENTRY_KEYS = frozenset(
    {
        "document_id",
        "namespace",
        "corpus_id",
        "document_key",
        "content_sha256",
        "filename",
        "mime_type",
        "chunk_count",
        "parser",
        "needs_ocr",
        "metadata",
    }
)

_UNSAFE_METADATA_KEYS = frozenset(
    {
        "converter_error",
        "error",
        "error_message",
        "exception",
        "exception_message",
        "parse_error",
        "parser_error",
        "provider_error",
        "quality_details",
        "raw_error",
        "stack_trace",
        "stacktrace",
        "traceback",
    }
)
_MAX_MANIFEST_PAGE_INDICES = 400
_PAGE_INDEX_METADATA_KEYS = frozenset(
    {
        "complex_ocr_page_indices",
        "image_only_page_indices",
        "ocr_page_indices",
        "ocr_page_indices_telemetry",
    }
)


class _UnsafeManifestMetadataError(ValueError):
    pass


class ManifestReadError(ValueError):
    def __init__(self, path: Path, line_number: int, reason: str) -> None:
        self.path = path
        self.line_number = line_number
        self.reason = reason
        super().__init__(
            f"invalid manifest entry in {path} at line {line_number}: {reason}"
        )


def entry_to_dict(entry: CorpusManifestEntry) -> dict[str, Any]:
    payload = asdict(entry)
    payload["metadata"] = sanitize_manifest_metadata(entry.metadata)
    return payload


def entry_from_json_line(
    path: Path,
    line_number: int,
    line: str,
) -> CorpusManifestEntry:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        raise ManifestReadError(path, line_number, "invalid_json") from None
    if not isinstance(payload, dict):
        raise ManifestReadError(path, line_number, "entry_must_be_object") from None
    try:
        return _entry_from_dict(payload)
    except _UnsafeManifestMetadataError:
        raise ManifestReadError(path, line_number, "unsafe_metadata") from None
    except (KeyError, TypeError, ValueError):
        raise ManifestReadError(path, line_number, "invalid_entry") from None


def _entry_from_dict(payload: dict[str, Any]) -> CorpusManifestEntry:
    missing_keys = _ENTRY_KEYS.difference(payload)
    extra_keys = payload.keys() - _ENTRY_KEYS
    if missing_keys or extra_keys:
        raise ValueError("manifest entry fields must match the canonical shape")
    return CorpusManifestEntry(
        document_id=_entry_required_str(payload, "document_id"),
        namespace=manifest_segment(
            "namespace", _entry_required_str(payload, "namespace")
        ),
        corpus_id=manifest_segment(
            "corpus_id", _entry_required_str(payload, "corpus_id")
        ),
        document_key=_entry_optional_str(payload, "document_key"),
        content_sha256=_entry_optional_str(payload, "content_sha256"),
        filename=_entry_required_str(payload, "filename"),
        mime_type=_entry_required_str(payload, "mime_type"),
        chunk_count=_entry_int(payload, "chunk_count"),
        parser=_entry_optional_str(payload, "parser"),
        needs_ocr=_entry_bool(payload, "needs_ocr"),
        metadata=_entry_metadata(payload),
    )


def _entry_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _entry_optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _entry_int(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return int(value)


def _entry_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _entry_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload["metadata"]
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    metadata: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if _is_unsafe_metadata_key(key):
            raise _UnsafeManifestMetadataError(
                "metadata contains unsafe manifest field"
            )
        metadata[key] = _entry_metadata_value(item)
    return metadata


def sanitize_manifest_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if _is_unsafe_metadata_key(key):
            continue
        if key in _PAGE_INDEX_METADATA_KEYS and isinstance(value, list):
            sanitized[key] = _sanitize_page_indices(value)
            continue
        sanitized[key] = _sanitize_manifest_metadata_value(value)
    return sanitized


def _sanitize_page_indices(value: list[Any]) -> list[int]:
    sanitized: list[int] = []
    seen: set[int] = set()
    for raw_index in value:
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or raw_index in seen
        ):
            continue
        sanitized.append(raw_index)
        seen.add(raw_index)
        if len(sanitized) >= _MAX_MANIFEST_PAGE_INDICES:
            break
    return sanitized


def _sanitize_manifest_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_manifest_metadata_value(item)
            for key, item in value.items()
            if isinstance(key, str) and not _is_unsafe_metadata_key(key)
        }
    if isinstance(value, list):
        return [_sanitize_manifest_metadata_value(item) for item in value]
    return value


def _entry_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        metadata: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if _is_unsafe_metadata_key(key):
                raise _UnsafeManifestMetadataError(
                    "metadata contains unsafe manifest field"
                )
            metadata[key] = _entry_metadata_value(item)
        return metadata
    if isinstance(value, list):
        return [_entry_metadata_value(item) for item in value]
    return value


def _is_unsafe_metadata_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in _UNSAFE_METADATA_KEYS


__all__ = [
    "ManifestReadError",
    "entry_from_json_line",
    "entry_to_dict",
    "sanitize_manifest_metadata",
]
