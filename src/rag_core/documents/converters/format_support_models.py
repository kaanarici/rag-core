"""Shared model and normalization helpers for format-support metadata."""

from __future__ import annotations

from dataclasses import dataclass

UNSUPPORTED_BINARY_EXTENSIONS: frozenset[str] = frozenset({".doc", ".ppt", ".xls"})
UNSUPPORTED_BINARY_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/msword",
        "application/vnd.ms-powerpoint",
        "application/vnd.ms-excel",
    }
)


@dataclass(frozen=True)
class FormatSupport:
    key: str
    label: str
    support_level: str
    converter_key: str | None
    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]
    local_ingest: bool
    ocr: str
    notes: str


def normalize_extension(extension: str) -> str:
    resolved = extension.lower().strip()
    if resolved and not resolved.startswith("."):
        return f".{resolved}"
    return resolved


def normalize_mime_type(mime_type: str) -> str:
    return mime_type.lower().strip()
