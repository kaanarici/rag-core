"""Current document-format support metadata."""

from __future__ import annotations

from pathlib import Path

from .converter_keys import IMAGE_CONVERTER_KEY
from .format_support_models import (
    FormatSupport,
    UNSUPPORTED_BINARY_EXTENSIONS,
    UNSUPPORTED_BINARY_MIME_TYPES,
    normalize_extension,
    normalize_mime_type,
)
from .format_support_lookup import (
    support_by_extension,
    support_by_mime_type,
)
from .format_support_matrix import FORMAT_SUPPORT_MATRIX

_SUPPORT_BY_EXTENSION = support_by_extension(FORMAT_SUPPORT_MATRIX)
_SUPPORT_BY_MIME_TYPE = support_by_mime_type(FORMAT_SUPPORT_MATRIX)


def format_support_for_extension(extension: str) -> FormatSupport | None:
    return _SUPPORT_BY_EXTENSION.get(normalize_extension(extension))


def format_support_for_mime_type(mime_type: str) -> FormatSupport | None:
    return _SUPPORT_BY_MIME_TYPE.get(normalize_mime_type(mime_type))


def is_local_ingest_extension(extension: str) -> bool:
    support = format_support_for_extension(extension)
    return bool(support and support.local_ingest)


def is_unsupported_binary_extension(extension: str) -> bool:
    return normalize_extension(extension) in UNSUPPORTED_BINARY_EXTENSIONS


def is_unsupported_binary_mime_type(mime_type: str) -> bool:
    return normalize_mime_type(mime_type) in UNSUPPORTED_BINARY_MIME_TYPES


def local_ingest_format_keys() -> tuple[str, ...]:
    return tuple(entry.key for entry in FORMAT_SUPPORT_MATRIX if entry.local_ingest)


def unsupported_local_file_message(path: Path, *, label: str = "path") -> str:
    support = format_support_for_extension(path.suffix)
    if support is not None and support.key == IMAGE_CONVERTER_KEY:
        reason = support.notes
    else:
        suffix = path.suffix.lower() or "no extension"
        reason = f"unsupported extension {suffix!r}"
    supported = ", ".join(local_ingest_format_keys())
    return (
        f"no supported file matched {str(path)!r} for {label}: {reason}. "
        f"Default local ingest supports: {supported}. "
        "See docs/parsing/formats.md."
    )
