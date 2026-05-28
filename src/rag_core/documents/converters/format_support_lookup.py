"""Lookup helpers for document-format support metadata."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from .format_support_models import (
    FormatSupport,
    normalize_extension,
    normalize_mime_type,
)
from .converter_keys import JSON_CONVERTER_KEY
from .registry_maps import (
    EXTENSION_MAP,
    JSONL_MIME_TYPES,
    MIME_TYPE_MAP,
    NDJSON_MIME_TYPES,
)

_JSONL_ALIAS_MIME_TYPES = frozenset(
    JSONL_MIME_TYPES[1:],
)
_PREFERRED_MIME_EXTENSIONS = {
    **{mime_type: ".jsonl" for mime_type in JSONL_MIME_TYPES},
    **{mime_type: ".ndjson" for mime_type in NDJSON_MIME_TYPES},
}


def extensions_for_converter(converter_key: str) -> tuple[str, ...]:
    return tuple(
        extension
        for extension, mapped_key in EXTENSION_MAP.items()
        if mapped_key == converter_key
    )


def mime_types_for_converter(converter_key: str) -> tuple[str, ...]:
    return tuple(
        mime_type
        for mime_type, mapped_key in MIME_TYPE_MAP.items()
        if mapped_key == converter_key and mime_type not in _JSONL_ALIAS_MIME_TYPES
    )


def support_by_extension(
    entries: Iterable[FormatSupport],
) -> dict[str, FormatSupport]:
    return {
        normalize_extension(extension): entry
        for entry in entries
        for extension in entry.extensions
    }


def support_by_mime_type(
    entries: Iterable[FormatSupport],
) -> dict[str, FormatSupport]:
    support: dict[str, FormatSupport] = {}
    for entry in entries:
        for mime_type in entry.mime_types:
            support[normalize_mime_type(mime_type)] = _with_preferred_extension(
                entry,
                mime_type=mime_type,
            )
        if entry.converter_key == JSON_CONVERTER_KEY:
            for mime_type in _JSONL_ALIAS_MIME_TYPES:
                support[mime_type] = _with_preferred_extension(
                    entry,
                    mime_type=mime_type,
                )
    return support


def _with_preferred_extension(
    entry: FormatSupport,
    *,
    mime_type: str,
) -> FormatSupport:
    preferred = _PREFERRED_MIME_EXTENSIONS.get(normalize_mime_type(mime_type))
    if preferred is None or preferred not in entry.extensions:
        return entry
    extensions = (
        preferred,
        *tuple(ext for ext in entry.extensions if ext != preferred),
    )
    return replace(entry, extensions=extensions)
