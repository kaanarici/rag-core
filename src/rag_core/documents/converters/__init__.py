"""Document converter registry with MIME type and extension lookup."""

from __future__ import annotations

import logging
import os

from .base import BaseConverter, ConversionResult, QualityVerdict
from .converter_keys import (
    DOCX_CONVERTER_KEY,
    IMAGE_CONVERTER_KEY,
    PDF_CONVERTER_KEY,
    PPTX_CONVERTER_KEY,
    TEXT_CONVERTER_KEY,
    XLSX_CONVERTER_KEY,
)
from .format_support import (
    is_unsupported_binary_extension,
    is_unsupported_binary_mime_type,
)
from .registry_loader import get_registered_converters
from .registry_maps import EXTENSION_MAP, MIME_TYPE_MAP

logger = logging.getLogger(__name__)

_STRICT_MAPPED_KEYS = frozenset(
    {
        PDF_CONVERTER_KEY,
        DOCX_CONVERTER_KEY,
        PPTX_CONVERTER_KEY,
        XLSX_CONVERTER_KEY,
        IMAGE_CONVERTER_KEY,
    }
)


def _resolve_text_fallback(converters: dict[str, BaseConverter]) -> BaseConverter:
    text_converter = converters.get(TEXT_CONVERTER_KEY)
    if text_converter is None:
        raise RuntimeError(
            "Text converter is required for fallback resolution but is unavailable"
        )
    return text_converter


def _resolve_mapped_converter_or_none(
    *,
    converters: dict[str, BaseConverter],
    converter_key: str | None,
    mime_type: str,
    filename: str,
) -> BaseConverter | None:
    if not converter_key:
        return None
    converter = converters.get(converter_key)
    if converter is not None:
        return converter
    if converter_key in _STRICT_MAPPED_KEYS:
        raise RuntimeError(
            "Converter %r is mapped for %r but unavailable"
            % (converter_key, filename or mime_type)
        )
    return None


def get_converter(
    *,
    mime_type: str = "",
    filename: str = "",
) -> BaseConverter:
    """Get the appropriate converter for a file.

    Resolution order:
    1. MIME type mapping
    2. File extension mapping
    3. Fallback to text converter (for text/* MIME types)
    4. Fallback to text converter (unknown types try text extraction)

    Args:
        mime_type: MIME type of the file.
        filename: Original filename for extension detection.

    Returns:
        A BaseConverter instance for the file type.
    """
    converters = get_registered_converters()
    mt = (mime_type or "").strip().lower()
    _, ext = os.path.splitext((filename or "").lower())
    if is_unsupported_binary_extension(ext):
        raise ValueError(
            f"Unsupported format for {filename or mime_type!r}: extension {ext!r}"
        )
    if is_unsupported_binary_mime_type(mt):
        raise ValueError(
            f"Unsupported format for {filename or mime_type!r}: MIME type {mt!r}"
        )

    mime_key = MIME_TYPE_MAP.get(mt)
    extension_key = EXTENSION_MAP.get(ext)
    if extension_key and (
        mime_key is None
        or (mime_key == TEXT_CONVERTER_KEY and extension_key != TEXT_CONVERTER_KEY)
    ):
        extension_converter = _resolve_mapped_converter_or_none(
            converters=converters,
            converter_key=extension_key,
            mime_type=mt,
            filename=filename,
        )
        if extension_converter is not None:
            return extension_converter

    mapped_converter = _resolve_mapped_converter_or_none(
        converters=converters,
        converter_key=mime_key,
        mime_type=mt,
        filename=filename,
    )
    if mapped_converter is not None:
        return mapped_converter

    text_fallback = _resolve_text_fallback(converters)
    if mt.startswith("text/"):
        return text_fallback

    logger.debug(
        "No specific converter for mime=%s ext=%s, using text fallback",
        mt,
        ext,
    )
    return text_fallback


async def convert_file(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> ConversionResult:
    """Convert a file with the registered converter for its type."""
    converter = get_converter(mime_type=mime_type, filename=filename)
    logger.debug(
        "Using %s converter",
        converter.format_name,
    )

    result = await converter.convert(file_bytes, filename, mime_type)

    if result.quality:
        logger.info(
            "Converted with converter=%s, quality=%s, chars=%d, needs_ocr=%s",
            converter.format_name,
            result.quality.verdict.value,
            result.quality.char_count,
            result.needs_ocr,
        )
    else:
        logger.info(
            "Converted with converter=%s, needs_ocr=%s",
            converter.format_name,
            result.needs_ocr,
        )

    return result


__all__ = (
    "BaseConverter",
    "ConversionResult",
    "QualityVerdict",
    "convert_file",
    "get_converter",
)
