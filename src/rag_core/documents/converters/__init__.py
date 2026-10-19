"""Select and lazily construct only the parser required by the current document."""

from __future__ import annotations
from functools import lru_cache
from rag_core.config.env_access import get_env_int
from importlib import import_module
from pathlib import Path
import logging
from .base import BaseConverter, ConversionResult, QualityVerdict
from rag_core.documents.formats import (
    FORMATS,
    is_unsupported_binary_extension,
    is_unsupported_binary_mime_type,
    registered_converter_key,
)
from rag_core.documents.exception_names import root_exception_type

logger = logging.getLogger(__name__)


def converter_settings(key: str) -> dict[str, int]:
    settings = {
        "csv": (("max_rows", "LOCAL_PARSE_CSV_MAX_ROWS", 1000),),
        "xlsx": (
            ("max_rows", "LOCAL_PARSE_XLSX_MAX_ROWS", 500),
            ("max_total_rows", "LOCAL_PARSE_XLSX_MAX_TOTAL_ROWS", 5000),
            ("max_cols", "LOCAL_PARSE_XLSX_MAX_COLS", 50),
        ),
    }
    return {
        name: max(1, get_env_int(env, default))
        for name, env, default in settings.get(key, ())
    }


@lru_cache(maxsize=32)
def _converter(key: str, settings: tuple[tuple[str, int], ...] = ()) -> BaseConverter:
    spec = FORMATS[key]
    try:
        cls = getattr(import_module(spec.module_name, __package__), spec.class_name)
        if not isinstance(cls, type) or not issubclass(cls, BaseConverter):
            raise TypeError("parser must be a BaseConverter")
        return cls(**dict(settings))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize {key} converter (error_type={root_exception_type(exc)})"
        ) from None


def get_converter(*, mime_type: str = "", filename: str = "") -> BaseConverter:
    ext = Path(filename.strip()).suffix.lower()
    if is_unsupported_binary_extension(ext):
        raise ValueError(f"Unsupported format: extension {ext!r}")
    if is_unsupported_binary_mime_type(mime_type):
        raise ValueError(f"Unsupported format: MIME type {mime_type!r}")
    key = registered_converter_key(filename=filename, mime_type=mime_type) or "text"
    return _converter(key, tuple(sorted(converter_settings(key).items())))


async def convert_file(
    file_bytes: bytes, filename: str, mime_type: str
) -> ConversionResult:
    parser = get_converter(mime_type=mime_type, filename=filename)
    result = await parser.convert(file_bytes, filename, mime_type)
    logger.info(
        "Converted with converter=%s, chars=%d, needs_ocr=%s",
        parser.format_name,
        len(result.content),
        result.needs_ocr,
    )
    return result


__all__ = [
    "BaseConverter",
    "ConversionResult",
    "QualityVerdict",
    "convert_file",
    "get_converter",
]
