from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

from rag_core.documents.converters.format_support import format_support_for_extension


async def read_file_bytes(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)


def detect_local_mime_type(path: Path) -> str:
    return detect_mime_type_for_name(str(path))


def detect_mime_type_for_name(name: str) -> str:
    mime_type, _ = mimetypes.guess_type(name)
    suffix = Path(name).suffix
    support = format_support_for_extension(suffix)
    if (
        support is not None
        and support.key == "code"
        and suffix.lower() in {".ts", ".tsx"}
    ):
        return _code_mime_type(suffix)
    if (
        support is not None
        and support.key == "code"
        and not _is_text_like_mime_type(mime_type)
    ):
        return _code_mime_type(suffix)
    return mime_type or "application/octet-stream"


def _is_text_like_mime_type(mime_type: str | None) -> bool:
    if mime_type is None:
        return False
    return mime_type.startswith("text/") or mime_type in {
        "application/javascript",
        "application/typescript",
    }


def _code_mime_type(extension: str) -> str:
    resolved = extension.lower().strip()
    if resolved in {".ts", ".tsx"}:
        return "text/typescript"
    return "text/x-source"
