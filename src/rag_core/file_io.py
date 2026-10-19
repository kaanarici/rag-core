from __future__ import annotations

from rag_core._async import run_sync

import hashlib
import mimetypes
from pathlib import Path

from rag_core.documents.formats import format_support_for_extension


def validate_document_size(data: bytes | bytearray, *, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise ValueError(f"document exceeds max_document_bytes ({max_bytes})")


async def read_file_bytes(path: Path, *, max_bytes: int | None = None) -> bytes:
    def read() -> bytes:
        with path.open("rb") as source:
            if max_bytes is None:
                return source.read()
            data = bytearray()
            while True:
                chunk = source.read(min(64 * 1024, max_bytes + 1 - len(data)))
                if not chunk:
                    return bytes(data)
                data.extend(chunk)
                validate_document_size(data, max_bytes=max_bytes)

    return await run_sync(read)


def compute_content_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


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
