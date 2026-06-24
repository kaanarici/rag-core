from __future__ import annotations

import mimetypes
import os

from .converter_keys import (
    CODE_CONVERTER_KEY,
    CSV_CONVERTER_KEY,
    DOCX_CONVERTER_KEY,
    HTML_CONVERTER_KEY,
    IMAGE_CONVERTER_KEY,
    JSON_CONVERTER_KEY,
    PDF_CONVERTER_KEY,
    PPTX_CONVERTER_KEY,
    TEXT_CONVERTER_KEY,
    XLSX_CONVERTER_KEY,
    XML_CONVERTER_KEY,
)


mimetypes.add_type("application/x-ndjson", ".ndjson")

JSONL_MIME_TYPES = (
    "application/jsonl",
    "application/jsonlines",
    "application/ldjson",
    "application/x-ldjson",
)
NDJSON_MIME_TYPES = (
    "application/ndjson",
    "application/x-ndjson",
)


MIME_TYPE_MAP: dict[str, str] = {
    "application/pdf": PDF_CONVERTER_KEY,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCX_CONVERTER_KEY,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PPTX_CONVERTER_KEY,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XLSX_CONVERTER_KEY,
    "text/html": HTML_CONVERTER_KEY,
    "application/xhtml+xml": HTML_CONVERTER_KEY,
    "text/csv": CSV_CONVERTER_KEY,
    "text/tab-separated-values": CSV_CONVERTER_KEY,
    "application/json": JSON_CONVERTER_KEY,
    **{mime_type: JSON_CONVERTER_KEY for mime_type in JSONL_MIME_TYPES},
    **{mime_type: JSON_CONVERTER_KEY for mime_type in NDJSON_MIME_TYPES},
    "application/xml": XML_CONVERTER_KEY,
    "text/xml": XML_CONVERTER_KEY,
    "text/plain": TEXT_CONVERTER_KEY,
    "text/markdown": TEXT_CONVERTER_KEY,
    "text/x-markdown": TEXT_CONVERTER_KEY,
    "text/yaml": TEXT_CONVERTER_KEY,
    "text/x-yaml": TEXT_CONVERTER_KEY,
    "application/x-yaml": TEXT_CONVERTER_KEY,
    "application/toml": TEXT_CONVERTER_KEY,
    "image/jpeg": IMAGE_CONVERTER_KEY,
    "image/png": IMAGE_CONVERTER_KEY,
    "image/gif": IMAGE_CONVERTER_KEY,
    "image/webp": IMAGE_CONVERTER_KEY,
    "image/bmp": IMAGE_CONVERTER_KEY,
    "image/tiff": IMAGE_CONVERTER_KEY,
    "image/svg+xml": TEXT_CONVERTER_KEY,
}

EXTENSION_MAP: dict[str, str] = {
    ".pdf": PDF_CONVERTER_KEY,
    ".docx": DOCX_CONVERTER_KEY,
    ".pptx": PPTX_CONVERTER_KEY,
    ".xlsx": XLSX_CONVERTER_KEY,
    ".html": HTML_CONVERTER_KEY,
    ".htm": HTML_CONVERTER_KEY,
    ".csv": CSV_CONVERTER_KEY,
    ".tsv": CSV_CONVERTER_KEY,
    ".json": JSON_CONVERTER_KEY,
    ".jsonl": JSON_CONVERTER_KEY,
    ".ndjson": JSON_CONVERTER_KEY,
    ".xml": XML_CONVERTER_KEY,
    ".txt": TEXT_CONVERTER_KEY,
    ".md": TEXT_CONVERTER_KEY,
    ".markdown": TEXT_CONVERTER_KEY,
    ".yaml": TEXT_CONVERTER_KEY,
    ".yml": TEXT_CONVERTER_KEY,
    ".toml": TEXT_CONVERTER_KEY,
    ".rst": TEXT_CONVERTER_KEY,
    ".adoc": TEXT_CONVERTER_KEY,
    ".tex": TEXT_CONVERTER_KEY,
    ".ini": TEXT_CONVERTER_KEY,
    ".cfg": TEXT_CONVERTER_KEY,
    ".conf": TEXT_CONVERTER_KEY,
    ".env": TEXT_CONVERTER_KEY,
    ".properties": TEXT_CONVERTER_KEY,
    ".log": TEXT_CONVERTER_KEY,
    ".py": CODE_CONVERTER_KEY,
    ".js": CODE_CONVERTER_KEY,
    ".ts": CODE_CONVERTER_KEY,
    ".tsx": CODE_CONVERTER_KEY,
    ".jsx": CODE_CONVERTER_KEY,
    ".java": CODE_CONVERTER_KEY,
    ".c": CODE_CONVERTER_KEY,
    ".cpp": CODE_CONVERTER_KEY,
    ".cc": CODE_CONVERTER_KEY,
    ".cxx": CODE_CONVERTER_KEY,
    ".h": CODE_CONVERTER_KEY,
    ".hpp": CODE_CONVERTER_KEY,
    ".m": CODE_CONVERTER_KEY,
    ".cs": CODE_CONVERTER_KEY,
    ".go": CODE_CONVERTER_KEY,
    ".rs": CODE_CONVERTER_KEY,
    ".rb": CODE_CONVERTER_KEY,
    ".php": CODE_CONVERTER_KEY,
    ".swift": CODE_CONVERTER_KEY,
    ".kt": CODE_CONVERTER_KEY,
    ".kts": CODE_CONVERTER_KEY,
    ".scala": CODE_CONVERTER_KEY,
    ".d": CODE_CONVERTER_KEY,
    ".jl": CODE_CONVERTER_KEY,
    ".ex": CODE_CONVERTER_KEY,
    ".exs": CODE_CONVERTER_KEY,
    ".erl": CODE_CONVERTER_KEY,
    ".clj": CODE_CONVERTER_KEY,
    ".groovy": CODE_CONVERTER_KEY,
    ".dart": CODE_CONVERTER_KEY,
    ".hs": CODE_CONVERTER_KEY,
    ".ml": CODE_CONVERTER_KEY,
    ".fs": CODE_CONVERTER_KEY,
    ".nim": CODE_CONVERTER_KEY,
    ".cr": CODE_CONVERTER_KEY,
    ".zig": CODE_CONVERTER_KEY,
    ".lua": CODE_CONVERTER_KEY,
    ".pl": CODE_CONVERTER_KEY,
    ".r": CODE_CONVERTER_KEY,
    ".sh": CODE_CONVERTER_KEY,
    ".bash": CODE_CONVERTER_KEY,
    ".zsh": CODE_CONVERTER_KEY,
    ".ps1": CODE_CONVERTER_KEY,
    ".bat": CODE_CONVERTER_KEY,
    ".cmd": CODE_CONVERTER_KEY,
    ".sql": CODE_CONVERTER_KEY,
    ".graphql": CODE_CONVERTER_KEY,
    ".gql": CODE_CONVERTER_KEY,
    ".proto": CODE_CONVERTER_KEY,
    ".tf": CODE_CONVERTER_KEY,
    ".tfvars": CODE_CONVERTER_KEY,
    ".hcl": CODE_CONVERTER_KEY,
    ".gradle": CODE_CONVERTER_KEY,
    ".cmake": CODE_CONVERTER_KEY,
    ".make": CODE_CONVERTER_KEY,
    ".mak": CODE_CONVERTER_KEY,
    ".css": CODE_CONVERTER_KEY,
    ".scss": CODE_CONVERTER_KEY,
    ".sass": CODE_CONVERTER_KEY,
    ".less": CODE_CONVERTER_KEY,
    ".vue": CODE_CONVERTER_KEY,
    ".svelte": CODE_CONVERTER_KEY,
    ".jpg": IMAGE_CONVERTER_KEY,
    ".jpeg": IMAGE_CONVERTER_KEY,
    ".png": IMAGE_CONVERTER_KEY,
    ".gif": IMAGE_CONVERTER_KEY,
    ".webp": IMAGE_CONVERTER_KEY,
    ".bmp": IMAGE_CONVERTER_KEY,
    ".tiff": IMAGE_CONVERTER_KEY,
    ".tif": IMAGE_CONVERTER_KEY,
}


def registered_converter_key(*, filename: str, mime_type: str) -> str | None:
    mt = (mime_type or "").strip().lower()
    _, ext = os.path.splitext((filename or "").lower())
    return MIME_TYPE_MAP.get(mt) or EXTENSION_MAP.get(ext)


def is_registered_image_document(*, filename: str, mime_type: str) -> bool:
    return (
        registered_converter_key(filename=filename, mime_type=mime_type)
        == IMAGE_CONVERTER_KEY
    )
