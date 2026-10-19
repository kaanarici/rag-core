"""One format table for parser selection, folder scans, and format documentation."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import mimetypes


@dataclass(frozen=True)
class FormatSupport:
    key: str
    label: str
    maturity: str
    converter_key: str
    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]
    local_ingest: bool
    ocr: str
    notes: str
    module_name: str
    class_name: str


FORMAT_SUPPORT_MATRIX = (
    FormatSupport(
        "text",
        "Text and markdown",
        "stable",
        "text",
        tuple(
            ".txt .md .markdown .yaml .yml .toml .rst .adoc .tex .ini .cfg .conf .env .properties .log".split()
        ),
        (
            "text/plain",
            "text/markdown",
            "text/x-markdown",
            "text/yaml",
            "text/x-yaml",
            "application/x-yaml",
            "application/toml",
            "image/svg+xml",
        ),
        True,
        "none",
        "Plain text, markdown, YAML, TOML, logs, config, and similar text files.",
        ".text_converter",
        "TextConverter",
    ),
    FormatSupport(
        "code",
        "Source code",
        "stable",
        "code",
        tuple(
            ".py .js .ts .tsx .jsx .java .c .cpp .cc .cxx .h .hpp .m .cs .go .rs .rb .php .swift .kt .kts .scala .d .jl .ex .exs .erl .clj .groovy .dart .hs .ml .fs .nim .cr .zig .lua .pl .r .sh .bash .zsh .ps1 .bat .cmd .sql .graphql .gql .proto .tf .tfvars .hcl .gradle .cmake .make .mak .css .scss .sass .less .vue .svelte".split()
        ),
        (),
        True,
        "none",
        "Text source files with language metadata for downstream chunking.",
        ".code_converter",
        "CodeConverter",
    ),
    FormatSupport(
        "html",
        "HTML",
        "stable",
        "html",
        tuple(".html .htm".split()),
        ("text/html", "application/xhtml+xml"),
        True,
        "none",
        "Local HTML extraction with markdown fallbacks; browser rendering is not part of this path.",
        ".html_converter",
        "HtmlConverter",
    ),
    FormatSupport(
        "csv",
        "CSV and TSV",
        "stable",
        "csv",
        tuple(".csv .tsv".split()),
        ("text/csv", "text/tab-separated-values"),
        True,
        "none",
        "Delimiter detection and markdown-table rendering.",
        ".csv_converter",
        "CsvConverter",
    ),
    FormatSupport(
        "json",
        "JSON and JSONL",
        "stable",
        "json",
        tuple(".json .jsonl .ndjson".split()),
        (
            "application/json",
            "application/jsonl",
            "application/ndjson",
            "application/x-ndjson",
        ),
        True,
        "none",
        "Pretty-printed structured text for indexing.",
        ".json_converter",
        "JsonConverter",
    ),
    FormatSupport(
        "xml",
        "XML",
        "stable",
        "xml",
        tuple(".xml".split()),
        ("application/xml", "text/xml"),
        True,
        "none",
        "Defused XML parsing with safe text fallback.",
        ".xml_converter",
        "XmlConverter",
    ),
    FormatSupport(
        "pdf",
        "PDF",
        "beta",
        "pdf",
        tuple(".pdf".split()),
        ("application/pdf",),
        True,
        "optional_pages",
        "PyMuPDF text extraction, reading order, page citations, and selective OCR routing.",
        ".pdf_converter",
        "PdfConverter",
    ),
    FormatSupport(
        "docx",
        "DOCX",
        "beta",
        "docx",
        tuple(".docx".split()),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
        True,
        "quality_fallback",
        "OOXML DOCX only; .doc is intentionally unsupported.",
        ".docx_converter",
        "DocxConverter",
    ),
    FormatSupport(
        "pptx",
        "PPTX",
        "beta",
        "pptx",
        tuple(".pptx".split()),
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation",),
        True,
        "quality_fallback",
        "OOXML PPTX only; .ppt is intentionally unsupported.",
        ".pptx_converter",
        "PptxConverter",
    ),
    FormatSupport(
        "xlsx",
        "XLSX",
        "beta",
        "xlsx",
        tuple(".xlsx".split()),
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
        True,
        "none",
        "OOXML XLSX only; .xls is intentionally unsupported.",
        ".xlsx_converter",
        "XlsxConverter",
    ),
    FormatSupport(
        "image",
        "Images",
        "beta",
        "image",
        tuple(".jpg .jpeg .png .gif .webp .bmp .tiff .tif".split()),
        (
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff",
        ),
        False,
        "required",
        "Images require an injected OCR provider; default local ingest excludes them.",
        ".image_converter",
        "ImageConverter",
    ),
)
FORMATS = {f.key: f for f in FORMAT_SUPPORT_MATRIX}
EXTENSION_MAP = {ext: f.key for f in FORMAT_SUPPORT_MATRIX for ext in f.extensions}
MIME_TYPE_MAP = {mime: f.key for f in FORMAT_SUPPORT_MATRIX for mime in f.mime_types}
JSONL_MIME_TYPES = (
    "application/jsonl",
    "application/jsonlines",
    "application/ldjson",
    "application/x-ldjson",
)
NDJSON_MIME_TYPES = ("application/ndjson", "application/x-ndjson")
MIME_TYPE_MAP.update(dict.fromkeys(JSONL_MIME_TYPES, "json"))
mimetypes.add_type("application/x-ndjson", ".ndjson")
UNSUPPORTED_BINARY_EXTENSIONS = frozenset({".doc", ".ppt", ".xls"})
UNSUPPORTED_BINARY_MIME_TYPES = frozenset(
    {"application/msword", "application/vnd.ms-powerpoint", "application/vnd.ms-excel"}
)


def normalize_extension(value: str) -> str:
    value = value.lower().strip()
    return f".{value}" if value and not value.startswith(".") else value


def normalize_mime_type(value: str) -> str:
    return value.lower().split(";", 1)[0].strip()


def registered_converter_key(*, filename: str, mime_type: str) -> str | None:
    mime = MIME_TYPE_MAP.get(normalize_mime_type(mime_type))
    ext = EXTENSION_MAP.get(Path(filename.strip()).suffix.lower())
    return ext if ext and mime in {None, "text"} else mime


def is_registered_image_document(*, filename: str, mime_type: str) -> bool:
    return registered_converter_key(filename=filename, mime_type=mime_type) == "image"


def is_registered_pdf_document(*, filename: str | None, mime_type: str | None) -> bool:
    return (
        normalize_mime_type(mime_type or "") == "application/pdf"
        or Path((filename or "").strip()).suffix.lower() == ".pdf"
    )


def format_support_for_extension(extension: str) -> FormatSupport | None:
    key = EXTENSION_MAP.get(normalize_extension(extension))
    return FORMATS.get(key) if key else None


def format_support_for_mime_type(mime_type: str) -> FormatSupport | None:
    mime = normalize_mime_type(mime_type)
    key = MIME_TYPE_MAP.get(mime)
    result = FORMATS.get(key) if key else None
    preferred = (
        ".jsonl"
        if mime in JSONL_MIME_TYPES
        else ".ndjson"
        if mime in NDJSON_MIME_TYPES
        else None
    )
    return (
        replace(
            result,
            extensions=(
                preferred,
                *(ext for ext in result.extensions if ext != preferred),
            ),
        )
        if result and preferred
        else result
    )


def is_local_ingest_extension(extension: str) -> bool:
    row = format_support_for_extension(extension)
    return bool(row and row.local_ingest)


def is_unsupported_binary_extension(extension: str) -> bool:
    return normalize_extension(extension) in UNSUPPORTED_BINARY_EXTENSIONS


def is_unsupported_binary_mime_type(mime_type: str) -> bool:
    return normalize_mime_type(mime_type) in UNSUPPORTED_BINARY_MIME_TYPES


def local_ingest_format_keys() -> tuple[str, ...]:
    return tuple(row.key for row in FORMAT_SUPPORT_MATRIX if row.local_ingest)


def format_support_payloads() -> list[dict[str, object]]:
    keys = (
        "key",
        "label",
        "maturity",
        "converter_key",
        "extensions",
        "mime_types",
        "local_ingest",
        "ocr",
        "notes",
    )
    return [
        {
            key: list(getattr(row, key))
            if key in {"extensions", "mime_types"}
            else getattr(row, key)
            for key in keys
        }
        for row in FORMAT_SUPPORT_MATRIX
    ]


def unsupported_local_file_message(path: Path, *, label: str = "path") -> str:
    support = format_support_for_extension(path.suffix)
    reason = (
        support.notes
        if support and support.key == "image"
        else f"unsupported extension {path.suffix.lower() or 'no extension'!r}"
    )
    return f"no supported file matched {str(path)!r} for {label}: {reason}. Default local ingest supports: {', '.join(local_ingest_format_keys())}. See https://github.com/kaanarici/rag-core/blob/main/docs/formats.md."
