from __future__ import annotations

import os
from pathlib import Path

import pytest

import rag_core.documents.converters as converters_module
from rag_core.file_io import detect_mime_type_for_name
from rag_core.documents.converters import get_converter
from rag_core.documents.converters.format_support import (
    FORMAT_SUPPORT_MATRIX,
    UNSUPPORTED_BINARY_MIME_TYPES,
    format_support_for_extension,
    format_support_for_mime_type,
)
from rag_core.documents.converters.registry_maps import EXTENSION_MAP, MIME_TYPE_MAP
from rag_core.documents.converters.registry_specs import CONVERTER_SPEC_BY_KEY
from rag_core.ingest.local import validate_supported_local_file
from rag_core.ingest.sources.remote import _extension_for_mime_type
from rag_core.ingest.sources.local import is_supported_local_file


def test_format_support_converter_keys_are_registered() -> None:
    for entry in FORMAT_SUPPORT_MATRIX:
        if entry.converter_key is None:
            continue
        assert entry.converter_key in CONVERTER_SPEC_BY_KEY


def test_format_support_covers_registry_maps() -> None:
    for extension, converter_key in EXTENSION_MAP.items():
        support = format_support_for_extension(extension)
        assert support is not None
        assert support.converter_key == converter_key

    for mime_type, converter_key in MIME_TYPE_MAP.items():
        support = format_support_for_mime_type(mime_type)
        assert support is not None
        assert support.converter_key == converter_key


def test_local_ingest_support_matches_format_matrix() -> None:
    for extension in EXTENSION_MAP:
        support = format_support_for_extension(extension)
        assert support is not None
        assert is_supported_local_file(Path(f"sample{extension}")) is support.local_ingest


BINARY_OFFICE_EXTENSIONS = (".doc", ".ppt", ".xls")


def test_binary_office_extensions_are_not_mapped() -> None:
    assert all(extension not in EXTENSION_MAP for extension in BINARY_OFFICE_EXTENSIONS)
    assert all(mime_type not in MIME_TYPE_MAP for mime_type in UNSUPPORTED_BINARY_MIME_TYPES)


@pytest.mark.parametrize("extension", BINARY_OFFICE_EXTENSIONS)
def test_binary_office_extension_rejects_converter_lookup(extension: str) -> None:
    assert format_support_for_extension(extension) is None
    assert is_supported_local_file(Path(f"binary-office{extension}")) is False
    with pytest.raises(ValueError, match="Unsupported format"):
        get_converter(filename=f"binary-office{extension}", mime_type="")


@pytest.mark.parametrize("mime_type", sorted(UNSUPPORTED_BINARY_MIME_TYPES))
def test_unsupported_binary_mime_type_rejects_converter_lookup(mime_type: str) -> None:
    assert format_support_for_mime_type(mime_type) is None
    with pytest.raises(ValueError, match="Unsupported format"):
        get_converter(filename="unknown.bin", mime_type=mime_type)


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [
        ("report.pdf", "application/pdf"),
        (
            "runbook.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("scan.png", "image/png"),
    ],
)
def test_known_mapped_binary_formats_do_not_fall_back_to_text_when_converter_missing(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    mime_type: str,
) -> None:
    text_converter = get_converter(filename="notes.txt", mime_type="text/plain")
    monkeypatch.setattr(
        converters_module,
        "get_registered_converters",
        lambda: {"text": text_converter},
    )

    with pytest.raises(RuntimeError, match="mapped.*unavailable"):
        converters_module.get_converter(filename=filename, mime_type=mime_type)


def test_validate_supported_local_file_rejects_hardlinked_path(tmp_path: Path) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    source = tmp_path / "source.md"
    source.write_text("secret", encoding="utf-8")
    alias = tmp_path / "alias.md"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink support unavailable: {exc}")

    with pytest.raises(ValueError, match="does not allow multi-link"):
        validate_supported_local_file(alias, label="ingest path")


def test_unsupported_local_file_message_is_actionable(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError) as image_error:
        validate_supported_local_file(image, label="ingest path")

    image_message = str(image_error.value)
    assert "Images require an injected OCR provider" in image_message
    assert "Default local ingest supports:" in image_message
    assert "https://kaanarici.github.io/rag-core/docs/formats" in image_message

    binary_office = tmp_path / "sample.xls"
    binary_office.write_bytes(b"not real xls")

    with pytest.raises(ValueError) as binary_office_error:
        validate_supported_local_file(binary_office, label="manifest path")

    binary_office_message = str(binary_office_error.value)
    assert "unsupported extension '.xls'" in binary_office_message
    assert "https://kaanarici.github.io/rag-core/docs/formats" in binary_office_message


def test_tsv_mime_support_without_extension_resolves_to_csv_converter() -> None:
    support = format_support_for_mime_type("text/tab-separated-values")
    assert support is not None
    assert support.converter_key == "csv"
    converter = get_converter(filename="scores", mime_type="text/tab-separated-values")
    assert converter.format_name == "csv"


def test_ndjson_extension_is_supported_like_ndjson_mime_types() -> None:
    support = format_support_for_extension(".ndjson")
    assert support is not None
    assert support.converter_key == "json"
    assert is_supported_local_file(Path("events.ndjson")) is True
    assert detect_mime_type_for_name("events.ndjson") == "application/x-ndjson"
    converter = get_converter(
        filename="events.ndjson",
        mime_type="application/octet-stream",
    )
    assert converter.format_name == "json"


@pytest.mark.parametrize(
    "mime_type",
    [
        "application/jsonlines",
        "application/ldjson",
        "application/x-ldjson",
    ],
)
def test_jsonl_alias_mime_types_route_to_json_converter(mime_type: str) -> None:
    support = format_support_for_mime_type(mime_type)
    assert support is not None
    assert support.converter_key == "json"
    assert support.extensions[0] == ".jsonl"
    assert _extension_for_mime_type(mime_type) == ".jsonl"

    converter = get_converter(filename="events", mime_type=mime_type)

    assert converter.format_name == "json"


def test_format_support_docs_cover_matrix() -> None:
    docs = Path("docs-site/content/docs/formats.mdx").read_text(encoding="utf-8")

    # Docs explain that converters emit parser metadata and that extraction
    # quality (not "parser quality") is observable in traces, payloads, and
    # context snippets.
    assert "converter registry" in docs
    assert "parser metadata under" in docs
    assert '`metadata["parser"]`' in docs
    assert "trace events" in docs
    assert "indexed\npayloads" in docs or "indexed payloads" in " ".join(docs.split())
    assert "Extraction quality is visible" in docs
    assert "Parser quality is visible" not in docs

    for entry in FORMAT_SUPPORT_MATRIX:
        assert f"| `{entry.key}` |" in docs
        for extension in entry.extensions:
            assert f"`{extension}`" in docs
        for mime_type in entry.mime_types:
            assert f"`{mime_type}`" in docs


def test_format_support_docs_name_known_gaps_and_locators() -> None:
    docs = Path("docs-site/content/docs/formats.mdx").read_text(encoding="utf-8")
    normalized = " ".join(docs.split())

    # Honesty guardrail: docs must name the real per-format gaps rather than
    # overclaiming uniform extraction quality.
    for phrase in (
        "What you can ingest",
        "PDF support is beta and document-shape dependent",
        "Text PDFs are searchable",
        "layout-faithful",
        "Dense filings can have word-joining",
        "damage",
        "Slide decks and chart-heavy PDFs are unreliable",
        "Mixed or scanned PDFs require OCR",
        "Images are recognized, but they are not default-ingestable without OCR",
        "HTML extraction reads static file text",
        "Source code means plain text source",
        "XLSX formulas are emitted as text",
        "PPTX keeps text",
        "rag_core.documents.converters.convert_file",
        "binary-like",
    ):
        assert phrase in docs

    # Locators are documented by the structured metadata field names they emit.
    for locator_field in (
        "section_path",
        "slide_number",
        "sheet_name",
        "row_range",
        "line_start",
    ):
        assert f"`{locator_field}`" in docs
    assert "locators" in docs

    # Internal test/fixture/judge proof references must stay out of user docs.
    for proof_reference in (
        "Apache Tika",
        "real-file judge",
        "JSON fixtures",
        "fixture coverage",
    ):
        assert proof_reference not in normalized
