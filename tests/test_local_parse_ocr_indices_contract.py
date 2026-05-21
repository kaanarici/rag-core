from __future__ import annotations

import asyncio
import io
import json
import sys

import pytest

from rag_core.documents import local_parse
from rag_core.documents.converters.pdf_converter_inspector import (
    _normalize_inspector_ocr_page_indices,
)
from rag_core.documents.converters.base import ConversionResult
from rag_core.documents.ocr_commands import gemini as gemini_command


def test_local_parse_ocr_page_indices_drop_boolean_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert_with_mixed_indices(
        file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        assert file_bytes == b"%PDF-1.7"
        assert filename == "scan.pdf"
        assert mime_type == "application/pdf"
        return ConversionResult(
            content="# scanned",
            metadata={
                "parser": "local:stub",
                "needs_ocr": True,
                "ocr_page_indices": [True, 2, False, 0, 2, "3"],
            },
        )

    import rag_core.documents.converters as converters_module

    monkeypatch.setattr(converters_module, "convert_file", convert_with_mixed_indices)

    _, metadata = asyncio.run(
        local_parse.parse_file_bytes(
            file_bytes=b"%PDF-1.7",
            filename="scan.pdf",
            mime_type="application/pdf",
        )
    )

    assert metadata["ocr_page_indices"] == [0, 2]


def test_local_parse_preserves_empty_normalized_ocr_page_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert_with_invalid_indices(
        file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        assert file_bytes == b"%PDF-1.7"
        assert filename == "scan.pdf"
        assert mime_type == "application/pdf"
        return ConversionResult(
            content="# scanned",
            metadata={
                "parser": "local:stub",
                "needs_ocr": True,
                "ocr_page_indices": [True, False, "3"],
            },
        )

    import rag_core.documents.converters as converters_module

    monkeypatch.setattr(converters_module, "convert_file", convert_with_invalid_indices)

    _, metadata = asyncio.run(
        local_parse.parse_file_bytes(
            file_bytes=b"%PDF-1.7",
            filename="scan.pdf",
            mime_type="application/pdf",
        )
    )

    assert metadata["ocr_page_indices"] == []


def test_inspector_page_indices_reject_boolean_values() -> None:
    assert _normalize_inspector_ocr_page_indices(
        [True, 2, False, 0, 2],
        page_count=5,
    ) == [2, 0]


def test_local_parse_empty_ocr_output_requires_real_page_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert_with_boolean_only_index(
        file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        assert file_bytes == b"%PDF-1.7"
        assert filename == "scan.pdf"
        assert mime_type == "application/pdf"
        return ConversionResult(
            content="",
            metadata={
                "parser": "local:stub",
                "needs_ocr": True,
                "ocr_page_indices": [True],
            },
        )

    import rag_core.documents.converters as converters_module

    monkeypatch.setattr(converters_module, "convert_file", convert_with_boolean_only_index)

    with pytest.raises(local_parse.LocalParseError, match="Converter returned empty output"):
        asyncio.run(
            local_parse.parse_file_bytes(
                file_bytes=b"%PDF-1.7",
                filename="scan.pdf",
                mime_type="application/pdf",
            )
        )


def test_local_parse_allows_empty_image_output_when_ocr_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert_image_for_ocr(
        file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        assert file_bytes == b"\x89PNG"
        assert filename == "scan.png"
        assert mime_type == "image/png"
        return ConversionResult(
            content="",
            metadata={
                "parser": "ocr_required",
                "needs_ocr": True,
            },
        )

    import rag_core.documents.converters as converters_module

    monkeypatch.setattr(converters_module, "convert_file", convert_image_for_ocr)

    markdown, metadata = asyncio.run(
        local_parse.parse_file_bytes(
            file_bytes=b"\x89PNG",
            filename="scan.png",
            mime_type="image/png",
        )
    )

    assert markdown == ""
    assert metadata["parser"] == "ocr_required"
    assert metadata["needs_ocr"] is True


def test_local_parse_allows_empty_encrypted_pdf_for_full_document_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert_encrypted_pdf(
        file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        assert file_bytes == b"%PDF-1.7"
        assert filename == "secret.pdf"
        assert mime_type == "application/pdf"
        return ConversionResult(
            content="",
            needs_ocr=True,
            metadata={
                "parser": "local:pymupdf",
                "needs_ocr": True,
                "is_encrypted": True,
                "ocr_processed_entire_document": True,
            },
        )

    import rag_core.documents.converters as converters_module

    monkeypatch.setattr(converters_module, "convert_file", convert_encrypted_pdf)

    markdown, metadata = asyncio.run(
        local_parse.parse_file_bytes(
            file_bytes=b"%PDF-1.7",
            filename="secret.pdf",
            mime_type="application/pdf",
        )
    )

    assert markdown == ""
    assert metadata["needs_ocr"] is True
    assert metadata["is_encrypted"] is True


def test_gemini_image_whole_document_ocr_records_page_count_and_pages_processed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"\x89PNG")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        gemini_command,
        "_load_json",
        lambda req: {
            "candidates": [
                {"content": {"parts": [{"text": "# OCR"}]}},
            ]
        },
    )
    monkeypatch.setattr(sys, "argv", ["gemini.py"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "file_path": str(image_path),
                    "filename": "scan.png",
                    "mime_type": "image/png",
                    "page_indices": [2],
                }
            )
        ),
    )

    assert gemini_command.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pages_processed"] == [0]
    assert output["metadata"]["ocr_pages_used_count"] == 1
    assert output["metadata"]["ocr_page_count"] == 1
    assert output["metadata"]["page_count"] == 1
    assert output["metadata"]["ocr_page_indices_ignored"] is True


def test_command_ocr_preserves_requested_page_indices_for_non_selecting_provider(
) -> None:
    from rag_core.documents.ocr_command_runtime import run_command_ocr

    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        del command
        raw_input = kwargs["input"]
        assert isinstance(raw_input, str)
        captured.update(json.loads(raw_input))

        class _Completed:
            returncode = 0
            stdout = json.dumps(
                {
                    "markdown": "# OCR",
                    "merge_mode": "replace",
                    "provider_name": "gemini",
                    "model_name": "gemini-2.5-flash",
                    "pages_processed": [0],
                    "metadata": {"ocr_page_indices_ignored": True},
                }
            )

        return _Completed()

    output = run_command_ocr(
        file_bytes=b"%PDF",
        filename="scan.pdf",
        mime_type="application/pdf",
        page_indices=[2],
        existing_markdown="",
        metadata={},
        command=["ocr"],
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        supports_page_selection=False,
        timeout_seconds=10,
        extra_env={},
        run_command=fake_run,
    )

    assert captured["page_indices"] == []
    assert captured["requested_page_indices"] == [2]
    assert output.metadata["ocr_page_indices_ignored"] is True
