from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from typing import cast
from urllib.request import Request

import pytest

from rag_core.documents import local_parse
from rag_core.documents.converters.pdf_converter_inspector import (
    _normalize_inspector_ocr_page_indices,
)
from rag_core.documents.converters.base import ConversionResult
from rag_core.documents.ocr_commands import gemini as gemini_command


def _make_pdf_bytes(page_count: int) -> bytes:
    pytest.importorskip("fitz")
    import fitz

    doc = fitz.open()
    try:
        for page_number in range(1, page_count + 1):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {page_number}")
        return bytes(doc.tobytes())
    finally:
        doc.close()


def _gemini_request_body(req: Request) -> dict[str, object]:
    assert isinstance(req.data, bytes)
    body = json.loads(req.data.decode("utf-8"))
    assert isinstance(body, dict)
    return cast(dict[str, object], body)


def _gemini_request_parts(body: dict[str, object]) -> list[object]:
    contents = body["contents"]
    assert isinstance(contents, list)
    first_content = contents[0]
    assert isinstance(first_content, dict)
    parts = first_content["parts"]
    assert isinstance(parts, list)
    return parts


def _gemini_inline_bytes(body: dict[str, object]) -> bytes:
    inline_part = _gemini_request_parts(body)[0]
    assert isinstance(inline_part, dict)
    inline_data = inline_part["inline_data"]
    assert isinstance(inline_data, dict)
    data = inline_data["data"]
    assert isinstance(data, str)
    return base64.b64decode(data)


def _gemini_prompt(body: dict[str, object]) -> str:
    text_part = _gemini_request_parts(body)[1]
    assert isinstance(text_part, dict)
    text = text_part["text"]
    assert isinstance(text, str)
    return text


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


def test_gemini_pdf_subset_uploads_only_requested_pages_and_reports_original_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf_bytes = _make_pdf_bytes(4)
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(pdf_bytes)

    def fake_load_json(req: Request) -> dict[str, object]:
        body = _gemini_request_body(req)
        uploaded_pdf = _gemini_inline_bytes(body)
        pytest.importorskip("fitz")
        import fitz

        doc = fitz.open(stream=uploaded_pdf, filetype="pdf")
        try:
            assert doc.page_count == 2
        finally:
            doc.close()
        assert (
            "original document page numbers in order: 2, 4." in _gemini_prompt(body)
        )
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "## Page 1\n\nsecond page\n\n"
                                    "## Page 2\n\nfourth page"
                                )
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(gemini_command, "_load_json", fake_load_json)
    monkeypatch.setattr(sys, "argv", ["gemini.py"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "file_path": str(pdf_path),
                    "filename": "scan.pdf",
                    "mime_type": "application/pdf",
                    "page_indices": [],
                    "requested_page_indices": [1, 3],
                    "metadata": {"page_count": 4},
                }
            )
        ),
    )

    assert gemini_command.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["markdown"] == "## Page 2\n\nsecond page\n\n## Page 4\n\nfourth page"
    assert output["merge_mode"] == "append"
    assert output["pages_processed"] == [1, 3]
    assert output["metadata"]["ocr_processed_entire_document"] is False
    assert output["metadata"]["ocr_page_selection_supported"] is True
    assert output["metadata"]["ocr_page_indices_ignored"] is False
    assert output["metadata"]["ocr_pages_used_count"] == 2
    assert output["metadata"]["ocr_page_count"] == 2
    assert output["metadata"]["page_count"] == 4


@pytest.mark.parametrize("requested_page_indices", [[], [0, 1]])
def test_gemini_pdf_empty_and_full_range_requests_upload_original_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    requested_page_indices: list[int],
) -> None:
    pdf_bytes = _make_pdf_bytes(2)
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(pdf_bytes)

    def fake_load_json(req: Request) -> dict[str, object]:
        body = _gemini_request_body(req)
        assert _gemini_inline_bytes(body) == pdf_bytes
        assert _gemini_prompt(body) == gemini_command._build_prompt(
            requested_page_indices
        )
        return {
            "candidates": [
                {"content": {"parts": [{"text": "whole document"}]}},
            ]
        }

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(gemini_command, "_load_json", fake_load_json)
    monkeypatch.setattr(sys, "argv", ["gemini.py"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "file_path": str(pdf_path),
                    "filename": "scan.pdf",
                    "mime_type": "application/pdf",
                    "requested_page_indices": requested_page_indices,
                    "metadata": {"page_count": 2},
                }
            )
        ),
    )

    assert gemini_command.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["merge_mode"] == "replace"
    assert output["pages_processed"] == [0, 1]
    assert output["metadata"]["ocr_processed_entire_document"] is True
    assert output["metadata"]["ocr_page_selection_supported"] is False
    assert output["metadata"]["ocr_page_indices_ignored"] is bool(requested_page_indices)


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
