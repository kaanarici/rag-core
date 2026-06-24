from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import rag_core.documents.pdf_inspector as pdf_inspector_module
import rag_core.documents.pdf_inspector_runtime as pdf_inspector_runtime


def _detection_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "pdf_type": "text",
        "page_count": 1,
        "pages_needing_ocr": [],
        "confidence": 0.9,
        "has_encoding_issues": False,
        "processing_time_ms": 3,
    }
    payload.update(overrides)
    return payload


def _extraction_payload(**overrides: object) -> dict[str, object]:
    payload = _detection_payload(markdown="# Report")
    payload.update(overrides)
    return payload


def _payload_with_unknown_timing_key(markdown: str | None = None) -> dict[str, object]:
    payload = _detection_payload()
    payload.pop("processing_time_ms")
    payload["detection_time_ms"] = 99
    if markdown is not None:
        payload["markdown"] = markdown
    return payload


@pytest.mark.parametrize("page_count", [0, -1])
def test_detection_rejects_non_positive_page_count(
    monkeypatch: pytest.MonkeyPatch,
    page_count: int,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(page_count=page_count),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is None


@pytest.mark.parametrize("page_count", [0, -1])
def test_extraction_rejects_non_positive_page_count(
    monkeypatch: pytest.MonkeyPatch,
    page_count: int,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _extraction_payload(page_count=page_count),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is None


def test_detection_defaults_missing_pages_needing_ocr_to_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _detection_payload()
    payload.pop("pages_needing_ocr")
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: payload,
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.pages_needing_ocr == []


def test_detection_normalizes_pages_needing_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(pages_needing_ocr=[2, 1, 2]),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.pages_needing_ocr == [1, 0]


@pytest.mark.parametrize(
    "invalid_value",
    ["1", [0], [-1], ["1"], [1.0], [True]],
)
def test_detection_rejects_invalid_pages_needing_ocr(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(pages_needing_ocr=invalid_value),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is None


@pytest.mark.parametrize(
    "invalid_value",
    ["1", [0], [-1], ["1"], [1.0], [True]],
)
def test_extraction_rejects_invalid_pages_needing_ocr(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _extraction_payload(pages_needing_ocr=invalid_value),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is None


@pytest.mark.parametrize("confidence", [float("inf"), "-inf", "nan"])
def test_detection_drops_non_finite_confidence(
    monkeypatch: pytest.MonkeyPatch,
    confidence: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(confidence=confidence),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.confidence is None


def test_detection_normalizes_optional_analysis_page_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(
            pages_with_tables=[2, 1, 2],
            pages_with_columns=[3, 3],
        ),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.pages_with_tables == [1, 0]
    assert result.pages_with_columns == [2]
    assert result.is_complex is True


def test_extraction_normalizes_optional_analysis_page_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _extraction_payload(
            pages_with_tables=[2, 1, 2],
            pages_with_columns=[3, 3],
        ),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.pages_with_tables == [1, 0]
    assert result.pages_with_columns == [2]
    assert result.is_complex is True


def test_detection_allows_null_optional_analysis_page_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(
            pages_with_tables=None,
            pages_with_columns=None,
        ),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.pages_with_tables is None
    assert result.pages_with_columns is None
    assert result.is_complex is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("pages_with_tables", "1"),
        ("pages_with_columns", 1),
        ("pages_with_tables", [0]),
        ("pages_with_columns", [-1]),
        ("pages_with_tables", ["1"]),
        ("pages_with_columns", [1.0]),
        ("pages_with_tables", [True]),
    ],
)
def test_detection_rejects_invalid_optional_analysis_page_lists(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_value: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(**{field_name: invalid_value}),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("pages_with_tables", "1"),
        ("pages_with_columns", 1),
        ("pages_with_tables", [0]),
        ("pages_with_columns", [-1]),
        ("pages_with_tables", ["1"]),
        ("pages_with_columns", [1.0]),
        ("pages_with_tables", [True]),
    ],
)
def test_extraction_rejects_invalid_optional_analysis_page_lists(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_value: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _extraction_payload(**{field_name: invalid_value}),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is None


@pytest.mark.parametrize("processing_time_ms", [-1, "-2"])
def test_detection_drops_negative_processing_time(
    monkeypatch: pytest.MonkeyPatch,
    processing_time_ms: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(
            processing_time_ms=processing_time_ms
        ),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.processing_time_ms is None


@pytest.mark.parametrize("processing_time_ms", [-1, "-2"])
def test_extraction_drops_negative_processing_time(
    monkeypatch: pytest.MonkeyPatch,
    processing_time_ms: object,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _extraction_payload(
            processing_time_ms=processing_time_ms
        ),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.processing_time_ms is None


def test_detection_ignores_unknown_timing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _payload_with_unknown_timing_key(),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.processing_time_ms is None


def test_extraction_ignores_unknown_timing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _payload_with_unknown_timing_key("# Report"),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.processing_time_ms is None


def test_detection_preserves_valid_numeric_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _detection_payload(
            confidence="0.82",
            processing_time_ms="4",
        ),
    )

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.page_count == 1
    assert result.confidence is not None
    assert math.isclose(result.confidence, 0.82)
    assert result.processing_time_ms == 4


def test_extraction_preserves_valid_numeric_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: _extraction_payload(processing_time_ms="4"),
    )

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is not None
    assert result.page_count == 1
    assert result.processing_time_ms == 4


def test_wheel_process_result_normalizes_to_inspector_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def process_pdf_bytes(file_bytes: bytes) -> SimpleNamespace:
        assert file_bytes == b"%PDF-1.7"
        return SimpleNamespace(
            pdf_type="mixed",
            page_count=4,
            pages_needing_ocr=[2, True, 0, 2, False, -1, 9],
            confidence="0.82",
            has_encoding_issues=True,
            processing_time_ms=7,
            is_complex_layout=True,
            pages_with_tables=[2],
            pages_with_columns=[1, 3],
            markdown=("mixed wheel markdown " * 8).strip(),
        )

    monkeypatch.setattr(
        pdf_inspector_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(process_pdf_bytes=process_pdf_bytes),
    )

    result = pdf_inspector_module.process_pdf_with_inspector_wheel(b"%PDF-1.7")

    assert result is not None
    assert result.has_explicit_ocr_page_info is True
    assert result.detection.pdf_type == "mixed"
    assert result.detection.page_count == 4
    assert result.detection.pages_needing_ocr == [2, 0]
    assert result.detection.confidence is not None
    assert math.isclose(result.detection.confidence, 0.82)
    assert result.detection.has_encoding_issues is True
    assert result.detection.processing_time_ms == 7
    assert result.detection.is_complex is True
    assert result.detection.pages_with_tables == [2]
    assert result.detection.pages_with_columns == [1, 3]
    assert result.extraction.markdown.startswith("mixed wheel markdown")


def test_wheel_process_result_defaults_scanned_pages_to_ocr() -> None:
    result = pdf_inspector_module._process_wheel_result(
        {
            "pdf_type": "scanned",
            "page_count": 3,
            "confidence": 0.9,
            "markdown": "",
        }
    )

    assert result.has_explicit_ocr_page_info is False
    assert result.detection.pages_needing_ocr == [0, 1, 2]
