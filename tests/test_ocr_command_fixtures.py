from __future__ import annotations

import json
from pathlib import Path

from rag_core.documents.ocr_commands import gemini, mistral

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ocr"


def test_mistral_recorded_response_shape_selects_requested_pages() -> None:
    payload = json.loads((_FIXTURE_DIR / "mistral_pages.json").read_text())
    pages = payload["pages"]

    assert mistral._processed_page_indices(pages, [1]) == [1]
    assert mistral._collect_markdown(pages, [1]) == (
        "## Page 2\n\n## Appendix\n\nCarrier customs decoy"
    )
    assert mistral._collect_markdown(pages, []) == (
        "## Page 1\n\n# Invoice Scan\n\nACH payment instructions\n\n"
        "## Page 2\n\n## Appendix\n\nCarrier customs decoy"
    )


def test_gemini_recorded_response_shape_extracts_markdown_text() -> None:
    payload = json.loads((_FIXTURE_DIR / "gemini_generate_content.json").read_text())

    markdown = gemini._extract_text(payload)

    assert markdown == "# OCR Result\n\nWebhook signature text from scanned page."
    assert gemini._whole_document_page_count(
        "scan.png",
        "image/png",
        metadata=None,
    ) == 1
    assert gemini._whole_document_page_count(
        "document.pdf",
        "application/pdf",
        metadata={"page_count": 3},
    ) == 3
