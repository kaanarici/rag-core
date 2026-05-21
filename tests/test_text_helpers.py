from __future__ import annotations

import asyncio

import pytest

from rag_core.documents.converters.quality import QualityVerdict, score_text_quality
from rag_core.documents.converters.text_helpers import (
    detect_encoding,
    render_markdown_table,
    safe_decode,
)
from rag_core.documents.local_parse import parse_file_bytes


def test_safe_decode_handles_cp1252_smart_punctuation_without_chardet() -> None:
    raw = "“Quoted” pricing – not mojibake".encode("cp1252")

    assert detect_encoding(raw) == "cp1252"
    assert safe_decode(raw) == "“Quoted” pricing – not mojibake"


def test_detect_encoding_does_not_treat_binary_controls_as_cp1252_text() -> None:
    raw = b"\x93quoted\x94\x00\x00\x00\x00\x00\x00"

    assert detect_encoding(raw) == "utf-8"


def test_render_markdown_table_escapes_pipe_cells() -> None:
    table = render_markdown_table([["team", "notes"], ["retrieval", "fast | precise"]])

    assert "| retrieval | fast \\| precise |" in table


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_safe_decode_honors_unicode_bom_encodings(encoding: str) -> None:
    text = "Retrieval parser fixture with snowman \u2603 and clean text."
    raw = text.encode(encoding)

    assert detect_encoding(raw) == encoding
    assert safe_decode(raw) == text


@pytest.mark.parametrize(
    ("encoding", "detected"),
    [
        ("utf-16-le", "utf-16-le"),
        ("utf-16-be", "utf-16-be"),
        ("utf-32-le", "utf-32-le"),
        ("utf-32-be", "utf-32-be"),
    ],
)
def test_safe_decode_honors_bomless_unicode_encodings(
    encoding: str,
    detected: str,
) -> None:
    text = "Retrieval parser fixture with clean Unicode text."
    raw = text.encode(encoding)

    assert detect_encoding(raw) == detected
    assert safe_decode(raw) == text


@pytest.mark.parametrize(
    ("filename", "mime_type", "payload", "expected"),
    [
        (
            "notes.txt",
            "text/plain",
            "Retrieval parser fixture with enough readable text for scoring.",
            "Retrieval parser fixture",
        ),
        (
            "scores.csv",
            "text/csv",
            "team,score\nretrieval,99\nparsing,98\n",
            "| team | score |",
        ),
        (
            "score.json",
            "application/json",
            '{"team": "retrieval", "score": 99}',
            '"team": "retrieval"',
        ),
        (
            "score.xml",
            "application/xml",
            "<root><team>retrieval</team></root>",
            "<team>retrieval</team>",
        ),
        (
            "report.html",
            "text/html",
            "<html><body><main><h1>Retrieval Report</h1><p>Clean text.</p></main></body></html>",
            "Retrieval Report",
        ),
    ],
)
@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_unicode_bom_text_decodes_for_text_converter_paths(
    filename: str,
    mime_type: str,
    payload: str,
    expected: str,
    encoding: str,
) -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=payload.encode(encoding),
            filename=filename,
            mime_type=mime_type,
        )
    )

    assert expected in markdown
    assert "\x00" not in markdown
    assert "\ufffd" not in markdown
    assert metadata["quality"]["mojibake_ratio"] == 0.0


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-32-le"])
def test_bomless_unicode_text_decodes_for_csv_converter_paths(
    encoding: str,
) -> None:
    rows = "\n".join(
        [
            "team,score",
            "retrieval,99",
            "parsing,98",
            "reranking,97",
            "evaluation,96",
        ]
    )
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=rows.encode(encoding),
            filename="scores.csv",
            mime_type="text/csv",
        )
    )

    assert "| team | score |" in markdown
    assert "\x00" not in markdown
    assert metadata["quality"]["verdict"] != QualityVerdict.POOR.value


def test_quality_rejects_control_character_corrupted_text() -> None:
    score = score_text_quality("R\x00e\x00t\x00r\x00i\x00e\x00v\x00a\x00l\x00 " * 8)

    assert score.verdict == QualityVerdict.POOR
    assert score.details.startswith("high mojibake/control ratio")
