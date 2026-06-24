from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.documents.converters.json_converter import JsonConverter


def test_jsonl_converter_parses_valid_json_lines_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    converter = JsonConverter()
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(
            converter.convert(
                b'{"team": "retrieval"}\n{"team": "parsing"}\n',
                filename="cases.jsonl",
                mime_type="application/octet-stream",
            )
        )

    assert result.content == (
        '```jsonl\n{"team": "retrieval"}\n{"team": "parsing"}\n```'
    )
    assert result.metadata["parser"] == "local:json"
    assert result.metadata["format"] == "jsonl"
    assert result.metadata["record_count"] == 2
    assert result.metadata["needs_ocr"] is False
    assert "Invalid json" not in caplog.text


def test_jsonl_converter_uses_jsonl_mime_type() -> None:
    converter = JsonConverter()
    result = asyncio.run(
        converter.convert(
            b'{"query": "billing"}\n',
            filename="cases.txt",
            mime_type="application/x-ndjson",
        )
    )

    assert result.content.startswith("```jsonl\n")
    assert result.metadata["format"] == "jsonl"
    assert result.metadata["record_count"] == 1
