from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.documents.converters.code_converter import CodeConverter
from rag_core.documents.converters.json_converter import JsonConverter
from rag_core.documents.converters.xml_converter import XmlConverter
from tests.support import assert_caplog_omits_private


def test_json_invalid_fallback_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_json = '{"secret": "sk-test-secret", "broken": }'

    with caplog.at_level(
        logging.WARNING,
        logger="rag_core.documents.converters.json_converter",
    ):
        result = asyncio.run(
            JsonConverter().convert(
                raw_json.encode(),
                filename="private-roadmap.json",
                mime_type="application/json",
            )
        )

    raw_error = result.metadata["parse_error"]
    assert isinstance(raw_error, str)
    assert result.content == f"```json\n{raw_json}\n```"
    assert result.metadata["parser"] == "local:json"
    assert result.quality is not None

    assert "json" in caplog.text
    assert "JSONDecodeError" in caplog.text
    assert_caplog_omits_private(
        caplog,
        "private-roadmap.json",
        raw_error,
        raw_json,
    )


def test_xml_malformed_fallback_debug_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_xml = "<root><secret>sk-test-secret</root>"

    with caplog.at_level(
        logging.DEBUG,
        logger="rag_core.documents.converters.xml_converter",
    ):
        result = asyncio.run(
            XmlConverter().convert(
                raw_xml.encode(),
                filename="private-roadmap.xml",
                mime_type="application/xml",
            )
        )

    assert result.content == f"```xml\n{raw_xml}\n```"
    assert result.metadata["parser"] == "local:xml"
    assert result.metadata["needs_ocr"] is False
    assert result.quality is not None

    log_text = caplog.text.lower()
    assert "xml" in log_text
    assert "expaterror" in log_text
    assert_caplog_omits_private(
        caplog,
        "private-roadmap.xml",
        "mismatched tag",
        raw_xml,
    )


def test_code_binary_replacement_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_code = b'print("sk-test-secret")\xff\xfe'

    with caplog.at_level(
        logging.WARNING,
        logger="rag_core.documents.converters.code_converter",
    ):
        result = asyncio.run(
            CodeConverter().convert(
                raw_code,
                filename="private-tool.py",
                mime_type="text/x-python",
            )
        )

    assert result.content == ""
    assert result.metadata == {
        "parser": "local:code",
        "error": "binary content detected",
    }
    assert result.quality is None

    assert "code" in caplog.text
    assert "2 replacement chars" in caplog.text
    assert_caplog_omits_private(caplog, "private-tool.py")
