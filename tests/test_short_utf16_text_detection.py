from __future__ import annotations

import pytest

from rag_core.documents.converters.text_helpers import detect_encoding, safe_decode


@pytest.mark.parametrize(
    ("text", "encoding", "detected"),
    [
        ("A", "utf-16-le", "utf-16-le"),
        ("A", "utf-16-be", "utf-16-be"),
        ("OK", "utf-16-le", "utf-16-le"),
        ("OK", "utf-16-be", "utf-16-be"),
    ],
)
def test_safe_decode_honors_short_bomless_utf16_text(
    text: str,
    encoding: str,
    detected: str,
) -> None:
    raw = text.encode(encoding)

    assert detect_encoding(raw) == detected
    assert safe_decode(raw) == text


def test_short_alternating_nul_control_payload_still_rejects_as_binary() -> None:
    with pytest.raises(ValueError, match="binary content detected"):
        safe_decode(b"\x01\x00")
