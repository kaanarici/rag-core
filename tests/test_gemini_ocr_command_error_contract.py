from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib import error, request

import pytest

from rag_core.documents.ocr_commands import gemini as gemini_command

SECRET = "sk-test-secret"
PRIVATE_BODY = f'{{"error": "private api key {SECRET}"}}'.encode()
PRIVATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={SECRET}"
)


def test_gemini_http_error_omits_response_body_and_raw_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_private_http_error(*args: object, **kwargs: object) -> None:
        raise error.HTTPError(
            url=PRIVATE_URL,
            code=429,
            msg=f"private status detail {SECRET}",
            hdrs=Message(),
            fp=BytesIO(PRIVATE_BODY),
        )

    monkeypatch.setattr(gemini_command.request, "urlopen", raise_private_http_error)
    req = request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )

    with pytest.raises(RuntimeError) as exc_info:
        gemini_command._load_json(req)

    message = str(exc_info.value)
    assert message == "Gemini OCR request failed (429)"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert SECRET not in message
    assert PRIVATE_URL not in message
    assert "private api key" not in message
    assert "private status detail" not in message
    assert "Traceback" not in message
