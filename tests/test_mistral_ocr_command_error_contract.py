from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib import error, request

import pytest

from rag_core.documents.ocr_commands import mistral_runtime

SECRET = "sk-test-secret"
PRIVATE_BODY = f'{{"error": "private api key {SECRET}"}}'.encode()
PRIVATE_URL = f"https://api.mistral.ai/v1/ocr?token={SECRET}"


def test_mistral_http_error_omits_response_body_and_raw_error(
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

    monkeypatch.setattr(mistral_runtime.request, "urlopen", raise_private_http_error)
    req = request.Request("https://api.mistral.ai/v1/ocr")

    with pytest.raises(RuntimeError) as exc_info:
        mistral_runtime._load_json(req)

    message = str(exc_info.value)
    assert message == "Mistral OCR request failed (429)"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert SECRET not in message
    assert PRIVATE_URL not in message
    assert "private api key" not in message
    assert "private status detail" not in message
    assert "Traceback" not in message
