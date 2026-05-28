from __future__ import annotations

from email.message import Message
from io import BytesIO
from types import ModuleType
from typing import Protocol
from urllib import error, request
from urllib.request import Request

import pytest

from rag_core.documents.ocr_commands import gemini as gemini_command
from rag_core.documents.ocr_commands import mistral_runtime

SECRET = "sk-test-secret"
PRIVATE_BODY = f'{{"error": "private api key {SECRET}"}}'.encode()


class _OcrHttpModule(Protocol):
    request: ModuleType

    def _load_json(self, req: Request) -> object: ...

_OCR_HTTP_CASES = (
    pytest.param(
        mistral_runtime,
        "https://api.mistral.ai/v1/ocr",
        f"https://api.mistral.ai/v1/ocr?token={SECRET}",
        "Mistral OCR request failed (429)",
        id="mistral",
    ),
    pytest.param(
        gemini_command,
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={SECRET}",
        "Gemini OCR request failed (429)",
        id="gemini",
    ),
)


@pytest.mark.parametrize(
    ("module", "request_url", "private_url", "expected_message"),
    _OCR_HTTP_CASES,
)
def test_ocr_http_error_omits_response_body_and_raw_error(
    monkeypatch: pytest.MonkeyPatch,
    module: _OcrHttpModule,
    request_url: str,
    private_url: str,
    expected_message: str,
) -> None:
    def raise_private_http_error(*args: object, **kwargs: object) -> None:
        raise error.HTTPError(
            url=private_url,
            code=429,
            msg=f"private status detail {SECRET}",
            hdrs=Message(),
            fp=BytesIO(PRIVATE_BODY),
        )

    monkeypatch.setattr(module.request, "urlopen", raise_private_http_error)
    req = request.Request(request_url)

    with pytest.raises(RuntimeError) as exc_info:
        module._load_json(req)

    message = str(exc_info.value)
    assert message == expected_message
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert SECRET not in message
    assert private_url not in message
    assert "private api key" not in message
    assert "private status detail" not in message
    assert "Traceback" not in message
