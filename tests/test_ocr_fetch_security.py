"""OCR subprocess egress hardening: host pinning, timeout, response cap.

The OCR subprocesses (``mistral_runtime`` / ``gemini``) run in their own Python
process and do not import ``rag_core.fetching``. They must still enforce the
same fetch discipline: pin to the vendor host, refuse redirects to any other
origin, pass a ``urlopen`` timeout, and cap the response body so a malicious
or runaway response cannot exhaust memory.
"""

from __future__ import annotations

import io
from email.message import EmailMessage
from http.client import HTTPResponse
from types import ModuleType
from typing import Protocol, cast
from urllib import request
from urllib.request import Request
from urllib.response import addinfourl

import pytest

from rag_core.documents.ocr_commands import gemini as gemini_command
from rag_core.documents.ocr_commands import mistral_runtime


class _OcrHttpModule(Protocol):
    request: ModuleType

    def _load_json(self, req: Request) -> object: ...

    def _assert_pinned_host(self, url: str) -> None: ...


_PINNED_HOST_CASES = (
    pytest.param(
        mistral_runtime,
        "https://api.mistral.ai/v1/ocr",
        "Mistral OCR refused unpinned host: evil.example.com",
        id="mistral",
    ),
    pytest.param(
        gemini_command,
        "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
        "Gemini OCR refused unpinned host: evil.example.com",
        id="gemini",
    ),
)


@pytest.mark.parametrize(
    ("module", "pinned_url", "expected_message"), _PINNED_HOST_CASES
)
def test_ocr_subprocess_refuses_unpinned_redirect_host(
    monkeypatch: pytest.MonkeyPatch,
    module: _OcrHttpModule,
    pinned_url: str,
    expected_message: str,
) -> None:
    class _RedirectingResponse:
        def __init__(self) -> None:
            self._payload = b'{"ok": true}'

        def __enter__(self) -> "_RedirectingResponse":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://evil.example.com/redirected"

        def read(self, _: int) -> bytes:
            return self._payload

    def fake_urlopen(*args: object, **kwargs: object) -> _RedirectingResponse:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, (int, float)), (
            "subprocess must pass a numeric urlopen timeout"
        )
        assert timeout > 0, "timeout must be positive"
        return _RedirectingResponse()

    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as exc_info:
        module._load_json(Request(pinned_url))

    assert str(exc_info.value) == expected_message


_CREDENTIAL_REDIRECT_CASES = (
    pytest.param(
        gemini_command,
        "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
        "x-goog-api-key",
        id="gemini",
    ),
    pytest.param(
        mistral_runtime,
        "https://api.mistral.ai/v1/ocr",
        "Authorization",
        id="mistral",
    ),
)


def _addinfourl(body: bytes, url: str, code: int, msg: str) -> addinfourl:
    resp = addinfourl(io.BytesIO(body), EmailMessage(), url, code)
    resp.msg = msg  # type: ignore[attr-defined]  # HTTPErrorProcessor reads .msg
    return resp


def _redirect_addinfourl(url: str, location: str) -> addinfourl:
    resp = _addinfourl(b"", url, 302, "Found")
    resp.headers["Location"] = location
    return resp


def _ok_addinfourl(url: str) -> addinfourl:
    return _addinfourl(b'{"ok": true}', url, 200, "OK")


@pytest.mark.parametrize(
    ("module", "pinned_url", "credential_header"), _CREDENTIAL_REDIRECT_CASES
)
def test_ocr_redirect_does_not_leak_credential_to_other_host(
    monkeypatch: pytest.MonkeyPatch,
    module: _OcrHttpModule,
    pinned_url: str,
    credential_header: str,
) -> None:
    """A 3xx to a different host must raise before the credential is re-sent.

    The vendor host answers with a 302 to ``evil.example.com``. The module's
    ``_NoRedirectHandler`` (installed as the process opener) must refuse the
    redirect so the redirect target is never contacted and never receives the
    credential header. The vanilla ``HTTPRedirectHandler`` would instead follow
    the redirect and re-send the header to the attacker host.
    """
    delivered: list[tuple[str, dict[str, str]]] = []
    secret = "super-secret-token"

    class _FakeHTTPSHandler(request.HTTPSHandler):
        def https_open(self, req: Request) -> HTTPResponse:
            delivered.append((req.host, dict(req.header_items())))
            if req.host != "evil.example.com":
                resp = _redirect_addinfourl(req.full_url, "https://evil.example.com/leak")
            else:
                resp = _ok_addinfourl(req.full_url)
            return cast(HTTPResponse, resp)

    no_redirect = module._NoRedirectHandler()  # type: ignore[attr-defined]
    opener = request.build_opener(_FakeHTTPSHandler, no_redirect)

    def fake_urlopen(req: Request, *, timeout: float | None = None) -> object:
        return opener.open(req, timeout=timeout)

    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)

    req = Request(pinned_url, headers={credential_header: secret})
    with pytest.raises(RuntimeError):
        module._load_json(req)

    # The credential reaches only the pinned vendor host on the first hop. The
    # redirect target must receive no request at all, hence no leaked header.
    assert "evil.example.com" not in [host for host, _ in delivered], (
        "redirect target must never receive a request"
    )
    leaked = [
        host
        for host, headers in delivered
        if host == "evil.example.com"
        and any(secret in value for value in headers.values())
    ]
    assert leaked == [], "credential must not be delivered to the redirect target"


@pytest.mark.parametrize(
    ("module", "pinned_url", "max_bytes_env", "expected_match"),
    [
        pytest.param(
            mistral_runtime,
            "https://api.mistral.ai/v1/ocr",
            "RAG_CORE_OCR_FETCH_MAX_BYTES",
            "Mistral OCR response exceeded max bytes",
            id="mistral",
        ),
        pytest.param(
            gemini_command,
            "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
            "RAG_CORE_OCR_FETCH_MAX_BYTES",
            "Gemini OCR response exceeded max bytes",
            id="gemini",
        ),
    ],
)
def test_ocr_subprocess_caps_response_body(
    monkeypatch: pytest.MonkeyPatch,
    module: _OcrHttpModule,
    pinned_url: str,
    max_bytes_env: str,
    expected_match: str,
) -> None:
    monkeypatch.setenv(max_bytes_env, "128")

    oversized = b"x" * 512

    class _OversizedResponse:
        def __enter__(self) -> "_OversizedResponse":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def geturl(self) -> str:
            return pinned_url

        def read(self, max_bytes: int) -> bytes:
            assert max_bytes == 128 + 1, "must read at most max_bytes + 1"
            return oversized[:max_bytes]

    def fake_urlopen(*args: object, **kwargs: object) -> _OversizedResponse:
        return _OversizedResponse()

    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match=expected_match):
        module._load_json(Request(pinned_url))


@pytest.mark.parametrize(
    ("module", "pinned_url", "expected_match"),
    [
        pytest.param(
            mistral_runtime,
            "https://api.mistral.ai/v1/ocr",
            "Mistral OCR request timed out",
            id="mistral",
        ),
        pytest.param(
            gemini_command,
            "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
            "Gemini OCR request timed out",
            id="gemini",
        ),
    ],
)
def test_ocr_subprocess_translates_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
    module: _OcrHttpModule,
    pinned_url: str,
    expected_match: str,
) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> None:
        raise TimeoutError("simulated socket timeout")

    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match=expected_match):
        module._load_json(Request(pinned_url))


@pytest.mark.parametrize(
    ("module", "expected_match"),
    [
        pytest.param(
            mistral_runtime,
            "Mistral OCR refused unpinned host",
            id="mistral",
        ),
        pytest.param(
            gemini_command,
            "Gemini OCR refused unpinned host",
            id="gemini",
        ),
    ],
)
def test_ocr_subprocess_assert_pinned_host_rejects_non_vendor(
    module: _OcrHttpModule, expected_match: str
) -> None:
    with pytest.raises(RuntimeError, match=expected_match):
        module._assert_pinned_host("https://elsewhere.example/api")
    with pytest.raises(RuntimeError, match=expected_match):
        module._assert_pinned_host("not a url")


def test_ocr_subprocess_succeeds_on_pinned_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"pages": []}'

    class _GoodResponse:
        def __enter__(self) -> "_GoodResponse":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://api.mistral.ai/v1/ocr"

        def read(self, _: int) -> bytes:
            return payload

    def fake_urlopen(*args: object, **kwargs: object) -> _GoodResponse:
        assert kwargs.get("timeout"), "timeout must be positive and present"
        return _GoodResponse()

    monkeypatch.setattr(mistral_runtime.request, "urlopen", fake_urlopen)
    result = mistral_runtime._load_json(Request("https://api.mistral.ai/v1/ocr"))
    assert result == {"pages": []}


# ---------------------------------------------------------------------------
# Multipart filename sanitization
# ---------------------------------------------------------------------------

_BOUNDARY = "test-boundary-abc"
_FILE_CONTENT = b"%PDF-1.4 fake content"


def _build_body(filename: str) -> bytes:
    return mistral_runtime._build_multipart_body(
        boundary=_BOUNDARY,
        fields={"purpose": "ocr"},
        files={
            "file": {
                "filename": filename,
                "content_type": "application/octet-stream",
                "content": _FILE_CONTENT,
            }
        },
    )


def _disposition_line(body: bytes) -> str:
    for line in body.split(b"\r\n"):
        if b"Content-Disposition" in line and b"filename" in line:
            return line.decode("utf-8", errors="replace")
    return ""


def test_multipart_hostile_filename_no_injection() -> None:
    hostile = 'evil"\r\nX-Injected: 1\r\n--test-boundary-abc.pdf'
    body = _build_body(hostile)
    disposition = _disposition_line(body)

    # No raw CR or LF inside the disposition line
    assert "\r" not in disposition
    assert "\n" not in disposition
    # No unescaped double-quote that would close the filename token prematurely;
    # The quote from the hostile string must be escaped as \"
    assert 'filename="' in disposition
    # "X-Injected: 1" must not appear as a standalone header line (preceded by CRLF
    # and followed by CRLF, which would inject a real header into the part).
    # After sanitization it may appear inside the quoted filename value. That is safe.
    assert b"\r\nX-Injected: 1\r\n" not in body
    # The premature boundary must not appear as an actual boundary line
    assert b"\r\n--test-boundary-abc\r\n" in body  # real boundary still present once
    # The hostile boundary attempt (after sanitization) is embedded in the filename, not a real boundary
    lines = body.split(b"\r\n")
    boundary_lines = [ln for ln in lines if ln == b"--test-boundary-abc"]
    assert len(boundary_lines) == 2  # start of purpose field + start of file field
    # File content is still present
    assert _FILE_CONTENT in body


def test_multipart_empty_filename_uses_fallback() -> None:
    # Control-only or empty filename → fallback "document"
    body = _build_body("\x00\x01\x02\r\n")
    disposition = _disposition_line(body)
    assert 'filename="document"' in disposition
    assert _FILE_CONTENT in body


def test_multipart_normal_filename_bytes_unchanged() -> None:
    body = _build_body("report.pdf")
    expected_disposition = (
        'Content-Disposition: form-data; name="file"; filename="report.pdf"'
    )
    disposition = _disposition_line(body)
    assert disposition == expected_disposition
    assert _FILE_CONTENT in body
