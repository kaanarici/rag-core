from __future__ import annotations

import json
import os
import uuid
from typing import cast
from urllib import error, request
from urllib.parse import urlsplit

from rag_core.documents.http_errors import safe_http_status

_MISTRAL_API_HOST = "api.mistral.ai"
_MISTRAL_FETCH_TIMEOUT_SECONDS = 60.0
_MISTRAL_RESPONSE_MAX_BYTES = 64 * 1024 * 1024


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """Refuse 3xx redirects so the ``Authorization`` header is never re-sent.

    ``urlopen`` follows redirects with the default opener, which would deliver
    the bearer token to the redirect target before the post-hoc host pin can
    fire. Raising here stops the credential from ever leaving for another
    origin; the pin in :func:`_assert_pinned_host` stays as defense in depth.
    """

    def redirect_request(self, req: request.Request, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:  # noqa: E501
        raise error.HTTPError(
            req.full_url, code, "Mistral OCR refused redirect", headers, fp  # type: ignore[arg-type]
        )


# Route this subprocess's urlopen through a redirect-refusing opener so a 3xx
# never re-sends the credential header before _assert_pinned_host can fire.
request.install_opener(request.build_opener(_NoRedirectHandler))


def upload_file(*, api_key: str, filename: str, file_bytes: bytes) -> str:
    boundary = f"ragcore-{uuid.uuid4().hex}"
    body = _build_multipart_body(
        boundary=boundary,
        fields={"purpose": "ocr"},
        files={
            "file": {
                "filename": filename,
                "content_type": "application/octet-stream",
                "content": file_bytes,
            }
        },
    )
    url = f"https://{_MISTRAL_API_HOST}/v1/files"
    _assert_pinned_host(url)
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    payload = _load_json(req)
    file_id = payload.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("Mistral file upload did not return an id")
    return file_id


def run_ocr(
    *,
    api_key: str,
    model: str,
    file_id: str,
    page_indices: list[int],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "document": {"file_id": file_id},
    }
    if page_indices:
        payload["pages"] = page_indices
    url = f"https://{_MISTRAL_API_HOST}/v1/ocr"
    _assert_pinned_host(url)
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _load_json(req)


def _load_json(req: request.Request) -> dict[str, object]:
    http_status: int | str
    timeout = _timeout_seconds()
    max_bytes = _max_response_bytes()
    try:
        # The installed opener refuses redirects before the credential header
        # is re-sent (_NoRedirectHandler); the host pin below is defense in depth.
        with request.urlopen(req, timeout=timeout) as response:
            _assert_pinned_host(response.geturl())
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise RuntimeError(
                    f"Mistral OCR response exceeded max bytes ({max_bytes})"
                )
            return cast(dict[str, object], json.loads(raw.decode("utf-8")))
    except error.HTTPError as exc:
        http_status = safe_http_status(exc)
    except TimeoutError as exc:
        raise RuntimeError(
            f"Mistral OCR request timed out after {timeout:g}s"
        ) from exc
    raise RuntimeError(f"Mistral OCR request failed ({http_status})")


def _assert_pinned_host(url: str) -> None:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if host != _MISTRAL_API_HOST:
        raise RuntimeError(
            f"Mistral OCR refused unpinned host: {host or '<missing>'}"
        )


def _timeout_seconds() -> float:
    raw = os.environ.get("RAG_CORE_OCR_FETCH_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _MISTRAL_FETCH_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _MISTRAL_FETCH_TIMEOUT_SECONDS
    return value if value > 0 else _MISTRAL_FETCH_TIMEOUT_SECONDS


def _max_response_bytes() -> int:
    raw = os.environ.get("RAG_CORE_OCR_FETCH_MAX_BYTES", "").strip()
    if not raw:
        return _MISTRAL_RESPONSE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _MISTRAL_RESPONSE_MAX_BYTES
    return value if value > 0 else _MISTRAL_RESPONSE_MAX_BYTES


def _multipart_safe(value: str, *, fallback: str) -> str:
    cleaned = "".join(ch for ch in value if ch >= " " and ch != "\x7f")
    cleaned = cleaned.replace("\\", "\\\\").replace('"', '\\"')
    return cleaned or fallback


def _build_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    files: dict[str, dict[str, str | bytes]],
) -> bytes:
    chunks: list[bytes] = []
    for key, value in fields.items():
        safe_key = _multipart_safe(key, fallback=key)
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{safe_key}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for key, spec in files.items():
        safe_key = _multipart_safe(key, fallback=key)
        safe_filename = _multipart_safe(str(spec["filename"]), fallback="document")
        safe_content_type = "".join(
            ch for ch in str(spec["content_type"]) if ch not in "\r\n"
        )
        content = spec["content"]
        if not isinstance(content, bytes):
            raise TypeError("multipart file content must be bytes")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{safe_key}"; filename="{safe_filename}"\r\n'
                    f"Content-Type: {safe_content_type}\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)
