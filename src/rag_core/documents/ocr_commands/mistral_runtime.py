from __future__ import annotations

import json
import uuid
from typing import cast
from urllib import error, request

from rag_core.documents.http_errors import safe_http_status


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
    req = request.Request(
        "https://api.mistral.ai/v1/files",
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
    req = request.Request(
        "https://api.mistral.ai/v1/ocr",
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
    try:
        with request.urlopen(req) as response:
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))
    except error.HTTPError as exc:
        http_status = safe_http_status(exc)
    raise RuntimeError(f"Mistral OCR request failed ({http_status})")


def _build_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    files: dict[str, dict[str, str | bytes]],
) -> bytes:
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for key, spec in files.items():
        filename = str(spec["filename"])
        content_type = str(spec["content_type"])
        content = spec["content"]
        if not isinstance(content, bytes):
            raise TypeError("multipart file content must be bytes")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)
