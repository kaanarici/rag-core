from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from urllib import error, request
from typing import cast

from rag_core.documents.converters.registry_maps import is_registered_image_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY is required")

    payload = json.load(__import__("sys").stdin)
    file_path = Path(str(payload["file_path"]))
    file_bytes = file_path.read_bytes()
    filename = str(payload.get("filename") or file_path.name)
    mime_type = str(payload.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/pdf")
    requested_page_indices = _normalize_page_indices(
        payload.get("requested_page_indices", payload.get("page_indices"))
    )
    whole_document_page_count = _whole_document_page_count(
        filename,
        mime_type,
        payload.get("metadata"),
    )
    whole_document_page_indices = list(range(whole_document_page_count))

    prompt = _build_prompt(requested_page_indices)
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(file_bytes).decode("utf-8"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ]
    }
    req = request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    response_payload = _load_json(req)
    markdown = _extract_text(response_payload)
    metadata = {
        "ocr_processed_entire_document": True,
        "ocr_page_selection_supported": False,
        "ocr_page_indices_ignored": bool(requested_page_indices),
        "ocr_source_mime_type": mime_type,
    }
    if whole_document_page_count > 0:
        metadata.update(
            {
                "ocr_pages_used_count": whole_document_page_count,
                "ocr_page_count": whole_document_page_count,
                "page_count": whole_document_page_count,
            }
        )
    result = {
        "markdown": markdown,
        "merge_mode": "replace",
        "provider_name": "gemini",
        "model_name": args.model,
        "pages_processed": whole_document_page_indices,
        "metadata": metadata,
    }
    print(json.dumps(result))
    return 0


def _build_prompt(page_indices: list[int]) -> str:
    if page_indices:
        return (
            "Convert this document to markdown. Preserve headings, lists, tables, and links. "
            "Return only markdown. Page filtering is not supported in this helper, so transcribe the document."
        )
    return (
        "Convert this document to markdown. Preserve headings, lists, tables, and links. "
        "Return only markdown with no explanation."
    )


def _load_json(req: request.Request) -> dict[str, object]:
    http_status: int | str
    try:
        with request.urlopen(req) as response:
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))
    except error.HTTPError as exc:
        http_status = _safe_http_status(exc)
    raise RuntimeError(f"Gemini OCR request failed ({http_status})")


def _safe_http_status(exc: error.HTTPError) -> int | str:
    code = getattr(exc, "code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return "unknown"
    return code


def _extract_text(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n\n".join(chunks)


def _whole_document_page_count(
    filename: str,
    mime_type: str,
    metadata: object = None,
) -> int:
    if is_registered_image_document(filename=filename, mime_type=mime_type):
        return 1
    if isinstance(metadata, dict):
        page_count = metadata.get("page_count")
        if not isinstance(page_count, bool) and isinstance(page_count, int) and page_count > 0:
            return page_count
    return 0


def _normalize_page_indices(raw_indices: object) -> list[int]:
    if not isinstance(raw_indices, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or raw_index in seen
        ):
            continue
        seen.add(raw_index)
        normalized.append(raw_index)
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
