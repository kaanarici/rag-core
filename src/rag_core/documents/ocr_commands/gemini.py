from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import cast
from urllib import error, request
from urllib.parse import urlsplit

from rag_core.documents.converters.registry_maps import is_registered_image_document
from rag_core.documents.http_errors import safe_http_status
from rag_core.documents.ocr_provider_names import (
    DEFAULT_GEMINI_OCR_MODEL,
    GEMINI_OCR_PROVIDER,
)
from rag_core.documents.page_indices import normalize_page_indices
from rag_core.provider_api_keys import GEMINI_API_KEY_ENVS, first_configured_api_key

_GEMINI_API_HOST = "generativelanguage.googleapis.com"
_GEMINI_FETCH_TIMEOUT_SECONDS = 60.0
_GEMINI_RESPONSE_MAX_BYTES = 64 * 1024 * 1024


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """Refuse 3xx redirects so the ``x-goog-api-key`` header is never re-sent.

    ``urlopen`` follows redirects with the default opener, which would deliver
    the credential header to the redirect target before the post-hoc host pin
    can fire. Raising here stops the credential from ever leaving for another
    origin; the pin in :func:`_assert_pinned_host` stays as defense in depth.
    """

    def redirect_request(self, req: request.Request, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:  # noqa: E501
        raise error.HTTPError(
            req.full_url, code, "Gemini OCR refused redirect", headers, fp  # type: ignore[arg-type]
        )


# Route this subprocess's urlopen through a redirect-refusing opener so a 3xx
# never re-sends the credential header before _assert_pinned_host can fire.
request.install_opener(request.build_opener(_NoRedirectHandler))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_GEMINI_OCR_MODEL)
    args = parser.parse_args()

    api_key = first_configured_api_key(GEMINI_API_KEY_ENVS)
    if not api_key:
        raise SystemExit(f"{' or '.join(GEMINI_API_KEY_ENVS)} is required")

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
    upload_file_bytes = file_bytes
    pages_processed = whole_document_page_indices
    processed_entire_document = True

    if (
        _is_pdf_document(filename=filename, mime_type=mime_type)
        and requested_page_indices
        and set(requested_page_indices) != set(whole_document_page_indices)
    ):
        subset = _subset_pdf_pages(file_bytes, requested_page_indices)
        if subset is not None:
            upload_file_bytes, pages_processed, whole_document_page_count = subset
            processed_entire_document = False

    prompt = _build_prompt(
        pages_processed if not processed_entire_document else requested_page_indices,
        pdf_subset=not processed_entire_document,
    )
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(upload_file_bytes).decode("utf-8"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ]
    }
    url = (
        f"https://{_GEMINI_API_HOST}/v1beta/models/"
        f"{args.model}:generateContent"
    )
    _assert_pinned_host(url)
    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    response_payload = _load_json(req)
    markdown = _extract_text(response_payload)
    if processed_entire_document:
        markdown = _ensure_page_heading(
            markdown,
            page_count=whole_document_page_count,
        )
    else:
        markdown = _with_original_page_headings(markdown, pages_processed)
    metadata = {
        "ocr_processed_entire_document": processed_entire_document,
        "ocr_page_selection_supported": not processed_entire_document,
        "ocr_page_indices_ignored": bool(
            requested_page_indices and processed_entire_document
        ),
        "ocr_source_mime_type": mime_type,
    }
    if processed_entire_document and whole_document_page_count > 0:
        metadata.update(
            {
                "ocr_pages_used_count": whole_document_page_count,
                "ocr_page_count": whole_document_page_count,
                "page_count": whole_document_page_count,
            }
        )
    elif not processed_entire_document:
        metadata.update(
            {
                "ocr_pages_used_count": len(pages_processed),
                "ocr_page_count": len(pages_processed),
                "page_count": whole_document_page_count,
            }
        )
    result = {
        "markdown": markdown,
        "merge_mode": "replace" if processed_entire_document else "append",
        "provider_name": GEMINI_OCR_PROVIDER,
        "model_name": args.model,
        "pages_processed": pages_processed,
        "metadata": metadata,
    }
    print(json.dumps(result))
    return 0


def _build_prompt(page_indices: list[int], *, pdf_subset: bool = False) -> str:
    if pdf_subset and page_indices:
        page_numbers = ", ".join(str(page_index + 1) for page_index in page_indices)
        return (
            "Convert this PDF excerpt to markdown. Preserve headings, lists, tables, and links. "
            "Return only markdown. Prefix each page with a `## Page N` heading using these "
            f"original document page numbers in order: {page_numbers}."
        )
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
                    f"Gemini OCR response exceeded max bytes ({max_bytes})"
                )
            return cast(dict[str, object], json.loads(raw.decode("utf-8")))
    except error.HTTPError as exc:
        http_status = safe_http_status(exc)
    except TimeoutError as exc:
        raise RuntimeError(
            f"Gemini OCR request timed out after {timeout:g}s"
        ) from exc
    raise RuntimeError(f"Gemini OCR request failed ({http_status})")


def _assert_pinned_host(url: str) -> None:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if host != _GEMINI_API_HOST:
        raise RuntimeError(
            f"Gemini OCR refused unpinned host: {host or '<missing>'}"
        )


def _timeout_seconds() -> float:
    raw = os.environ.get("RAG_CORE_OCR_FETCH_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _GEMINI_FETCH_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _GEMINI_FETCH_TIMEOUT_SECONDS
    return value if value > 0 else _GEMINI_FETCH_TIMEOUT_SECONDS


def _max_response_bytes() -> int:
    raw = os.environ.get("RAG_CORE_OCR_FETCH_MAX_BYTES", "").strip()
    if not raw:
        return _GEMINI_RESPONSE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _GEMINI_RESPONSE_MAX_BYTES
    return value if value > 0 else _GEMINI_RESPONSE_MAX_BYTES


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


_PAGE_HEADING_RE = re.compile(r"^## Page ([1-9]\d*)\s*$", re.MULTILINE)


def _ensure_page_heading(
    markdown: str, *, page_count: int, page_number: int = 1
) -> str:
    """Prepend a ``## Page 1`` heading so single-page Gemini output carries a locator.

    Gemini returns the whole document as one markdown blob, so multi-page
    documents cannot be partitioned reliably here. Only single-page (image)
    inputs are wrapped. Pre-existing ``## Page N`` markup is preserved.
    """
    stripped = markdown.strip()
    if not stripped:
        return markdown
    if _PAGE_HEADING_RE.search(stripped) is not None:
        return markdown
    if page_count != 1:
        return markdown
    return "## Page %d\n\n%s" % (page_number, stripped)


def _with_original_page_headings(markdown: str, page_indices: list[int]) -> str:
    page_numbers = [page_index + 1 for page_index in page_indices]
    remapped = _remap_ordinal_page_headings(markdown, page_numbers)
    page_number = page_numbers[0] if len(page_numbers) == 1 else 1
    return _ensure_page_heading(
        remapped,
        page_count=len(page_numbers),
        page_number=page_number,
    )


def _remap_ordinal_page_headings(markdown: str, page_numbers: list[int]) -> str:
    matches = list(_PAGE_HEADING_RE.finditer(markdown))
    if [int(match.group(1)) for match in matches] != list(
        range(1, len(page_numbers) + 1)
    ):
        return markdown
    chunks: list[str] = []
    position = 0
    for match, page_number in zip(matches, page_numbers, strict=True):
        chunks.append(markdown[position : match.start(1)])
        chunks.append(str(page_number))
        position = match.end(1)
    chunks.append(markdown[position:])
    return "".join(chunks)


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
    return normalize_page_indices(raw_indices)


def _is_pdf_document(*, filename: str, mime_type: str) -> bool:
    if mime_type.strip().lower() == "application/pdf":
        return True
    return filename.strip().lower().endswith(".pdf")


def _subset_pdf_pages(
    file_bytes: bytes, page_indices: list[int]
) -> tuple[bytes, list[int], int] | None:
    if not page_indices:
        return None
    import fitz

    with fitz.open(stream=file_bytes, filetype="pdf") as source:
        page_count = cast(int, source.page_count)
        selected = normalize_page_indices(page_indices, page_count=page_count)
        if not selected or len(selected) == page_count:
            return None
        subset = fitz.open()
        try:
            for page_index in selected:
                subset.insert_pdf(source, from_page=page_index, to_page=page_index)
            return cast(bytes, subset.tobytes()), selected, page_count
        finally:
            subset.close()

if __name__ == "__main__":
    raise SystemExit(main())
