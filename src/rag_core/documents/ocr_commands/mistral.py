from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from pathlib import Path

from rag_core.documents.ocr_commands.mistral_runtime import run_ocr, upload_file

_PAGE_HEADING_RE = re.compile(r"^#{1,6}\s+Page\s+([1-9]\d*)\s*$", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-ocr-latest")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY is required")

    payload = json.load(__import__("sys").stdin)
    file_path = Path(str(payload["file_path"]))
    file_bytes = file_path.read_bytes()
    filename = str(payload.get("filename") or file_path.name)
    mime_type = str(
        payload.get("mime_type")
        or mimetypes.guess_type(filename)[0]
        or "application/pdf"
    )
    page_indices = _normalize_page_indices(payload.get("page_indices"))

    file_id = upload_file(
        api_key=api_key,
        filename=filename,
        file_bytes=file_bytes,
    )
    ocr_payload = run_ocr(
        api_key=api_key,
        model=args.model,
        file_id=file_id,
        page_indices=page_indices,
    )
    raw_pages = ocr_payload.get("pages")
    page_count = len(raw_pages) if isinstance(raw_pages, list) else 0
    markdown = _collect_markdown(raw_pages, page_indices)
    pages_processed = _processed_page_indices(raw_pages, page_indices)
    result = {
        "markdown": markdown,
        "merge_mode": "append" if page_indices else "replace",
        "provider_name": "mistral",
        "model_name": args.model,
        "pages_processed": pages_processed,
        "metadata": {
            "ocr_source_mime_type": mime_type,
            "ocr_page_count": len(pages_processed) if page_indices else page_count,
        },
    }
    print(json.dumps(result))
    return 0


def _collect_markdown(raw_pages: object, requested_indices: list[int]) -> str:
    if not isinstance(raw_pages, list):
        return ""
    selected: list[str] = []
    selected_indices = set(requested_indices)
    seen_indices: set[int] = set()
    for fallback_index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            continue
        raw_markdown = raw_page.get("markdown")
        if not _has_markdown(raw_markdown):
            continue
        assert isinstance(raw_markdown, str)
        page_index = _response_page_index(raw_page.get("index"), fallback_index)
        if selected_indices and page_index not in selected_indices:
            continue
        if selected_indices and page_index in seen_indices:
            continue
        seen_indices.add(page_index)
        markdown = raw_markdown.strip()
        selected.append(_with_page_heading(markdown, page_index))
    return "\n\n".join(selected)


def _with_page_heading(markdown: str, page_index: int) -> str:
    first_line = markdown.splitlines()[0].strip() if markdown else ""
    match = _PAGE_HEADING_RE.match(first_line)
    if match is not None and int(match.group(1)) == page_index + 1:
        return markdown
    return "## Page %d\n\n%s" % (page_index + 1, markdown)


def _response_page_index(value: object, fallback_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return fallback_index
    return value - 1


def _default_page_indices(raw_pages: object) -> list[int]:
    if not isinstance(raw_pages, list):
        return []
    processed: list[int] = []
    seen: set[int] = set()
    for fallback_index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            continue
        if not _has_markdown(raw_page.get("markdown")):
            continue
        page_index = _response_page_index(raw_page.get("index"), fallback_index)
        if page_index in seen:
            continue
        seen.add(page_index)
        processed.append(page_index)
    return processed


def _processed_page_indices(
    raw_pages: object, requested_indices: list[int]
) -> list[int]:
    if not requested_indices:
        return _default_page_indices(raw_pages)
    if not isinstance(raw_pages, list):
        return []
    requested = set(requested_indices)
    processed: list[int] = []
    seen: set[int] = set()
    for fallback_index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            continue
        if not _has_markdown(raw_page.get("markdown")):
            continue
        page_index = _response_page_index(raw_page.get("index"), fallback_index)
        if page_index not in requested or page_index in seen:
            continue
        seen.add(page_index)
        processed.append(page_index)
    return processed


def _has_markdown(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
