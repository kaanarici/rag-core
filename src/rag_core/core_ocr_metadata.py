from __future__ import annotations

from dataclasses import asdict

from rag_core.core_models import OcrMetadata

OCR_METADATA_KEY = "ocr"


def write_ocr_metadata(metadata: dict[str, object], ocr: OcrMetadata) -> None:
    """Stamp ``OcrMetadata`` into the document metadata under a single key."""

    metadata[OCR_METADATA_KEY] = asdict(ocr)


def read_ocr_metadata(metadata: dict[str, object]) -> OcrMetadata:
    """Read ``OcrMetadata`` from document metadata; default to empty if absent."""

    raw = metadata.get(OCR_METADATA_KEY)
    if not isinstance(raw, dict):
        return OcrMetadata()
    raw_pages = raw.get("pages_used")
    pages: tuple[int, ...]
    if isinstance(raw_pages, (list, tuple)):
        pages = tuple(
            int(p)
            for p in raw_pages
            if not isinstance(p, bool) and isinstance(p, int) and p >= 0
        )
    else:
        pages = ()
    page_count_raw = raw.get("page_count")
    page_count = (
        int(page_count_raw)
        if (
            not isinstance(page_count_raw, bool)
            and isinstance(page_count_raw, int)
            and page_count_raw >= 0
        )
        else 0
    )
    provider = raw.get("provider")
    model = raw.get("model")
    merge_mode = raw.get("merge_mode")
    return OcrMetadata(
        provider=str(provider) if provider is not None else None,
        model=str(model) if model is not None else None,
        pages_used=pages,
        page_count=page_count,
        merge_mode=str(merge_mode) if merge_mode is not None else None,
    )


__all__ = [
    "OCR_METADATA_KEY",
    "read_ocr_metadata",
    "write_ocr_metadata",
]
