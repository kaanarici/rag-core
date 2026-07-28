from __future__ import annotations


MAX_PDF_PAGE_COUNT = 10_000


def validate_pdf_page_count(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(_page_count_error())
    if isinstance(value, int):
        page_count = value
    elif isinstance(value, float) and value.is_integer():
        page_count = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        page_count = int(value.strip())
    else:
        raise ValueError(_page_count_error())
    if page_count <= 0 or page_count > MAX_PDF_PAGE_COUNT:
        raise ValueError(_page_count_error())
    return page_count


def _page_count_error() -> str:
    return f"page_count must be between 1 and {MAX_PDF_PAGE_COUNT}"


__all__ = ["MAX_PDF_PAGE_COUNT", "validate_pdf_page_count"]
