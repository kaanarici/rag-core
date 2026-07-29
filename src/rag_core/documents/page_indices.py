from __future__ import annotations


def normalize_page_indices(
    raw_indices: object,
    *,
    page_count: int | None = None,
    default_all_pages: bool = False,
    sort: bool = False,
) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()

    if isinstance(raw_indices, list):
        for raw_index in raw_indices:
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, int)
                or raw_index < 0
            ):
                continue
            if page_count is not None and raw_index >= page_count:
                continue
            if raw_index in seen:
                continue
            seen.add(raw_index)
            normalized.append(raw_index)

    if sort:
        normalized.sort()
    if normalized or not default_all_pages or not page_count or page_count <= 0:
        return normalized
    return list(range(page_count))


__all__ = ["normalize_page_indices"]
