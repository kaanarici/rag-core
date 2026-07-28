from __future__ import annotations

from rag_core.core_models import estimate_token_count


def token_budget_for_char_limit(max_chars: int) -> int:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    return max(1, (max_chars + 3) // 4)


def fits_chunk_budget(text: str, *, max_chars: int) -> bool:
    return len(text) <= max_chars and estimate_token_count(
        text
    ) <= token_budget_for_char_limit(max_chars)


def fitting_overlap_tail(
    previous_text: str,
    following_text: str,
    *,
    separator: str,
    overlap: int,
    max_chars: int,
) -> str:
    """Return the largest requested tail that leaves room for following text."""
    if overlap <= 0 or not previous_text:
        return ""
    tail_start = max(0, len(previous_text) - min(overlap, max_chars))
    candidate = previous_text[tail_start:]
    if fits_chunk_budget(
        f"{candidate}{separator}{following_text}",
        max_chars=max_chars,
    ):
        return candidate

    low = tail_start
    high = len(previous_text)
    while low < high:
        middle = (low + high) // 2
        candidate = previous_text[middle:]
        if fits_chunk_budget(
            f"{candidate}{separator}{following_text}",
            max_chars=max_chars,
        ):
            high = middle
        else:
            low = middle + 1
    candidate = previous_text[low:]
    if not candidate:
        return ""
    return (
        candidate
        if fits_chunk_budget(
            f"{candidate}{separator}{following_text}",
            max_chars=max_chars,
        )
        else ""
    )


__all__ = [
    "fitting_overlap_tail",
    "fits_chunk_budget",
    "token_budget_for_char_limit",
]
