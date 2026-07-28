from __future__ import annotations

import re

from rag_core.core_models import estimate_token_count

from .budget import fits_chunk_budget, token_budget_for_char_limit

_HEADER_DELIMITER_RE = re.compile(r"((?<!\\) \| )")


def header_units(payload: str) -> list[str]:
    parts = _HEADER_DELIMITER_RE.split(payload)
    return ["".join(parts[index : index + 2]) for index in range(0, len(parts), 2)]


def split_semantic_payload(
    units: list[str],
    *,
    prefix: str,
    max_chars: int,
) -> list[str]:
    _validate_additive_boundaries(units, prefix=prefix)
    token_budget = token_budget_for_char_limit(max_chars)
    prefix_chars = len(prefix)
    prefix_tokens = estimate_token_count(prefix)
    if prefix_chars > max_chars or prefix_tokens > token_budget:
        raise ValueError("XLSX provenance context leaves no room for content")

    fragments: list[str] = []
    active: list[str] = []
    active_chars = active_tokens = 0
    for unit in units:
        unit_chars = len(unit)
        unit_tokens = estimate_token_count(unit)
        if (
            prefix_chars + active_chars + unit_chars <= max_chars
            and prefix_tokens + active_tokens + unit_tokens <= token_budget
        ):
            active.append(unit)
            active_chars += unit_chars
            active_tokens += unit_tokens
            continue
        if active:
            fragments.append("".join(active))
            active = []
            active_chars = active_tokens = 0
        if (
            prefix_chars + unit_chars <= max_chars
            and prefix_tokens + unit_tokens <= token_budget
        ):
            active = [unit]
            active_chars = unit_chars
            active_tokens = unit_tokens
            continue
        unit_fragments = split_payload(unit, prefix=prefix, max_chars=max_chars)
        fragments.extend(unit_fragments[:-1])
        active = [unit_fragments[-1]]
        active_chars = len(active[0])
        active_tokens = estimate_token_count(active[0])
    if active:
        fragments.append("".join(active))
    if any(
        not fits_chunk_budget(f"{prefix}{fragment}", max_chars=max_chars)
        for fragment in fragments
    ):
        raise AssertionError("XLSX semantic fragment exceeded configured budget")
    return fragments


def _validate_additive_boundaries(units: list[str], *, prefix: str) -> None:
    if not units:
        return
    # Whitespace boundaries prevent the estimator's ASCII token runs from merging.
    if (
        not prefix
        or not prefix[-1].isspace()
        or not prefix.strip()
        or any(not unit.strip() for unit in units)
        or any(not unit[-1].isspace() for unit in units[:-1])
    ):
        raise ValueError("XLSX semantic units require additive token boundaries")


def split_payload(
    payload: str,
    *,
    prefix: str,
    max_chars: int,
) -> list[str]:
    fragments: list[str] = []
    offset = 0
    while offset < len(payload):
        low = offset + 1
        high = min(len(payload), offset + max_chars)
        best = offset
        while low <= high:
            middle = (low + high) // 2
            if fits_chunk_budget(
                f"{prefix}{payload[offset:middle]}",
                max_chars=max_chars,
            ):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == offset:
            raise ValueError("XLSX provenance context leaves no room for content")
        fragments.append(payload[offset:best])
        offset = best
    return fragments


__all__ = ["header_units", "split_payload", "split_semantic_payload"]
