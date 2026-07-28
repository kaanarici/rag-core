from __future__ import annotations

import re
from typing import NamedTuple


_FENCE_LINE_RE = re.compile(r" {0,3}(`{3,}|~{3,})([^\r\n]*)")


class MarkdownFenceState(NamedTuple):
    marker: str
    length: int


def advance_markdown_fence(
    line: str,
    active: MarkdownFenceState | None,
) -> tuple[bool, MarkdownFenceState | None]:
    match = _FENCE_LINE_RE.fullmatch(line)
    if match is None:
        return False, active
    marker = match.group(1)
    remainder = match.group(2)
    if active is None:
        if marker[0] == "`" and "`" in remainder:
            return False, None
        return True, MarkdownFenceState(marker=marker[0], length=len(marker))
    if (
        marker[0] == active.marker
        and len(marker) >= active.length
        and not remainder.strip()
    ):
        return True, None
    return False, active


__all__ = ["MarkdownFenceState", "advance_markdown_fence"]
