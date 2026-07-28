from __future__ import annotations

import re
from typing import NamedTuple

_ATX_HEADING_RE = re.compile(
    r"(?P<indent> {0,3})(?P<marker>#{1,6})[^\S\r\n]+(?P<title>.*)"
)
_ATX_CLOSER_RE = re.compile(r"[^\S\r\n]+#+[^\S\r\n]*$")


class AtxHeading(NamedTuple):
    indent: str
    level: int
    title: str


def parse_atx_heading(line: str) -> AtxHeading | None:
    match = _ATX_HEADING_RE.fullmatch(line)
    if match is None:
        return None
    title = _ATX_CLOSER_RE.sub("", match.group("title")).rstrip()
    if not title:
        return None
    return AtxHeading(
        indent=match.group("indent"),
        level=len(match.group("marker")),
        title=title,
    )


__all__ = ["AtxHeading", "parse_atx_heading"]
