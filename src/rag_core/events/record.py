"""Common metadata carried by every lifecycle event."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class EventRecord:
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""
    search_id: str = ""
