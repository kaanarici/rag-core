"""EventSink protocol.

Sinks receive engine lifecycle events synchronously. Sinks must not raise;
they should swallow their own errors. Async sinks can be implemented as
adapters that fire-and-forget.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast, Protocol, runtime_checkable

from .types import AuditContext, Event


@runtime_checkable
class EventSink(Protocol):
    """Receiver for engine lifecycle events."""

    def emit(self, event: Event) -> None: ...


class CorrelatedSink:
    """Apply caller correlation; an Engine-minted search ID takes precedence."""

    def __init__(
        self,
        sink: EventSink,
        *,
        audit_context: AuditContext | None = None,
        search_id: str = "",
        ingest_id: str = "",
    ) -> None:
        self._sink = sink
        self._context = audit_context or AuditContext()
        self._search_id = search_id
        self._ingest_id = ingest_id or self._context.ingest_id

    def emit(self, event: Event) -> None:
        values = {
            "actor": self._context.actor,
            "request_id": self._context.request_id,
            "ingest_id": self._ingest_id,
        }
        updates = {
            name: value
            for name, value in values.items()
            if value and not getattr(event, name)
        }
        if self._search_id:
            updates["search_id"] = self._search_id
        if updates:
            event = cast(Event, replace(cast(Any, event), **updates))
        self._sink.emit(event)
