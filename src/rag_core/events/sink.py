"""EventSink protocol.

Sinks receive engine lifecycle events synchronously. Sinks must not raise;
they should swallow their own errors. Async sinks can be implemented as
adapters that fire-and-forget.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Event


@runtime_checkable
class EventSink(Protocol):
    """Receiver for engine lifecycle events."""

    def emit(self, event: Event) -> None: ...
