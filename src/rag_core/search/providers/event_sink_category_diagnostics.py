"""Event sink provider diagnostics."""

from __future__ import annotations

from rag_core.events.sinks import (
    DEFAULT_EVENT_SINK_PROVIDER,
    EVENT_SINK_PROVIDER_ORDER,
    EventBuffer,
    JsonlSink,
    LoggingSink,
    MultiSink,
    OpenTelemetrySink,
)
from rag_core.provider_package_names import OPENTELEMETRY_TRACE_PACKAGE

from .diagnostic_support import (
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_MATURITY,
    MATURITY_DISABLED,
    MATURITY_OPTIONAL,
    MATURITY_UTILITY,
)
from .provider_category_helpers import (
    add_injected_provider,
    normalize_runtime_provider,
    package_available,
)

_PACKAGE_BY_PROVIDER = {
    OpenTelemetrySink.provider_name: OPENTELEMETRY_TRACE_PACKAGE,
}
_EVENT_SINK_RUNTIME_CONFIG = {
    LoggingSink.provider_name: "Engine(..., event_sink=LoggingSink(...))",
    JsonlSink.provider_name: "Engine(..., event_sink=JsonlSink(...))",
    EventBuffer.provider_name: "Engine(..., event_sink=EventBuffer())",
    MultiSink.provider_name: "Engine(..., event_sink=MultiSink(...))",
}


def describe_event_sink_provider_diagnostics(
    *,
    runtime_provider: str | None = None,
) -> dict[str, object]:
    configured = normalize_runtime_provider(
        runtime_provider,
        default=DEFAULT_EVENT_SINK_PROVIDER,
    ) or DEFAULT_EVENT_SINK_PROVIDER
    providers: dict[str, object] = {
        DEFAULT_EVENT_SINK_PROVIDER: {
            FIELD_MATURITY: MATURITY_DISABLED,
            FIELD_CONFIGURED: configured == DEFAULT_EVENT_SINK_PROVIDER,
            FIELD_PACKAGE_AVAILABLE: True,
            FIELD_RUNTIME_CONFIG: "Engine(..., event_sink=None)",
        },
        LoggingSink.provider_name: _event_sink_diagnostics(
            LoggingSink.provider_name,
            configured=configured,
        ),
        JsonlSink.provider_name: _event_sink_diagnostics(
            JsonlSink.provider_name,
            configured=configured,
        ),
        EventBuffer.provider_name: _event_sink_diagnostics(
            EventBuffer.provider_name,
            configured=configured,
        ),
        MultiSink.provider_name: _event_sink_diagnostics(
            MultiSink.provider_name,
            configured=configured,
        ),
        OpenTelemetrySink.provider_name: {
            FIELD_MATURITY: MATURITY_OPTIONAL,
            FIELD_CONFIGURED: configured == OpenTelemetrySink.provider_name,
            FIELD_PACKAGE_AVAILABLE: package_available(
                OpenTelemetrySink.provider_name,
                packages_by_provider=_PACKAGE_BY_PROVIDER,
            ),
            FIELD_RUNTIME_CONFIG: "Engine(..., event_sink=OpenTelemetrySink())",
        },
    }
    add_injected_provider(
        providers,
        configured,
        known=EVENT_SINK_PROVIDER_ORDER,
    )
    return {
        FIELD_CONFIGURED: configured,
        FIELD_REGISTERED: [],
        FIELD_PROVIDERS: providers,
    }


def _event_sink_diagnostics(provider: str, *, configured: str) -> dict[str, object]:
    return {
        FIELD_MATURITY: MATURITY_UTILITY,
        FIELD_CONFIGURED: provider == configured,
        FIELD_PACKAGE_AVAILABLE: True,
        FIELD_RUNTIME_CONFIG: _EVENT_SINK_RUNTIME_CONFIG[provider],
    }


__all__ = ["describe_event_sink_provider_diagnostics"]
