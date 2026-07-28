from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from rag_core.events.event_types import (
    EMBED_COMPLETED_EVENT,
    EMBED_REQUESTED_EVENT,
    EmbeddingTraceEventType,
)
from rag_core.events.types import Event
from rag_core.events.trace_payload_fields import (
    float_field,
    int_field,
    safe_optional_label_field,
    str_field,
)
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    RETRIEVAL_CHANNELS,
    SPARSE_RETRIEVAL_CHANNEL,
    RetrievalChannel,
)


@dataclass(frozen=True)
class EmbeddingTraceEvent:
    event_type: EmbeddingTraceEventType
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: RetrievalChannel = DENSE_RETRIEVAL_CHANNEL
    duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_bypasses: int = 0


@dataclass(frozen=True)
class EmbeddingTraceSummary:
    requested_event_count: int = 0
    completed_event_count: int = 0
    requested_text_count: int = 0
    completed_text_count: int = 0
    dense_completed_text_count: int = 0
    sparse_completed_text_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_bypasses: int = 0
    duration_ms: float = 0.0
    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()

    @property
    def has_events(self) -> bool:
        return bool(self.requested_event_count or self.completed_event_count)

    def to_payload(self) -> dict[str, object]:
        return {
            "requested_event_count": self.requested_event_count,
            "completed_event_count": self.completed_event_count,
            "requested_text_count": self.requested_text_count,
            "completed_text_count": self.completed_text_count,
            "dense_completed_text_count": self.dense_completed_text_count,
            "sparse_completed_text_count": self.sparse_completed_text_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
            "cache_bypasses": self.cache_bypasses,
            "duration_ms": self.duration_ms,
            "providers": list(self.providers),
            "models": list(self.models),
        }


def summarize_embedding_trace_payloads(
    payloads: Iterable[Mapping[str, object]],
) -> EmbeddingTraceSummary:
    events = [
        event
        for payload in payloads
        if (event := embedding_trace_event_from_payload(payload)) is not None
    ]
    providers: list[str] = []
    models: list[str] = []
    for event in events:
        _append_unique(providers, event.provider)
        _append_unique(models, event.model)
    return EmbeddingTraceSummary(
        requested_event_count=sum(
            1 for event in events if event.event_type == EMBED_REQUESTED_EVENT
        ),
        completed_event_count=sum(
            1 for event in events if event.event_type == EMBED_COMPLETED_EVENT
        ),
        requested_text_count=sum(
            event.text_count
            for event in events
            if event.event_type == EMBED_REQUESTED_EVENT
        ),
        completed_text_count=sum(
            event.text_count
            for event in events
            if event.event_type == EMBED_COMPLETED_EVENT
        ),
        dense_completed_text_count=sum(
            event.text_count
            for event in events
            if event.event_type == EMBED_COMPLETED_EVENT
            and event.role == DENSE_RETRIEVAL_CHANNEL
        ),
        sparse_completed_text_count=sum(
            event.text_count
            for event in events
            if event.event_type == EMBED_COMPLETED_EVENT
            and event.role == SPARSE_RETRIEVAL_CHANNEL
        ),
        cache_hits=sum(event.cache_hits for event in events),
        cache_misses=sum(event.cache_misses for event in events),
        cache_writes=sum(event.cache_writes for event in events),
        cache_bypasses=sum(event.cache_bypasses for event in events),
        duration_ms=sum(event.duration_ms for event in events),
        providers=tuple(providers),
        models=tuple(models),
    )


def summarize_embedding_trace(events: Iterable[Event]) -> EmbeddingTraceSummary:
    return summarize_embedding_trace_payloads(asdict(event) for event in events)


def embedding_trace_event_from_payload(
    payload: Mapping[str, object],
) -> EmbeddingTraceEvent | None:
    event_type = payload.get("event_type")
    if event_type == EMBED_REQUESTED_EVENT:
        return EmbeddingTraceEvent(
            event_type=EMBED_REQUESTED_EVENT,
            provider=safe_optional_label_field(payload, "provider"),
            model=safe_optional_label_field(payload, "model"),
            text_count=int_field(payload, "text_count"),
            role=_embedding_role_field(payload),
        )
    if event_type == EMBED_COMPLETED_EVENT:
        return EmbeddingTraceEvent(
            event_type=EMBED_COMPLETED_EVENT,
            provider=safe_optional_label_field(payload, "provider"),
            model=safe_optional_label_field(payload, "model"),
            text_count=int_field(payload, "text_count"),
            role=_embedding_role_field(payload),
            duration_ms=float_field(payload, "duration_ms"),
            cache_hits=int_field(payload, "cache_hits"),
            cache_misses=int_field(payload, "cache_misses"),
            cache_writes=int_field(payload, "cache_writes"),
            cache_bypasses=int_field(payload, "cache_bypasses"),
        )
    return None


def _embedding_role_field(payload: Mapping[str, object]) -> RetrievalChannel:
    value = str_field(payload, "role")
    if value not in RETRIEVAL_CHANNELS:
        expected = " or ".join(RETRIEVAL_CHANNELS)
        raise ValueError(f"trace field role must be {expected}")
    return value


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


__all__ = [
    "EmbeddingTraceSummary",
    "embedding_trace_event_from_payload",
    "summarize_embedding_trace",
    "summarize_embedding_trace_payloads",
]
