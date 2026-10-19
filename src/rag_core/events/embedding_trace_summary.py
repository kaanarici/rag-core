from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from rag_core.events.event_types import (
    EMBED_COMPLETED_EVENT,
    EMBED_REQUESTED_EVENT,
)
from rag_core.events.types import Event, EmbedRequested, EmbedCompleted
from rag_core.events.trace_payload_fields import (
    float_field,
    int_field,
    safe_optional_label_field,
    str_field,
)
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    RETRIEVAL_CHANNELS,
    RetrievalChannel,
)


@dataclass(frozen=True)
class EmbeddingTraceSummary:
    requested_event_count: int = 0
    completed_event_count: int = 0
    requested_text_count: int = 0
    completed_text_count: int = 0
    dense_completed_text_count: int = 0
    sparse_completed_text_count: int = 0
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
            "duration_ms": self.duration_ms,
            "providers": list(self.providers),
            "models": list(self.models),
        }


def summarize_embedding_trace_payloads(
    payloads: Iterable[Mapping[str, object]],
) -> EmbeddingTraceSummary:
    requested_events = completed_events = requested_texts = completed_texts = 0
    dense_texts = sparse_texts = 0
    duration = 0.0
    providers: dict[str, None] = {}
    models: dict[str, None] = {}
    for payload in payloads:
        event_type = payload.get("event_type")
        if event_type not in (EMBED_REQUESTED_EVENT, EMBED_COMPLETED_EVENT):
            continue
        provider = safe_optional_label_field(payload, "provider")
        model = safe_optional_label_field(payload, "model")
        count = int_field(payload, "text_count")
        role = _embedding_role_field(payload)
        if provider:
            providers[provider] = None
        if model:
            models[model] = None
        if event_type == EMBED_REQUESTED_EVENT:
            requested_events += 1
            requested_texts += count
        else:
            completed_events += 1
            completed_texts += count
            duration += float_field(payload, "duration_ms")
            if role == DENSE_RETRIEVAL_CHANNEL:
                dense_texts += count
            else:
                sparse_texts += count
    return EmbeddingTraceSummary(
        requested_event_count=requested_events,
        completed_event_count=completed_events,
        requested_text_count=requested_texts,
        completed_text_count=completed_texts,
        dense_completed_text_count=dense_texts,
        sparse_completed_text_count=sparse_texts,
        duration_ms=duration,
        providers=tuple(providers),
        models=tuple(models),
    )


def summarize_embedding_trace(events: Iterable[Event]) -> EmbeddingTraceSummary:
    return summarize_embedding_trace_payloads(
        asdict(event)
        for event in events
        if isinstance(event, (EmbedRequested, EmbedCompleted))
    )


def _embedding_role_field(payload: Mapping[str, object]) -> RetrievalChannel:
    value = str_field(payload, "role")
    if value not in RETRIEVAL_CHANNELS:
        expected = " or ".join(RETRIEVAL_CHANNELS)
        raise ValueError(f"trace field role must be {expected}")
    return value


__all__ = [
    "EmbeddingTraceSummary",
    "summarize_embedding_trace",
    "summarize_embedding_trace_payloads",
]
