"""Public engine lifecycle event record entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from rag_core.events.event_types import AUDIT_EVENT_TYPES as AUDIT_EVENT_TYPES
from rag_core.events.document_events import (
    ChunkProduced as ChunkProduced,
    ContextualizeCompleted as ContextualizeCompleted,
    ContextualizeStarted as ContextualizeStarted,
    EmbedCompleted as EmbedCompleted,
    EmbedRequested as EmbedRequested,
    IndexDeleted as IndexDeleted,
    IndexUpserted as IndexUpserted,
    OcrApplied as OcrApplied,
    ParseCompleted as ParseCompleted,
)
from rag_core.events.ingest_events import (
    FetchCompleted as FetchCompleted,
    FetchFailed as FetchFailed,
    FetchStarted as FetchStarted,
    IngestBatchCompleted as IngestBatchCompleted,
    IngestBatchFailed as IngestBatchFailed,
    IngestBatchProgress as IngestBatchProgress,
    IngestBatchStarted as IngestBatchStarted,
    IngestCompleted as IngestCompleted,
    IngestSkipped as IngestSkipped,
    IngestStarted as IngestStarted,
)
from rag_core.events.search_events import (
    LexicalSidecarBoundExceeded as LexicalSidecarBoundExceeded,
    NeighborExpandSkipped as NeighborExpandSkipped,
    RerankApplied as RerankApplied,
    SearchCompleted as SearchCompleted,
    SearchPlanned as SearchPlanned,
    SearchStageCompleted as SearchStageCompleted,
    SearchStarted as SearchStarted,
    SidecarApplied as SidecarApplied,
    StageError as StageError,
)

@dataclass(frozen=True)
class AuditContext:
    """Caller-supplied correlation for audit trails.

    Threaded into engine entrypoints (ingest/search/delete) so every emitted
    event carries the same actor/request/ingest correlation. ``request_id`` is
    typically minted at the gateway; ``actor`` is the human/service identity
    the gateway authenticated; ``ingest_id`` and
    ``search_id`` are workflow-scoped (``search_id`` is minted internally by
    the pipeline runner, ``ingest_id`` is accepted from the caller so a batch
    can correlate per-document events back to the originating batch).

    All fields are optional. ``None`` means "not provided"; the engine must
    not invent these identifiers. Engine-minted identifiers (``search_id``)
    flow on the event objects themselves, not here.
    """

    actor: str | None = None
    request_id: str | None = None
    ingest_id: str | None = None
    search_id: str | None = None


Event = Union[
    IngestStarted,
    IngestSkipped,
    IngestBatchStarted,
    IngestBatchProgress,
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestCompleted,
    FetchStarted,
    FetchCompleted,
    FetchFailed,
    ParseCompleted,
    OcrApplied,
    ChunkProduced,
    ContextualizeStarted,
    ContextualizeCompleted,
    EmbedRequested,
    EmbedCompleted,
    IndexUpserted,
    IndexDeleted,
    SearchStarted,
    SearchPlanned,
    SearchStageCompleted,
    SearchCompleted,
    RerankApplied,
    SidecarApplied,
    NeighborExpandSkipped,
    LexicalSidecarBoundExceeded,
    StageError,
]

__all__ = (
    "AUDIT_EVENT_TYPES",
    "AuditContext",
    "ChunkProduced",
    "ContextualizeCompleted",
    "ContextualizeStarted",
    "EmbedCompleted",
    "EmbedRequested",
    "Event",
    "FetchCompleted",
    "FetchFailed",
    "FetchStarted",
    "IndexDeleted",
    "IndexUpserted",
    "IngestBatchCompleted",
    "IngestBatchFailed",
    "IngestBatchProgress",
    "IngestBatchStarted",
    "IngestCompleted",
    "IngestSkipped",
    "IngestStarted",
    "LexicalSidecarBoundExceeded",
    "NeighborExpandSkipped",
    "OcrApplied",
    "ParseCompleted",
    "RerankApplied",
    "SearchCompleted",
    "SearchPlanned",
    "SearchStageCompleted",
    "SearchStarted",
    "SidecarApplied",
    "StageError",
)
