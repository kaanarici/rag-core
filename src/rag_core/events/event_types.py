from __future__ import annotations

from typing import Final, Literal, TypeAlias

INGEST_STARTED_EVENT: Final[Literal["ingest.started"]] = "ingest.started"
INGEST_SKIPPED_EVENT: Final[Literal["ingest.skipped"]] = "ingest.skipped"
INGEST_BATCH_STARTED_EVENT: Final[Literal["ingest.batch.started"]] = (
    "ingest.batch.started"
)
INGEST_BATCH_PROGRESS_EVENT: Final[Literal["ingest.batch.progress"]] = (
    "ingest.batch.progress"
)
INGEST_BATCH_COMPLETED_EVENT: Final[Literal["ingest.batch.completed"]] = (
    "ingest.batch.completed"
)
INGEST_BATCH_FAILED_EVENT: Final[Literal["ingest.batch.failed"]] = (
    "ingest.batch.failed"
)
INGEST_COMPLETED_EVENT: Final[Literal["ingest.completed"]] = "ingest.completed"
FETCH_STARTED_EVENT: Final[Literal["fetch.started"]] = "fetch.started"
FETCH_COMPLETED_EVENT: Final[Literal["fetch.completed"]] = "fetch.completed"
FETCH_FAILED_EVENT: Final[Literal["fetch.failed"]] = "fetch.failed"

PARSE_COMPLETED_EVENT: Final[Literal["parse.completed"]] = "parse.completed"
OCR_APPLIED_EVENT: Final[Literal["ocr.applied"]] = "ocr.applied"
CHUNK_PRODUCED_EVENT: Final[Literal["chunk.produced"]] = "chunk.produced"
CONTEXTUALIZE_STARTED_EVENT: Final[Literal["contextualize.started"]] = (
    "contextualize.started"
)
CONTEXTUALIZE_COMPLETED_EVENT: Final[Literal["contextualize.completed"]] = (
    "contextualize.completed"
)
EMBED_REQUESTED_EVENT: Final[Literal["embed.requested"]] = "embed.requested"
EMBED_COMPLETED_EVENT: Final[Literal["embed.completed"]] = "embed.completed"
INDEX_UPSERTED_EVENT: Final[Literal["index.upserted"]] = "index.upserted"
INDEX_DELETED_EVENT: Final[Literal["index.deleted"]] = "index.deleted"

SEARCH_STARTED_EVENT: Final[Literal["search.started"]] = "search.started"
SEARCH_PLANNED_EVENT: Final[Literal["search.planned"]] = "search.planned"
SEARCH_STAGE_COMPLETED_EVENT: Final[Literal["search.stage.completed"]] = (
    "search.stage.completed"
)
SEARCH_COMPLETED_EVENT: Final[Literal["search.completed"]] = "search.completed"
RERANK_APPLIED_EVENT: Final[Literal["rerank.applied"]] = "rerank.applied"
SIDECAR_APPLIED_EVENT: Final[Literal["sidecar.applied"]] = "sidecar.applied"
STAGE_ERROR_EVENT: Final[Literal["stage.error"]] = "stage.error"

EmbeddingTraceEventType: TypeAlias = Literal[
    "embed.requested",
    "embed.completed",
]

__all__ = (
    "CHUNK_PRODUCED_EVENT",
    "CONTEXTUALIZE_COMPLETED_EVENT",
    "CONTEXTUALIZE_STARTED_EVENT",
    "EMBED_COMPLETED_EVENT",
    "EMBED_REQUESTED_EVENT",
    "EmbeddingTraceEventType",
    "FETCH_COMPLETED_EVENT",
    "FETCH_FAILED_EVENT",
    "FETCH_STARTED_EVENT",
    "INDEX_DELETED_EVENT",
    "INDEX_UPSERTED_EVENT",
    "INGEST_BATCH_COMPLETED_EVENT",
    "INGEST_BATCH_FAILED_EVENT",
    "INGEST_BATCH_PROGRESS_EVENT",
    "INGEST_BATCH_STARTED_EVENT",
    "INGEST_COMPLETED_EVENT",
    "INGEST_SKIPPED_EVENT",
    "INGEST_STARTED_EVENT",
    "OCR_APPLIED_EVENT",
    "PARSE_COMPLETED_EVENT",
    "RERANK_APPLIED_EVENT",
    "SEARCH_COMPLETED_EVENT",
    "SEARCH_PLANNED_EVENT",
    "SEARCH_STAGE_COMPLETED_EVENT",
    "SEARCH_STARTED_EVENT",
    "SIDECAR_APPLIED_EVENT",
    "STAGE_ERROR_EVENT",
)
