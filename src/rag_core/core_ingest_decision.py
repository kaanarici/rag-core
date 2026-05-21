from __future__ import annotations

from dataclasses import dataclass

from rag_core.core_lifecycle import resolve_ingest_state
from rag_core.core_models import ProcessingFingerprint
from rag_core.search.types import StoredDocumentRecord, VectorStore


@dataclass(frozen=True)
class IngestDecision:
    existing: StoredDocumentRecord | None
    ingest_state: str
    should_index: bool


async def resolve_ingest_decision(
    *,
    store: VectorStore,
    namespace: str,
    corpus_id: str,
    document_id: str,
    content_sha256: str,
    processing_version: ProcessingFingerprint,
    force_reindex: bool,
) -> IngestDecision:
    existing = (
        await store.get_document_record(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        )
        if store.capabilities.document_record_lookup
        else None
    )
    ingest_state, should_index = resolve_ingest_state(
        existing,
        content_sha256=content_sha256,
        processing_version=processing_version,
        force_reindex=force_reindex,
    )
    return IngestDecision(
        existing=existing,
        ingest_state=ingest_state,
        should_index=should_index,
    )


__all__ = ["IngestDecision", "resolve_ingest_decision"]
