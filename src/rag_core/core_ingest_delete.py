"""Public delete path for ingested documents (vector index, sidecar, manifest, events)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.core_ingest_events import emit_index_deleted
from rag_core.core_models import DeleteDocumentResult
from rag_core.events.emit import stage_guard
from rag_core.manifest_persistence import delete_entry

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.search.indexer import DocumentIndexer
    from rag_core.search.provider_protocols import SearchSidecar


class VectorIndexDeleteOrder(StrEnum):
    """Order for removing a document from the vector index and lexical sidecar."""

    INDEX_THEN_SIDECAR = "index_then_sidecar"
    SIDECAR_THEN_INDEX = "sidecar_then_index"


async def delete_vector_index_and_sidecar(
    *,
    indexer: "DocumentIndexer",
    sidecar: "SearchSidecar | None",
    document_id: str,
    namespace: str,
    corpus_id: str,
    order: VectorIndexDeleteOrder = VectorIndexDeleteOrder.INDEX_THEN_SIDECAR,
) -> bool | None:
    """Remove a document from the vector index and optional lexical sidecar."""
    sidecar_deleted: bool | None = None
    if order is VectorIndexDeleteOrder.SIDECAR_THEN_INDEX and sidecar is not None:
        sidecar.delete_document(
            namespace=namespace,
            document_id=document_id,
            corpus_id=corpus_id,
        )
        sidecar_deleted = True
    await indexer.delete_document(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
    )
    if order is VectorIndexDeleteOrder.INDEX_THEN_SIDECAR and sidecar is not None:
        sidecar.delete_document(
            namespace=namespace,
            document_id=document_id,
            corpus_id=corpus_id,
        )
        sidecar_deleted = True
    return sidecar_deleted


async def delete_ingested_document(
    *,
    indexer: "DocumentIndexer",
    sidecar: "SearchSidecar | None",
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    document_id: str,
    namespace: str,
    corpus_id: str,
) -> DeleteDocumentResult:
    with stage_guard(event_sink, stage="delete"):
        sidecar_deleted = await delete_vector_index_and_sidecar(
            indexer=indexer,
            sidecar=sidecar,
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
            order=VectorIndexDeleteOrder.SIDECAR_THEN_INDEX,
        )

    manifest_entry_deleted: bool | None = None
    if manifest_directory is not None:
        with stage_guard(event_sink, stage="manifest"):
            manifest_entry_deleted = delete_entry(
                manifest_directory,
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=document_id,
            )

    emit_index_deleted(
        event_sink,
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
    )
    return DeleteDocumentResult(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        index_deleted=True,
        sidecar_deleted=sidecar_deleted,
        manifest_entry_deleted=manifest_entry_deleted,
    )


__all__ = [
    "VectorIndexDeleteOrder",
    "delete_ingested_document",
    "delete_vector_index_and_sidecar",
]
