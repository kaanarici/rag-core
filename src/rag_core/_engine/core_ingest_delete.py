"""Right-to-forget delete: vector store, sidecar, caches, manifest.

The order is the public delete-completeness contract:

1. Vector store (the canonical retrieval surface).
2. Lexical sidecar (derived; without the index ack we never touch it).
3. Embedding cache (scoped purge; ``None`` for caches that can't scope).
4. Chunk-context cache (scoped purge; ``None`` for caches that can't scope).
5. Manifest entry (the local ingest record).

Each step is idempotent. A crash mid-delete raises; the caller retries
``delete_document``. Ingest of the same document replaces in place via
deterministic point IDs and stale-chunk pruning.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core._engine.core_ingest_events import emit_index_deleted
from rag_core.core_models import DeleteDocumentResult
from rag_core.events.emit import stage_guard
from rag_core.manifest.persistence import delete_entry

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.search.indexer import DocumentIndexer
    from rag_core.search.provider_protocols import SearchSidecar
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache
    from rag_core.search.providers.embedding_cache_models import EmbeddingCache


async def rollback_index_and_sidecar(
    *,
    indexer: "DocumentIndexer",
    sidecar: "SearchSidecar | None",
    document_id: str,
    namespace: str,
    collection: str,
) -> bool | None:
    await indexer.delete_document(
        document_id=document_id,
        namespace=namespace,
        collection=collection,
    )
    if sidecar is None:
        return None
    sidecar.delete_document(
        namespace=namespace,
        document_id=document_id,
        collection=collection,
    )
    return True


async def best_effort_rollback_delete(
    *,
    indexer: "DocumentIndexer",
    event_sink: "EventSink | None",
    namespace: str,
    collection: str,
    document_id: str,
) -> None:
    """Purge torn upsert residue before restoring the manifest.

    A Qdrant batch upsert can land some points before raising. The call is
    best-effort: failures never mask the original index error.
    """
    try:
        with stage_guard(event_sink, stage="index"):
            await indexer.delete_document(
                document_id=document_id,
                namespace=namespace,
                collection=collection,
            )
    except Exception:
        pass


async def delete_collection_via_indexer(
    *,
    indexer: "DocumentIndexer",
    namespace: str,
    collection: str,
) -> None:
    await indexer.delete_collection(namespace=namespace, collection=collection)


def refuse_namespace_wide_delete(namespace: str) -> None:
    del namespace
    raise NotImplementedError(
        "delete_namespace is reserved for tenant offboarding; iterate "
        "known collections and call delete_collection for each"
    )


async def delete_ingested_document(
    *,
    indexer: "DocumentIndexer",
    sidecar: "SearchSidecar | None",
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    document_id: str,
    namespace: str,
    collection: str,
    embedding_cache: "EmbeddingCache | None" = None,
    chunk_context_cache: "ChunkContextCache | None" = None,
) -> DeleteDocumentResult:
    """Canonical right-to-forget delete.

    Order: vector store -> sidecar -> embedding cache -> chunk-context
    cache -> manifest. Vector-store failure raises (no later step touched).
    Failure on a later step re-raises so the caller retries the whole delete.
    """
    index_acked = await _stage_delete_vector_store(
        indexer=indexer,
        event_sink=event_sink,
        document_id=document_id,
        namespace=namespace,
        collection=collection,
    )

    sidecar_deleted = await _stage_delete_sidecar(
        sidecar=sidecar,
        document_id=document_id,
        namespace=namespace,
        collection=collection,
    )

    embedding_cache_purged = await _purge_scoped_cache(
        cache=embedding_cache,
        namespace=namespace,
        collection=collection,
        document_id=document_id,
    )
    chunk_context_cache_purged = await _purge_scoped_cache(
        cache=chunk_context_cache,
        namespace=namespace,
        collection=collection,
        document_id=document_id,
    )

    manifest_entry_deleted = await _stage_delete_manifest(
        event_sink=event_sink,
        manifest_directory=manifest_directory,
        document_id=document_id,
        namespace=namespace,
        collection=collection,
    )

    emit_index_deleted(
        event_sink,
        namespace=namespace,
        collection=collection,
        document_id=document_id,
    )
    return DeleteDocumentResult(
        document_id=document_id,
        namespace=namespace,
        collection=collection,
        vector_store_acked=index_acked,
        lexical_sidecar_purged=sidecar_deleted,
        embedding_cache_purged=embedding_cache_purged,
        chunk_context_cache_purged=chunk_context_cache_purged,
        manifest_removed=manifest_entry_deleted,
    )


async def _stage_delete_vector_store(
    *,
    indexer: "DocumentIndexer",
    event_sink: "EventSink | None",
    document_id: str,
    namespace: str,
    collection: str,
) -> bool:
    with stage_guard(event_sink, stage="delete"):
        ack = await indexer.delete_document(
            document_id=document_id,
            namespace=namespace,
            collection=collection,
        )
    return bool(ack.succeeded)


async def _stage_delete_sidecar(
    *,
    sidecar: "SearchSidecar | None",
    document_id: str,
    namespace: str,
    collection: str,
) -> bool | None:
    if sidecar is None:
        return None
    sidecar.delete_document(
        namespace=namespace,
        document_id=document_id,
        collection=collection,
    )
    return True


async def _stage_delete_manifest(
    *,
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    document_id: str,
    namespace: str,
    collection: str,
) -> bool | None:
    if manifest_directory is None:
        return None
    with stage_guard(event_sink, stage="manifest"):
        return delete_entry(
            manifest_directory,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
        )


async def _purge_scoped_cache(
    *,
    cache: object | None,
    namespace: str,
    collection: str,
    document_id: str,
) -> bool | None:
    if cache is None:
        return None
    purge = getattr(cache, "delete_by_document_scope", None)
    if not callable(purge):
        return None
    await purge(
        namespace=namespace,
        collection=collection,
        document_id=document_id,
    )
    return True


__all__ = [
    "best_effort_rollback_delete",
    "delete_collection_via_indexer",
    "delete_ingested_document",
    "refuse_namespace_wide_delete",
    "rollback_index_and_sidecar",
]
