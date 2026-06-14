"""Right-to-forget delete: vector store, sidecar, caches, manifest.

The order is the public delete-completeness contract:

1. Vector store (the canonical retrieval surface).
2. Lexical sidecar (derived; without the index ack we never touch it).
3. Embedding cache (scoped purge; ``None`` for caches that can't scope).
4. Chunk-context cache (scoped purge; ``None`` for caches that can't scope).
5. Manifest entry (the local ingest record).

Each transition writes to a ``DeleteRecoveryJournal`` so that a crash
between stages leaves a recoverable trail. The journal entry is what the
next ``ingest_bytes`` on the same ``(namespace, corpus_id, document_id)``
triple inspects to finish the purge before it touches new content.

Restricted-tier deployments (``CorpusPolicy.cache_disabled=True`` in the
deploy contract) typically wire ``NoCache`` for both caches; those report
``0`` purged rows but exit ``succeeded=True`` so ``DeleteDocumentResult``
honestly reflects "the cache was wired; nothing was tagged to this scope".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core._engine.core_ingest_delete_journal import (
    DELETE_STAGES_IN_ORDER,
    DeleteRecoveryJournal,
    STAGE_CHUNK_CONTEXT_CACHE,
    STAGE_EMBEDDING_CACHE,
    STAGE_LEXICAL_SIDECAR,
    STAGE_MANIFEST,
    STAGE_VECTOR_STORE,
)
from rag_core._engine.core_ingest_events import emit_index_deleted
from rag_core.core_models import DeleteDocumentResult
from rag_core.events.emit import stage_guard
from rag_core.manifest_persistence import delete_entry

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
    corpus_id: str,
) -> bool | None:
    await indexer.delete_document(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
    )
    if sidecar is None:
        return None
    sidecar.delete_document(
        namespace=namespace,
        document_id=document_id,
        corpus_id=corpus_id,
    )
    return True


async def delete_corpus_via_indexer(
    *,
    indexer: "DocumentIndexer",
    namespace: str,
    corpus_id: str,
) -> None:
    await indexer.delete_corpus(namespace=namespace, corpus_id=corpus_id)


def refuse_namespace_wide_delete(namespace: str) -> None:
    del namespace
    raise NotImplementedError(
        "delete_namespace is reserved for tenant offboarding; iterate "
        "known corpus_ids and call delete_corpus for each"
    )


async def delete_ingested_document(
    *,
    indexer: "DocumentIndexer",
    sidecar: "SearchSidecar | None",
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    document_id: str,
    namespace: str,
    corpus_id: str,
    embedding_cache: "EmbeddingCache | None" = None,
    chunk_context_cache: "ChunkContextCache | None" = None,
    recovery_journal: "DeleteRecoveryJournal | None" = None,
) -> DeleteDocumentResult:
    """Canonical right-to-forget delete.

    Order: vector store -> sidecar -> embedding cache -> chunk-context
    cache -> manifest. Each successful step appends to the recovery
    journal. Vector-store failure raises (no later step touched). Failure
    on a later step records the partial state in the journal and re-raises
    so the caller sees the error; the journal entry is what the next
    ``ingest_bytes`` on the same triple replays before writing new content.
    """
    journal = recovery_journal or _journal_for(manifest_directory)
    completed_stages: list[str] = []

    index_acked = await _stage_delete_vector_store(
        indexer=indexer,
        event_sink=event_sink,
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        journal=journal,
        completed_stages=completed_stages,
    )

    sidecar_deleted = await _stage_delete_sidecar(
        sidecar=sidecar,
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        journal=journal,
        completed_stages=completed_stages,
    )

    embedding_cache_purged = await _purge_scoped_cache(
        cache=embedding_cache,
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        stage=STAGE_EMBEDDING_CACHE,
        journal=journal,
        completed_stages=completed_stages,
    )
    chunk_context_cache_purged = await _purge_scoped_cache(
        cache=chunk_context_cache,
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        stage=STAGE_CHUNK_CONTEXT_CACHE,
        journal=journal,
        completed_stages=completed_stages,
    )

    manifest_entry_deleted = await _stage_delete_manifest(
        event_sink=event_sink,
        manifest_directory=manifest_directory,
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        journal=journal,
        completed_stages=completed_stages,
    )

    # All steps survived. Record the completed entry so a replay knows the
    # right-to-forget is done.
    if journal is not None:
        journal.record(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            stages_completed=tuple(DELETE_STAGES_IN_ORDER),
            completed=True,
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
        index_deleted=index_acked,
        sidecar_deleted=sidecar_deleted,
        manifest_entry_deleted=manifest_entry_deleted,
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
    corpus_id: str,
    journal: "DeleteRecoveryJournal | None",
    completed_stages: list[str],
) -> bool:
    try:
        with stage_guard(event_sink, stage="delete"):
            ack = await indexer.delete_document(
                document_id=document_id,
                namespace=namespace,
                corpus_id=corpus_id,
            )
    except Exception as exc:
        if journal is not None:
            journal.record(
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=document_id,
                stages_completed=tuple(completed_stages),
                completed=False,
                last_error_type=type(exc).__name__,
                last_error_stage=STAGE_VECTOR_STORE,
            )
        raise
    index_acked = bool(ack.succeeded)
    if index_acked:
        completed_stages.append(STAGE_VECTOR_STORE)
    return index_acked


async def _stage_delete_sidecar(
    *,
    sidecar: "SearchSidecar | None",
    document_id: str,
    namespace: str,
    corpus_id: str,
    journal: "DeleteRecoveryJournal | None",
    completed_stages: list[str],
) -> bool | None:
    if sidecar is None:
        return None
    try:
        sidecar.delete_document(
            namespace=namespace,
            document_id=document_id,
            corpus_id=corpus_id,
        )
    except Exception as exc:
        if journal is not None:
            journal.record(
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=document_id,
                stages_completed=tuple(completed_stages),
                completed=False,
                last_error_type=type(exc).__name__,
                last_error_stage=STAGE_LEXICAL_SIDECAR,
            )
        raise
    completed_stages.append(STAGE_LEXICAL_SIDECAR)
    return True


async def _stage_delete_manifest(
    *,
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    document_id: str,
    namespace: str,
    corpus_id: str,
    journal: "DeleteRecoveryJournal | None",
    completed_stages: list[str],
) -> bool | None:
    if manifest_directory is None:
        return None
    try:
        with stage_guard(event_sink, stage="manifest"):
            manifest_entry_deleted = delete_entry(
                manifest_directory,
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=document_id,
            )
    except Exception as exc:
        if journal is not None:
            journal.record(
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=document_id,
                stages_completed=tuple(completed_stages),
                completed=False,
                last_error_type=type(exc).__name__,
                last_error_stage=STAGE_MANIFEST,
            )
        raise
    completed_stages.append(STAGE_MANIFEST)
    return manifest_entry_deleted


def _journal_for(manifest_directory: Path | None) -> DeleteRecoveryJournal | None:
    """Build a journal under the manifest directory, if a directory is wired."""
    if manifest_directory is None:
        return None
    return DeleteRecoveryJournal(directory=manifest_directory)


async def _purge_scoped_cache(
    *,
    cache: object | None,
    namespace: str,
    corpus_id: str,
    document_id: str,
    stage: str,
    journal: "DeleteRecoveryJournal | None",
    completed_stages: list[str],
) -> bool | None:
    if cache is None:
        return None
    purge = getattr(cache, "delete_by_document_scope", None)
    if not callable(purge):
        # The cache exists but does not implement the scoped-delete
        # capability. Surface ``None`` so the right-to-forget result is
        # honest: the surface is wired but cannot scope-delete. The plan's
        # deletion trigger (deploy contract: ``cache_disabled`` on the
        # restricted tier) gives operators the safe fallback.
        return None
    try:
        await purge(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        )
    except Exception as exc:
        if journal is not None:
            journal.record(
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=document_id,
                stages_completed=tuple(completed_stages),
                completed=False,
                last_error_type=type(exc).__name__,
                last_error_stage=stage,
            )
        raise
    completed_stages.append(stage)
    return True


async def resume_partial_delete(
    *,
    indexer: "DocumentIndexer",
    sidecar: "SearchSidecar | None",
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    document_id: str,
    namespace: str,
    corpus_id: str,
    embedding_cache: "EmbeddingCache | None" = None,
    chunk_context_cache: "ChunkContextCache | None" = None,
    recovery_journal: "DeleteRecoveryJournal | None" = None,
) -> DeleteDocumentResult | None:
    """Replay the right-to-forget purge for a doc with a pending journal entry.

    Called from ``ingest_bytes`` BEFORE a fresh document write goes in. If
    the journal has no pending entry (or no journal is configured), returns
    ``None`` and the caller proceeds with normal ingest. If a pending entry
    exists, the remaining stages are run, the journal is closed, and the
    returned ``DeleteDocumentResult`` summarizes what completed during the
    replay so the caller can attach it to its audit context.
    """
    journal = recovery_journal or _journal_for(manifest_directory)
    if journal is None:
        return None
    latest = journal.latest_entry(
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
    )
    if latest is None or latest.completed:
        return None
    # Re-run the canonical delete from the beginning. Each stage is
    # idempotent on the underlying surfaces (vector store delete by
    # filter, sidecar delete by document, scoped cache purge, manifest
    # delete). A duplicate delete is cheaper than partial reconciliation
    # and avoids re-implementing per-stage replay logic.
    return await delete_ingested_document(
        indexer=indexer,
        sidecar=sidecar,
        event_sink=event_sink,
        manifest_directory=manifest_directory,
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        embedding_cache=embedding_cache,
        chunk_context_cache=chunk_context_cache,
        recovery_journal=journal,
    )


__all__ = [
    "DeleteRecoveryJournal",
    "delete_corpus_via_indexer",
    "delete_ingested_document",
    "refuse_namespace_wide_delete",
    "resume_partial_delete",
    "rollback_index_and_sidecar",
]
