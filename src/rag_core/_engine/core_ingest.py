from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core._engine.core_builders import build_index_request
from rag_core._engine.core_ingest_decision import resolve_ingest_decision
from rag_core._engine.core_ingest_delete import (
    delete_corpus_via_indexer,
    delete_ingested_document,
    refuse_namespace_wide_delete,
    resume_partial_delete,
    rollback_index_and_sidecar,
)
from rag_core._engine.core_ingest_fence import DocumentFence
from rag_core._engine.core_ingest_events import (
    emit_ingest_completed,
    emit_ingest_skipped,
    emit_ingest_started,
    maybe_wrap_with_ingest_correlation,
)
from rag_core._engine.core_ingest_identity import ResolvedIngestIdentity, resolve_ingest_identity
from rag_core._engine.core_ingest_recovery import (
    latest_manifest_entry,
    manifest_entry_from_existing_record,
    restore_manifest_from_vector_store,
    refreshed_manifest_entry,
    sync_sidecar_or_emit_error,
    write_final_manifest,
)
from rag_core._engine.core_ingest_write_ahead import (
    best_effort_rollback_delete,
    resume_pending_write_ahead,
    write_ahead_journal_for,
)
from rag_core._engine.core_ingest_results import (
    build_fast_skipped_ingested_document,
    build_indexed_ingested_document,
    build_skipped_ingested_document,
)
from rag_core._engine.core_manifest_builders import build_staged_manifest_entry
from rag_core.core_models import (
    DeleteDocumentResult,
    IngestedDocument,
    PreparedDocument,
    ProcessingFingerprint,
)
from rag_core.events.emit import now_ms, stage_guard
from rag_core.events.types import AuditContext
from rag_core.manifest_persistence import manifest_path, write_entry, write_entry_if_stale
from rag_core.config import SKIP_UNCHANGED_FAST
from rag_core.search.indexer import DocumentIndexer
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.provider_protocols import SearchSidecar, VectorStore

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache
    from rag_core.search.providers.embedding_cache_models import EmbeddingCache

class PrepareBytes(Protocol):
    async def __call__(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
        namespace: str = "",
        corpus_id: str = "",
        document_id: str = "",
    ) -> PreparedDocument: ...


class CoreIngestor:
    def __init__(
        self,
        *,
        collection_name: str,
        source_type: str,
        embedding_model: str,
        processing_version: ProcessingFingerprint,
        store: VectorStore,
        indexer: DocumentIndexer,
        sidecar: SearchSidecar | None,
        prepare_bytes: PrepareBytes,
        event_sink: "EventSink | None" = None,
        manifest_directory: Path | None = None,
        policy: VectorStorePolicy = DEFAULT_POLICY,
        skip_unchanged: str = SKIP_UNCHANGED_FAST,
        embedding_cache: "EmbeddingCache | None" = None,
        chunk_context_cache: "ChunkContextCache | None" = None,
        fence: DocumentFence | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._source_type = source_type
        self._embedding_model = embedding_model
        self._processing_version = processing_version
        self._store = store
        self._indexer = indexer
        self._sidecar = sidecar
        self._prepare_bytes = prepare_bytes
        self._event_sink = event_sink
        self._manifest_directory = manifest_directory
        self._policy = policy
        self._skip_unchanged = skip_unchanged
        # Right-to-forget seam: optional caches purged scoped per doc on delete.
        self._embedding_cache = embedding_cache
        self._chunk_context_cache = chunk_context_cache
        # Per-doc fence shared with the delete mixin so concurrent ingest +
        # delete on the same triple cannot interleave.
        self._fence = fence or DocumentFence()

    async def delete_document(
        self, *, document_id: str, namespace: str, corpus_id: str,
    ) -> DeleteDocumentResult:
        # Acquire the same per-doc lock the ingestor takes so a concurrent
        # re-ingest cannot interleave between the vector-store ack and the
        # later stages of the right-to-forget walk.
        async with self._fence.acquire(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        ):
            return await delete_ingested_document(
                indexer=self._indexer,
                sidecar=self._sidecar,
                event_sink=self._event_sink,
                manifest_directory=self._manifest_directory,
                document_id=document_id,
                namespace=namespace,
                corpus_id=corpus_id,
                embedding_cache=self._embedding_cache,
                chunk_context_cache=self._chunk_context_cache,
            )

    async def delete_corpus(self, *, namespace: str, corpus_id: str) -> None:
        await delete_corpus_via_indexer(
            indexer=self._indexer, namespace=namespace, corpus_id=corpus_id,
        )

    async def delete_namespace(self, *, namespace: str) -> None:
        refuse_namespace_wide_delete(namespace)

    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument:
        if self._manifest_directory is not None:
            manifest_path(self._manifest_directory, namespace, corpus_id)
        sink = maybe_wrap_with_ingest_correlation(self._event_sink, ingest_id=ingest_id, audit_context=audit_context)
        started_ms = now_ms()
        identity = resolve_ingest_identity(
            default_source_type=self._source_type,
            source_type=source_type,
            processing_version=self._processing_version,
            file_bytes=file_bytes,
            filename=filename,
            path=path,
            document_key=document_key,
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
            policy=self._policy,
        )
        emit_ingest_started(
            sink,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=identity.document_id,
            filename=filename,
            mime_type=mime_type,
            content_sha256=identity.content_sha256,
        )
        # Per-doc fence: serialize any other ingest or delete on the same
        # (namespace, corpus_id, document_id) so the resume-stale-delete /
        # decision / index / manifest sequence runs uninterleaved. Different
        # docs run concurrently. The fence is keyed by the triple.
        async with self._fence.acquire(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=identity.document_id,
        ):
            return await self._ingest_inside_fence(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=mime_type,
                namespace=namespace,
                corpus_id=corpus_id,
                metadata=metadata,
                force_reindex=force_reindex,
                path=path,
                identity=identity,
                sink=sink,
                started_ms=started_ms,
            )

    async def _ingest_inside_fence(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        metadata: dict[str, str] | None,
        force_reindex: bool,
        path: str | None,
        identity: ResolvedIngestIdentity,
        sink: "EventSink | None",
        started_ms: float,
    ) -> IngestedDocument:
        # Hoisted out of ``ingest_bytes`` so the fence wrapper above stays
        # readable. The body below is the original ingest flow. Unchanged
        # apart from now running under the per-doc lock.
        # Replay a pending right-to-forget purge for this triple before writing new content.
        await resume_partial_delete(
            indexer=self._indexer, sidecar=self._sidecar, event_sink=sink,
            manifest_directory=self._manifest_directory, document_id=identity.document_id,
            namespace=namespace, corpus_id=corpus_id,
            embedding_cache=self._embedding_cache, chunk_context_cache=self._chunk_context_cache,
        )
        # Replay a torn ingest of this triple: if a prior run crashed
        # between ``indexer.upsert`` and ``manifest.write``, purge the
        # orphan chunks so the fresh ingest does not pile on top of a
        # half-committed state.
        write_ahead = write_ahead_journal_for(self._manifest_directory)
        await resume_pending_write_ahead(
            indexer=self._indexer,
            event_sink=sink,
            manifest_directory=self._manifest_directory,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=identity.document_id,
            journal=write_ahead,
        )
        with stage_guard(sink, stage="ingest"):
            decision = await resolve_ingest_decision(
                store=self._store,
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=identity.document_id,
                content_sha256=identity.content_sha256,
                processing_version=identity.processing_version,
                force_reindex=force_reindex,
            )

            if not decision.should_index:
                emit_ingest_skipped(
                    sink,
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=identity.document_id,
                )
                if self._skip_unchanged == SKIP_UNCHANGED_FAST:
                    skipped = build_fast_skipped_ingested_document(
                        identity=identity,
                        decision=decision,
                        filename=filename,
                        mime_type=mime_type,
                        corpus_id=corpus_id,
                        namespace=namespace,
                        collection_name=self._collection_name,
                        embedding_model=self._embedding_model,
                        metadata=metadata,
                    )
                else:
                    prepared = await self._prepare_bytes(
                        file_bytes=file_bytes,
                        filename=filename,
                        mime_type=mime_type,
                        path=path,
                        namespace=namespace,
                        corpus_id=corpus_id,
                        document_id=identity.document_id,
                    )
                    skipped = build_skipped_ingested_document(
                        prepared=prepared,
                        identity=identity,
                        decision=decision,
                        corpus_id=corpus_id,
                        namespace=namespace,
                        collection_name=self._collection_name,
                        embedding_model=self._embedding_model,
                        metadata=metadata,
                    )
                if self._manifest_directory is not None:
                    with stage_guard(sink, stage="manifest"):
                        previous_entry = latest_manifest_entry(
                            self._manifest_directory,
                            namespace=namespace,
                            corpus_id=corpus_id,
                            document_id=identity.document_id,
                        )
                        write_entry_if_stale(
                            self._manifest_directory,
                            refreshed_manifest_entry(
                                previous=previous_entry,
                                existing=decision.existing,
                                document_id=identity.document_id,
                                namespace=namespace,
                                corpus_id=corpus_id,
                                document_key=identity.document_key,
                                content_sha256=identity.content_sha256,
                                filename=filename,
                                mime_type=mime_type,
                                metadata=(
                                    dict(metadata or {})
                                    if self._skip_unchanged == SKIP_UNCHANGED_FAST
                                    else skipped.metadata
                                ),
                            ),
                        )
                return skipped
            prepared = await self._prepare_bytes(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=mime_type,
                path=path,
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=identity.document_id,
            )
            rollback_manifest_entry = None
            if (
                self._manifest_directory is not None
                and decision.existing is not None
            ):
                rollback_manifest_entry = latest_manifest_entry(
                    self._manifest_directory,
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=identity.document_id,
                )
                if rollback_manifest_entry is None:
                    rollback_manifest_entry = manifest_entry_from_existing_record(
                        decision.existing,
                        filename=filename,
                        mime_type=mime_type,
                        document_key=identity.document_key,
                    )
            if (
                self._manifest_directory is not None
                and decision.existing is not None
            ):
                with stage_guard(sink, stage="manifest"):
                    write_entry(
                        self._manifest_directory,
                        build_staged_manifest_entry(
                            prepared=prepared,
                            document_id=identity.document_id,
                            namespace=namespace,
                            corpus_id=corpus_id,
                            document_key=identity.document_key,
                            content_sha256=identity.content_sha256,
                            filename=filename,
                            mime_type=mime_type,
                            metadata=metadata,
                        ),
                    )

            # Write-ahead: record the intent to upsert BEFORE touching the
            # store. A crash between here and ``write_final_manifest`` leaves
            # an ``upserted_pending_manifest`` row on disk that the next
            # ingest of the same triple replays via
            # ``resume_pending_write_ahead`` above.
            if write_ahead is not None:
                write_ahead.record_pending(
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=identity.document_id,
                    content_sha256=identity.content_sha256,
                    expected_chunk_count=len(prepared.chunks),
                )
            try:
                with stage_guard(sink, stage="index"):
                    result = await self._indexer.index_document(
                        build_index_request(
                            prepared=prepared,
                            document_id=identity.document_id,
                            document_key=identity.document_key,
                            content_sha256=identity.content_sha256,
                            processing_version=identity.processing_version,
                            existing=decision.existing,
                            corpus_id=corpus_id,
                            namespace=namespace,
                            source_type=identity.source_type,
                            embedding_model=self._embedding_model,
                            metadata=metadata,
                        ),
                        event_sink=sink,
                    )
            except Exception as index_exc:
                manifest_directory = self._manifest_directory
                # Torn-write rollback BEFORE manifest restore so a partial
                # Qdrant batch upsert cannot leave residue under the new
                # content_sha256. Deliberate tradeoff: point ids are
                # deterministic per chunk index, so a partial upsert mixes old
                # and new chunk content irrecoverably. Consistency wins over
                # availability and the document becomes unsearchable until the
                # caller retries the ingest, rather than serving a mixed set.
                await best_effort_rollback_delete(
                    indexer=self._indexer,
                    event_sink=sink,
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=identity.document_id,
                )
                if manifest_directory is not None and decision.existing is not None:
                    try:
                        await restore_manifest_from_vector_store(
                            store=self._store,
                            event_sink=sink,
                            manifest_directory=manifest_directory,
                            namespace=namespace,
                            corpus_id=corpus_id,
                            document_id=identity.document_id,
                            filename=filename,
                            mime_type=mime_type,
                            document_key=identity.document_key,
                            fallback_entry=rollback_manifest_entry,
                        )
                    except Exception as repair_exc:
                        raise ExceptionGroup(
                            "index failed and manifest restore failed",
                            [index_exc, repair_exc],
                        ) from None
                raise
        ingested = build_indexed_ingested_document(
            prepared=prepared,
            identity=identity,
            decision=decision,
            result=result,
            corpus_id=corpus_id,
            namespace=namespace,
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            metadata=metadata,
        )
        try:
            sync_sidecar_or_emit_error(
                sidecar=self._sidecar,
                event_sink=self._event_sink,
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=identity.document_id,
                policy=self._policy,
                result=result,
            )
        except Exception as sidecar_exc:
            if decision.existing is not None and write_ahead is not None:
                if self._manifest_directory is not None:
                    try:
                        write_final_manifest(
                            event_sink=sink,
                            manifest_directory=self._manifest_directory,
                            ingested=ingested,
                        )
                    except Exception as repair_exc:
                        raise ExceptionGroup(
                            "sidecar sync failed and manifest repair failed",
                            [sidecar_exc, repair_exc],
                        ) from None
                raise
            # Without a write-ahead marker, a successful vector upsert plus failed
            # sidecar sync would make a same-content retry skip indexing forever.
            # Roll back the canonical index so retry lands through the normal path.
            try:
                await rollback_index_and_sidecar(
                    indexer=self._indexer,
                    sidecar=self._sidecar,
                    namespace=namespace,
                    corpus_id=corpus_id,
                    document_id=identity.document_id,
                )
            except Exception as rollback_exc:
                raise ExceptionGroup(
                    "sidecar sync failed and index rollback failed",
                    [sidecar_exc, rollback_exc],
                ) from None
            raise
        if self._manifest_directory is not None:
            try:
                write_final_manifest(
                    event_sink=sink,
                    manifest_directory=self._manifest_directory,
                    ingested=ingested,
                )
            except Exception as manifest_exc:
                if decision.existing is not None:
                    try:
                        write_final_manifest(
                            event_sink=sink,
                            manifest_directory=self._manifest_directory,
                            ingested=ingested,
                        )
                    except Exception as repair_exc:
                        raise ExceptionGroup(
                            "manifest write failed and manifest retry failed",
                            [manifest_exc, repair_exc],
                        ) from None
                else:
                    try:
                        await rollback_index_and_sidecar(
                            indexer=self._indexer,
                            sidecar=self._sidecar,
                            namespace=namespace,
                            corpus_id=corpus_id,
                            document_id=identity.document_id,
                        )
                    except Exception as rollback_exc:
                        raise ExceptionGroup(
                            "manifest write failed and index rollback failed",
                            [manifest_exc, rollback_exc],
                        ) from None
                    raise
        # Write-ahead commit: only reached when the upsert and manifest
        # write both landed. The committed marker turns the prior pending
        # entry into a no-op for the next ``resume_pending_write_ahead``.
        if write_ahead is not None:
            write_ahead.record_committed(
                namespace=namespace,
                corpus_id=corpus_id,
                document_id=identity.document_id,
                content_sha256=identity.content_sha256,
                expected_chunk_count=result.chunk_count,
            )
        emit_ingest_completed(
            sink,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=identity.document_id,
            chunk_count=result.chunk_count,
            duration_ms=now_ms() - started_ms,
        )
        return ingested
