from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core._engine.core_builders import build_index_request
from rag_core._engine.core_ingest_decision import resolve_ingest_decision
from rag_core._engine.core_ingest_delete import (
    delete_ingested_document,
    delete_vector_index_and_sidecar,
)
from rag_core._engine.core_ingest_events import (
    emit_ingest_completed,
    emit_ingest_skipped,
    emit_ingest_started,
)
from rag_core._engine.core_ingest_identity import resolve_ingest_identity
from rag_core._engine.core_ingest_recovery import (
    latest_manifest_entry,
    manifest_entry_from_existing_record,
    restore_manifest_from_vector_store,
    refreshed_manifest_entry,
    retry_final_manifest_write,
    sync_sidecar_or_emit_error,
    write_final_manifest,
)
from rag_core._engine.core_ingest_results import (
    build_fast_skipped_ingested_document,
    build_indexed_ingested_document,
    build_skipped_ingested_document,
)
from rag_core._engine.core_manifest_builders import build_staged_manifest_entry
from rag_core.core_models import DeleteDocumentResult, IngestedDocument, PreparedDocument
from rag_core.core_models import ProcessingFingerprint
from rag_core.events.emit import now_ms, stage_guard
from rag_core.manifest_persistence import (
    manifest_path,
    write_entry,
    write_entry_if_stale,
)
from rag_core.config import SKIP_UNCHANGED_FAST
from rag_core.search.indexer import DocumentIndexer
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.provider_protocols import SearchSidecar, VectorStore

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

class PrepareBytes(Protocol):
    async def __call__(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
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
    ) -> IngestedDocument:
        if self._manifest_directory is not None:
            manifest_path(self._manifest_directory, namespace, corpus_id)
        sink = self._event_sink
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
                        )
                    )
            except Exception as index_exc:
                manifest_directory = self._manifest_directory
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
            if decision.existing is not None:
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
            try:
                await delete_vector_index_and_sidecar(
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
                        retry_final_manifest_write(
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
                        await delete_vector_index_and_sidecar(
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
        emit_ingest_completed(
            sink,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=identity.document_id,
            chunk_count=result.chunk_count,
            duration_ms=now_ms() - started_ms,
        )
        return ingested

    async def delete_document(
        self,
        *,
        document_id: str,
        namespace: str,
        corpus_id: str,
    ) -> DeleteDocumentResult:
        return await delete_ingested_document(
            indexer=self._indexer,
            sidecar=self._sidecar,
            event_sink=self._event_sink,
            manifest_directory=self._manifest_directory,
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
        )
