"""Document ingest: identity, skip, prepare, index, and recover under one fence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core.config import (
    SKIP_UNCHANGED_FAST,
    INGEST_SOURCE_TYPE_FILE,
    INGEST_SOURCE_TYPE_ARCHIVE,
)
from rag_core._engine.core_ingest_fence import DocumentFence
from rag_core._engine.core_ingest_events import (
    emit_ingest_started,
    emit_ingest_skipped,
    emit_ingest_completed,
    emit_index_deleted,
    maybe_wrap_with_ingest_correlation,
)
from rag_core.core_models import (
    DeleteDocumentResult,
    IngestedDocument,
    PreparedDocument,
    ProcessingFingerprint,
    OcrRoutingSignal,
)
from rag_core.events.emit import now_ms, stage_guard
from rag_core.events.types import AuditContext
from rag_core.ingest.identity import resolve_ingest_identity, resolve_ingest_state
from rag_core.ingest.manifest import (
    filename_from_document_key,
    latest_manifest_entry,
    refreshed_manifest_entry,
    write_final_manifest,
)
from rag_core.manifest.persistence import (
    manifest_path,
    write_entry_if_stale,
    delete_entry,
)
from rag_core.search.indexer import DocumentIndexer, validate_delete_scope
from rag_core.search.indexer_models import IndexRequest
from rag_core.search.request_models import StoredDocumentRecord
from rag_core.search.policy import (
    DEFAULT_POLICY,
    VectorStorePolicy,
    CollectionPolicy,
    validate_ingest_scope,
)
from rag_core.search.document_versions import LegacyIndexError
from rag_core.search.provider_protocols import VectorStore

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
        tenant_id: str = "",
        collection: str = "",
        document_id: str = "",
        event_sink: EventSink | None = None,
    ) -> PreparedDocument: ...


class CoreIngestor:
    def __init__(
        self,
        *,
        collection_name: str,
        source_type: str,
        embedding_model: str,
        embedding_dimensions: int | None = None,
        ocr_identity: tuple[str, str | None] | None = None,
        processing_version: ProcessingFingerprint,
        store: VectorStore,
        indexer: DocumentIndexer,
        prepare_bytes: PrepareBytes,
        event_sink: EventSink | None = None,
        manifest_directory: Path | None = None,
        policy: VectorStorePolicy = DEFAULT_POLICY,
        skip_unchanged: str = SKIP_UNCHANGED_FAST,
        fence: DocumentFence | None = None,
        collection_policy: CollectionPolicy | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._source_type = source_type
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._ocr_identity = ocr_identity
        self._processing_version = processing_version
        self._store = store
        self._indexer = indexer
        self._prepare_bytes = prepare_bytes
        self._event_sink = event_sink
        self._manifest_directory = manifest_directory
        self._policy = policy
        self._skip_unchanged = skip_unchanged
        self._fence = fence or DocumentFence()
        self._collection_policy = collection_policy

    async def delete_document(
        self, *, document_id: str, tenant_id: str, collection: str
    ) -> DeleteDocumentResult:
        tenant_id, collection = validate_delete_scope(tenant_id, collection)
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id is required for delete_document")
        document_id = document_id.strip()
        if self._collection_policy is not None:
            self._collection_policy.validate_delete(
                tenant_id=tenant_id, collection=collection
            )
        scope = dict(
            document_id=document_id, tenant_id=tenant_id, collection=collection
        )
        async with self._fence.acquire(**scope):
            with stage_guard(self._event_sink, stage="delete"):
                ack = await self._indexer.delete_document(**scope)
            removed = None
            if self._manifest_directory is not None:
                with stage_guard(self._event_sink, stage="manifest"):
                    removed = delete_entry(self._manifest_directory, **scope)
            emit_index_deleted(self._event_sink, **scope)
            return DeleteDocumentResult(
                **scope,
                vector_store_acked=bool(ack.succeeded),
                manifest_removed=removed,
            )

    async def delete_collection(self, *, tenant_id: str, collection: str) -> None:
        tenant_id, collection = validate_delete_scope(tenant_id, collection)
        if self._collection_policy is not None:
            self._collection_policy.validate_delete(
                tenant_id=tenant_id, collection=collection
            )
        async with self._fence.collection(tenant_id=tenant_id, collection=collection):
            await self._indexer.delete_collection(
                tenant_id=tenant_id, collection=collection
            )
            if self._manifest_directory is not None:
                from rag_core.manifest.persistence import clear_entries

                clear_entries(
                    self._manifest_directory, tenant_id=tenant_id, collection=collection
                )

    async def delete_tenant(self, *, tenant_id: str) -> None:
        raise NotImplementedError(
            "delete_tenant is reserved for tenant offboarding; iterate known collections and call delete_collection for each"
        )

    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument:
        tenant_id, collection = validate_ingest_scope(
            tenant_id, collection, self._collection_policy
        )
        directory = self._manifest_directory
        if directory is not None:
            manifest_path(directory, tenant_id, collection)
        sink = maybe_wrap_with_ingest_correlation(
            self._event_sink, ingest_id=ingest_id, audit_context=audit_context
        )
        started = now_ms()
        identity = resolve_ingest_identity(
            default_source_type=self._source_type,
            source_type=source_type,
            processing_version=self._processing_version,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            document_key=document_key,
            document_id=document_id,
            metadata=metadata,
            tenant_id=tenant_id,
            collection=collection,
            policy=self._policy,
            embedding_model=self._embedding_model,
            embedding_dimensions=self._embedding_dimensions,
            ocr_identity=self._ocr_identity,
        )
        scope = dict(
            tenant_id=tenant_id, collection=collection, document_id=identity.document_id
        )
        emit_ingest_started(
            sink,
            **scope,
            filename=filename,
            mime_type=mime_type,
            content_sha256=identity.content_sha256,
        )
        async with self._fence.acquire(**scope):
            with stage_guard(sink, stage="ingest"):
                existing = await self._store.get_document_record(**scope)
                if existing is None and document_id is None:
                    aliased = await self._store.get_document_record(
                        tenant_id=tenant_id,
                        collection=collection,
                        document_key=identity.document_key,
                    )
                    if (
                        aliased is not None
                        and aliased.document_id != identity.document_id
                    ):
                        raise LegacyIndexError(
                            "Document key uses a different ID scheme; rebuild into a fresh index instead of mixing old and new IDs"
                        )
                state, should_index = resolve_ingest_state(
                    existing,
                    content_sha256=identity.content_sha256,
                    ingest_fingerprint=identity.ingest_fingerprint,
                    processing_version=identity.processing_version,
                    force_reindex=force_reindex,
                )
                prepared = None
                if should_index or self._skip_unchanged != SKIP_UNCHANGED_FAST:
                    prepared = await self._prepare_bytes(
                        file_bytes=file_bytes,
                        filename=filename,
                        mime_type=mime_type,
                        path=path,
                        event_sink=sink,
                        **scope,
                    )
                document = IngestedDocument(
                    **scope,
                    filename=prepared.filename if prepared else filename,
                    mime_type=prepared.mime_type if prepared else mime_type,
                    document_key=identity.document_key,
                    content_sha256=identity.content_sha256,
                    chunk_count=len(prepared.chunks) if prepared else 0,
                    ingest_state=state,
                    replaced_existing=should_index and existing is not None,
                    collection_name=self._collection_name,
                    embedding_model=self._embedding_model,
                    processing_version=identity.processing_version.serialize(),
                    ocr=prepared.ocr if prepared else OcrRoutingSignal(),
                    metadata={
                        **(prepared.metadata if prepared else {}),
                        **(metadata or {}),
                    },
                )
                if not should_index:
                    assert existing is not None
                    emit_ingest_skipped(sink, **scope)
                    document = replace(
                        document,
                        document_key=existing.document_key or identity.document_key,
                        content_sha256=existing.content_sha256
                        or identity.content_sha256,
                        chunk_count=existing.chunk_count,
                        filename=filename_from_document_key(
                            existing.document_key, fallback=document.filename
                        ),
                        metadata=document.metadata
                        if prepared
                        else {**document.metadata, "skip_mode": "fast"},
                    )
                    if directory is not None:
                        with stage_guard(sink, stage="manifest"):
                            write_entry_if_stale(
                                directory,
                                refreshed_manifest_entry(
                                    previous=latest_manifest_entry(directory, **scope),
                                    existing=existing,
                                    **scope,
                                    document_key=identity.document_key,
                                    content_sha256=identity.content_sha256,
                                    filename=filename,
                                    mime_type=mime_type,
                                    metadata=document.metadata
                                    if prepared
                                    else dict(metadata or {}),
                                ),
                            )
                    return document
                assert prepared is not None
                with stage_guard(sink, stage="index"):
                    result = await self._indexer.index_document(
                        build_index_request(
                            prepared=prepared,
                            **scope,
                            document_key=identity.document_key,
                            content_sha256=identity.content_sha256,
                            ingest_fingerprint=identity.ingest_fingerprint,
                            processing_version=identity.processing_version,
                            existing=existing,
                            source_type=identity.source_type,
                            embedding_model=self._embedding_model,
                            metadata=metadata,
                        ),
                        event_sink=sink,
                    )
            document = replace(
                document,
                document_key=result.document_key,
                content_sha256=result.content_sha256,
                chunk_count=result.chunk_count,
            )
            if directory is not None:
                write_final_manifest(
                    event_sink=sink, manifest_directory=directory, ingested=document
                )
            emit_ingest_completed(
                sink,
                **scope,
                chunk_count=result.chunk_count,
                duration_ms=now_ms() - started,
            )
            return document


def build_index_request(
    *,
    prepared: PreparedDocument,
    document_id: str,
    document_key: str | None,
    content_sha256: str | None,
    ingest_fingerprint: str | None = None,
    processing_version: ProcessingFingerprint,
    existing: StoredDocumentRecord | None,
    collection: str,
    tenant_id: str,
    source_type: str,
    metadata: dict[str, str] | None,
    embedding_model: str,
) -> IndexRequest:
    return IndexRequest(
        document_id=document_id,
        document_key=document_key,
        content_sha256=content_sha256,
        ingest_fingerprint=ingest_fingerprint,
        processing_version=processing_version.serialize(),
        collection=collection,
        tenant_id=tenant_id,
        text=prepared.markdown,
        filename=prepared.filename,
        mime_type=prepared.mime_type,
        source_type=source_type,
        path=_display_path_for_index(
            source_type=source_type,
            path=prepared.path,
            document_key=document_key,
        ),
        document_path=_stored_document_path(
            source_type=source_type,
            path=prepared.path,
        ),
        document_metadata=dict(prepared.metadata) or None,
        extra_fields=dict(metadata or {}) or None,
        embedding_model=embedding_model,
        prepared_chunks=prepared.chunks,
    )


def _display_path_for_index(
    *,
    source_type: str,
    path: str | None,
    document_key: str | None,
) -> str | None:
    if (
        source_type in {INGEST_SOURCE_TYPE_FILE, INGEST_SOURCE_TYPE_ARCHIVE}
        and document_key
    ):
        return document_key
    return path


def _stored_document_path(
    *,
    source_type: str,
    path: str | None,
) -> str | None:
    if source_type in {INGEST_SOURCE_TYPE_FILE, INGEST_SOURCE_TYPE_ARCHIVE}:
        return None
    return path
