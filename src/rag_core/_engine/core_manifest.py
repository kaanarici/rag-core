from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core._engine.core_builders import (
    build_preview_document,
)
from rag_core._engine.core_manifest_builders import (
    build_corpus_manifest,
    build_manifest_entry,
)
from rag_core._engine.core_file_io import detect_local_mime_type, read_file_bytes
from rag_core.core_models import CorpusManifest, CorpusManifestEntry, IngestedDocument

if TYPE_CHECKING:
    from rag_core.core_models import PreparedDocument


class PrepareBytes(Protocol):
    async def __call__(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> "PreparedDocument": ...


async def manifest_bytes_for_core(
    *,
    prepare_bytes: PrepareBytes,
    collection_name: str,
    embedding_model: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    namespace: str,
    corpus_id: str,
    document_id: str | None = None,
    document_key: str | None = None,
    path: str | None = None,
    metadata: dict[str, str] | None = None,
) -> CorpusManifestEntry:
    prepared = await prepare_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        path=path,
    )
    preview = build_preview_document(
        file_bytes=file_bytes,
        prepared=prepared,
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        document_key=document_key,
        metadata=metadata,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    return build_manifest_entry(preview)


async def manifest_file_for_core(
    path: str | Path,
    *,
    prepare_bytes: PrepareBytes,
    collection_name: str,
    embedding_model: str,
    namespace: str,
    corpus_id: str,
    document_id: str | None = None,
    document_key: str | None = None,
    metadata: dict[str, str] | None = None,
) -> CorpusManifestEntry:
    file_path = Path(path)
    return await manifest_bytes_for_core(
        prepare_bytes=prepare_bytes,
        collection_name=collection_name,
        embedding_model=embedding_model,
        file_bytes=await read_file_bytes(file_path),
        filename=file_path.name,
        mime_type=detect_local_mime_type(file_path),
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        document_key=document_key,
        path=str(file_path),
        metadata=metadata,
    )


def corpus_manifest_for_core(
    *,
    collection_name: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    namespace: str,
    corpus_id: str,
    documents: list[IngestedDocument],
) -> CorpusManifest:
    return build_corpus_manifest(
        namespace=namespace,
        corpus_id=corpus_id,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        documents=documents,
    )


def manifest_entry_for_core(document: IngestedDocument) -> CorpusManifestEntry:
    return build_manifest_entry(document)
