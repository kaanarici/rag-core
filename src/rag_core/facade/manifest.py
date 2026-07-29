from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core.core_models import (
    CollectionManifest,
    CollectionManifestEntry,
    Config,
    IngestedDocument,
    PreparedDocument,
)


class _ManifestEmbedding(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...


class _EngineManifestMethods:
    _collection_name: str
    _config: Config
    _embedding: _ManifestEmbedding

    if TYPE_CHECKING:
        async def prepare_bytes(
            self,
            *,
            file_bytes: bytes,
                filename: str,
                mime_type: str,
                path: str | None = None,
                namespace: str = "",
                collection: str = "",
                document_id: str = "",
            ) -> PreparedDocument: ...

    def build_manifest_entry(
        self,
        *,
        document: IngestedDocument,
    ) -> CollectionManifestEntry:
        from rag_core._engine.core_manifest import manifest_entry_for_core

        return manifest_entry_for_core(document)

    async def manifest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CollectionManifestEntry:
        from rag_core._engine.core_manifest import manifest_bytes_for_core

        return await manifest_bytes_for_core(
            prepare_bytes=self.prepare_bytes,
            collection_name=self._collection_name,
            embedding_model=self._embedding.model_name,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            path=path,
            metadata=metadata,
        )

    async def manifest_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> CollectionManifestEntry:
        from rag_core._engine.core_manifest import manifest_file_for_core

        return await manifest_file_for_core(
            path,
            prepare_bytes=self.prepare_bytes,
            collection_name=self._collection_name,
            embedding_model=self._embedding.model_name,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
        )

    def build_collection_manifest(
        self,
        *,
        namespace: str,
        collection: str,
        documents: list[IngestedDocument],
    ) -> CollectionManifest:
        from rag_core._engine.core_manifest import collection_manifest_for_core

        return collection_manifest_for_core(
            namespace=namespace,
            collection=collection,
            collection_name=self._collection_name,
            embedding_provider=self._config.embedding.provider,
            embedding_model=self._embedding.model_name,
            embedding_dimensions=self._embedding.dimensions,
            documents=documents,
        )
