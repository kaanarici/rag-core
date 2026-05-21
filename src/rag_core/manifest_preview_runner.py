from __future__ import annotations

from rag_core.core_builders import build_preview_document
from rag_core.core_manifest_builders import build_manifest_entry
from rag_core.core_file_io import detect_local_mime_type, read_file_bytes
from rag_core.core_prepare import prepare_document_bytes
from rag_core.local_sources import document_key as local_document_key
from rag_core.local_sources import reject_local_symlink_path
from rag_core.manifest_preview_models import (
    ManifestPreviewRequest,
    ManifestPreviewResult,
)


async def preview_manifest(request: ManifestPreviewRequest) -> ManifestPreviewResult:
    file_path = request.path
    reject_local_symlink_path(file_path)
    document_key = (
        request.document_key.strip()
        if request.document_key and request.document_key.strip()
        else local_document_key(file_path.parent, file_path)
    )
    file_bytes = await read_file_bytes(file_path)
    prepared = await prepare_document_bytes(
        file_bytes=file_bytes,
        filename=file_path.name,
        mime_type=detect_local_mime_type(file_path),
        path=str(file_path),
        ocr_provider=None,
        allow_needs_ocr=True,
    )
    preview = build_preview_document(
        file_bytes=file_bytes,
        prepared=prepared,
        namespace=request.namespace,
        corpus_id=request.corpus_id,
        document_id=request.document_id,
        document_key=document_key,
        metadata=request.metadata,
    )
    return ManifestPreviewResult(
        document=preview,
        manifest_entry=build_manifest_entry(preview),
    )
