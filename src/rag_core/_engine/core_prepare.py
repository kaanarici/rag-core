from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core._engine.core_prepare_events import (
    emit_chunk_produced,
    emit_ocr_applied,
    emit_parse_completed,
)
from rag_core._engine.core_ocr_metadata import write_ocr_metadata
from rag_core.events.emit import now_ms, stage_guard
from rag_core.config import ChunkingConfig

from rag_core.core_models import (
    OcrMetadata,
    ParsedDocument,
    PreparedDocument,
)
from rag_core.documents.prepare_chunks import (
    prepare_pre_chunked_texts as prepare_pre_chunked_texts,
    prepare_text_chunks_async,
    prepare_text_chunks as prepare_text_chunks,
)
from rag_core._engine.core_prepare_contextualizer import (
    apply_contextualizer as _apply_contextualizer,
)
from rag_core._engine.core_prepare_figure_locators import (
    with_figure_locators as _with_figure_locators,
)
from rag_core._engine.core_prepare_metadata import (
    build_ocr_signal,
    merge_markdown,
    normalize_page_indices,
    resolve_document_page_count as _resolve_document_page_count,
    resolve_ocr_page_count as _resolve_ocr_page_count,
    resolve_ocr_pages_used as _resolve_ocr_pages_used,
)
from rag_core.documents.quality_metadata import quality_score_to_metadata

if TYPE_CHECKING:
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.documents.ocr import OcrProvider
    from rag_core.events.sink import EventSink
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache


async def parse_document_bytes(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    path: str | None = None,
    event_sink: "EventSink | None" = None,
) -> ParsedDocument:
    from rag_core.documents.local_parse import parse_file_bytes

    started_ms = now_ms()
    with stage_guard(event_sink, stage="parse"):
        markdown, metadata = await parse_file_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
        )
    parsed = ParsedDocument(
        filename=filename,
        mime_type=mime_type,
        markdown=markdown,
        metadata=metadata,
        path=path,
    )
    emit_parse_completed(
        event_sink,
        filename=filename,
        mime_type=mime_type,
        metadata=metadata,
        duration_ms=now_ms() - started_ms,
    )
    return parsed


async def prepare_document_bytes(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    path: str | None,
    ocr_provider: OcrProvider | None,
    allow_needs_ocr: bool = False,
    event_sink: "EventSink | None" = None,
    contextualizer: "ChunkContextualizer | None" = None,
    chunk_context_cache: "ChunkContextCache | None" = None,
    chunking_config: ChunkingConfig | None = None,
    namespace: str = "",
    collection: str = "",
    document_id: str = "",
) -> PreparedDocument:
    parsed = await parse_document_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        path=path,
        event_sink=event_sink,
    )
    if ocr_provider is not None and parsed.metadata.get("needs_ocr"):
        parsed = await apply_ocr(
            parsed=parsed,
            file_bytes=file_bytes,
            provider=ocr_provider,
            event_sink=event_sink,
        )
    if parsed.metadata.get("needs_ocr") and not allow_needs_ocr:
        raise ValueError(
            f"{filename} requires OCR, but no OCR provider remediated the document"
        )

    chunks = await prepare_text_chunks_async(
        parsed.markdown,
        mime_type=mime_type,
        filename=filename,
        chunking_config=chunking_config,
    )
    chunks = _with_figure_locators(chunks=chunks, metadata=parsed.metadata)
    chunking_strategy = chunks[0].chunking_strategy if chunks else "none"
    emit_chunk_produced(
        event_sink,
        filename=parsed.filename,
        chunk_count=len(chunks),
        chunking_strategy=chunking_strategy,
    )
    chunks = await _apply_contextualizer(
        chunks=chunks,
        markdown=parsed.markdown,
        filename=parsed.filename,
        namespace=namespace,
        collection=collection,
        document_id=document_id,
        contextualizer=contextualizer,
        chunk_context_cache=chunk_context_cache,
        event_sink=event_sink,
    )

    return PreparedDocument(
        filename=parsed.filename,
        mime_type=parsed.mime_type,
        markdown=parsed.markdown,
        chunks=chunks,
        metadata=parsed.metadata,
        path=parsed.path,
        ocr=build_ocr_signal(parsed.metadata),
    )


async def apply_ocr(
    *,
    parsed: ParsedDocument,
    file_bytes: bytes,
    provider: OcrProvider,
    event_sink: "EventSink | None" = None,
) -> ParsedDocument:
    from rag_core.documents.ocr import OcrRequest

    started_ms = now_ms()
    page_indices = normalize_page_indices(parsed.metadata.get("ocr_page_indices"))
    with stage_guard(event_sink, stage="ocr"):
        ocr_result = await provider.extract_markdown(
            OcrRequest(
                file_bytes=file_bytes,
                filename=parsed.filename,
                mime_type=parsed.mime_type,
                page_indices=page_indices,
                existing_markdown=parsed.markdown,
                metadata=dict(parsed.metadata),
            )
        )
    if not ocr_result.markdown.strip():
        raise ValueError("OCR provider returned empty markdown")
    processed_pages = normalize_page_indices(ocr_result.pages_processed)
    if (
        page_indices
        and not bool(ocr_result.metadata.get("ocr_processed_entire_document"))
        and set(processed_pages) != set(page_indices)
    ):
        missing_pages = sorted(set(page_indices) - set(processed_pages))
        raise ValueError(
            "OCR provider did not return all requested pages "
            f"(missing page indices: {missing_pages})"
        )
    merged_metadata = dict(parsed.metadata)
    merged_metadata.update(ocr_result.metadata)
    pages_used = _resolve_ocr_pages_used(
        parsed_metadata=parsed.metadata,
        ocr_result=ocr_result,
        requested_page_indices=page_indices,
    )
    page_count = _resolve_ocr_page_count(
        parsed_metadata=parsed.metadata,
        ocr_result=ocr_result,
        ocr_pages_used=pages_used,
        requested_page_indices=page_indices,
    )
    ocr_metadata = OcrMetadata(
        provider=ocr_result.provider_name,
        model=ocr_result.model_name,
        pages_used=tuple(pages_used),
        page_count=page_count,
        merge_mode=ocr_result.merge_mode,
    )
    write_ocr_metadata(merged_metadata, ocr_metadata)
    if bool(ocr_result.metadata.get("ocr_processed_entire_document")):
        merged_metadata.pop("ocr_page_indices", None)
        merged_metadata.pop("ocr_page_indices_telemetry", None)
        if not pages_used and page_count == 0:
            merged_metadata["ocr_page_count_unknown"] = True
    elif processed_pages:
        merged_metadata["ocr_page_indices"] = processed_pages
    merged_metadata["needs_ocr"] = False
    merged_markdown = merge_markdown(parsed.markdown, ocr_result)
    from rag_core.documents.converters.quality import score_text_quality

    document_page_count = _resolve_document_page_count(
        parsed_metadata=parsed.metadata,
        ocr_metadata=ocr_result.metadata,
    )
    merged_metadata["quality"] = quality_score_to_metadata(
        score_text_quality(
            merged_markdown,
            page_count=document_page_count
            if document_page_count is not None
            else page_count,
        )
    )
    emit_ocr_applied(
        event_sink,
        filename=parsed.filename,
        provider=ocr_result.provider_name or "",
        pages_processed=page_count,
        duration_ms=now_ms() - started_ms,
    )
    return ParsedDocument(
        filename=parsed.filename,
        mime_type=parsed.mime_type,
        markdown=merged_markdown,
        metadata=merged_metadata,
        path=parsed.path,
    )
