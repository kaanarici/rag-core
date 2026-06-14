from __future__ import annotations

from rag_core.search.context_pack_helpers import (
    base_source_id as _base_source_id,
    metadata_bbox as _metadata_bbox,
    metadata_int as _metadata_int,
    metadata_str as _metadata_str,
    unique_source_id as _unique_source_id,
)
from rag_core.search.context_pack_models import SourceLocator, SourceReference
from rag_core.search.vector_models import SearchResult


def source_reference_from_result(
    result: SearchResult,
    *,
    used_source_ids: set[str] | None = None,
    require_stable_suffix: bool = False,
) -> SourceReference:
    source_id = _unique_source_id(
        _base_source_id(result),
        used_source_ids,
        stable_suffix=result.id,
        require_stable_suffix=require_stable_suffix,
    )
    return SourceReference(
        source_id=source_id,
        result_id=result.id,
        document_id=result.document_id,
        corpus_id=result.corpus_id,
        document_key=result.document_key,
        title=result.title,
        section_id=result.section_id,
        section_title=result.section_title,
        section_path=result.section_path,
        chunk_index=result.chunk_index,
        content_sha256=result.content_sha256,
        source_type=result.source_type,
        result_type=result.result_type,
    )


def source_locator_from_result(result: SearchResult) -> SourceLocator:
    if result.metadata.get("offset_reconstruction") == "unreliable":
        return SourceLocator(
            chunk_index=result.chunk_index,
            source_hash=result.content_sha256,
            section_path=result.section_path,
        )
    start_offset = result.start_char
    if start_offset is None:
        start_offset = _metadata_int(result.metadata, "start_char")
    end_offset = result.end_char
    if end_offset is None:
        end_offset = _metadata_int(result.metadata, "end_char")
    return SourceLocator(
        chunk_index=result.chunk_index,
        source_hash=result.content_sha256,
        section_path=result.section_path,
        page_number=_metadata_int(result.metadata, "page_number"),
        page_index=_metadata_int(result.metadata, "page_index"),
        slide_number=_metadata_int(result.metadata, "slide_number"),
        sheet_name=_metadata_str(result.metadata, "sheet_name"),
        row_range=_metadata_str(result.metadata, "row_range"),
        line_start=_metadata_int(result.metadata, "line_start"),
        line_end=_metadata_int(result.metadata, "line_end"),
        start_offset=start_offset,
        end_offset=end_offset,
        bbox=_metadata_bbox(result.metadata.get("bbox") or result.metadata.get("figure_bbox")),
        figure_id=result.figure_id,
        figure_caption=_metadata_str(result.metadata, "figure_caption"),
        figure_thumbnail_url=result.figure_thumbnail_url
        or _metadata_str(result.metadata, "figure_thumbnail_url"),
    )
