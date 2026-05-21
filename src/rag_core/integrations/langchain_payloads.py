"""LangChain payload mapping helpers."""

from __future__ import annotations

from rag_core.contracts import search_user_documents_tool_result
from rag_core.integrations.integration_context_text import context_pack_model_text
from rag_core.search.context_pack import ModelContextPack
from rag_core.search.types import SearchResult


def search_result_to_document_kwargs(result: SearchResult) -> dict[str, object]:
    """Map one ``SearchResult`` into LangChain ``Document`` init kwargs."""

    metadata = dict(result.metadata)
    metadata.update(
        _compact(
            {
                "rag_core_result_id": result.id,
                "rag_core_score": result.score,
                "rag_core_content_type": result.content_type,
                "rag_core_source_type": result.source_type,
                "rag_core_document_id": result.document_id,
                "rag_core_corpus_id": result.corpus_id,
                "rag_core_title": result.title,
                "rag_core_section_id": result.section_id,
                "rag_core_section_title": result.section_title,
                "rag_core_section_path": result.section_path,
                "rag_core_chunk_index": result.chunk_index,
                "rag_core_result_type": result.result_type,
                "rag_core_figure_id": result.figure_id,
                "rag_core_figure_thumbnail_url": result.figure_thumbnail_url,
            }
        )
    )
    return {
        "id": result.id,
        "page_content": result.text,
        "metadata": metadata,
    }


def context_pack_to_tool_output(pack: ModelContextPack) -> tuple[str, dict[str, object]]:
    """Map ``ModelContextPack`` into a `(content, artifact)` tool response."""

    return context_pack_model_text(pack), search_user_documents_tool_result(pack)


def _compact(data: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in data.items() if value is not None}
