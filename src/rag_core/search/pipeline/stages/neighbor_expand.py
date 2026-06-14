"""Neighbor-chunk context expansion postprocess stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from rag_core.events.emit import emit_event
from rag_core.events.types import NeighborExpandSkipped
from rag_core.search.context_pack import (
    CONTEXT_EXPANSION_AFTER_METADATA_KEY,
    CONTEXT_EXPANSION_BEFORE_METADATA_KEY,
)
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.vector_models import SearchResult

_DocKey = tuple[str, str, str]
_ChunkKey = tuple[str, str, str, int]


class NeighborExpandPostprocess:
    """Attach adjacent same-document chunk text for context packing."""

    def __init__(self, *, window: int = 1, max_hits: int = 5) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError("window must be a non-negative integer")
        if isinstance(max_hits, bool) or not isinstance(max_hits, int) or max_hits <= 0:
            raise ValueError("max_hits must be a positive integer")
        self._window = window
        self._max_hits = max_hits

    async def postprocess(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        if not results or self._window == 0:
            return results
        if not ctx.vector_store.capabilities.chunk_index_lookup:
            emit_event(
                ctx.event_sink,
                NeighborExpandSkipped(
                    reason="unsupported_store",
                    input_count=len(results),
                    result_count=len(results),
                ),
            )
            return results

        plans = _expansion_plans(results[: self._max_hits], query, self._window)
        if not plans:
            return results

        chunks_by_doc = await _fetch_chunks_by_doc(ctx, plans)
        claimed: set[_ChunkKey] = set()
        expanded: list[SearchResult] = []
        for result in results:
            plan = plans.get(result.id)
            if plan is None:
                expanded.append(result)
                continue
            expanded.append(_expand_result(result, plan, chunks_by_doc, claimed))
        return expanded


def _expansion_plans(
    results: Sequence[SearchResult],
    query: PipelineQuery,
    window: int,
) -> dict[str, tuple[_DocKey, int, tuple[int, ...]]]:
    plans: dict[str, tuple[_DocKey, int, tuple[int, ...]]] = {}
    for result in results:
        doc_key = _doc_key(result, query)
        chunk_index = result.chunk_index
        if doc_key is None or isinstance(chunk_index, bool) or chunk_index is None:
            continue
        if chunk_index < 0:
            continue
        start = max(0, chunk_index - window)
        stop = chunk_index + window
        plans[result.id] = (doc_key, chunk_index, tuple(range(start, stop + 1)))
    return plans


async def _fetch_chunks_by_doc(
    ctx: PipelineContext,
    plans: dict[str, tuple[_DocKey, int, tuple[int, ...]]],
) -> dict[_DocKey, dict[int, SearchResult]]:
    indices_by_doc: defaultdict[_DocKey, set[int]] = defaultdict(set)
    for doc_key, _, indices in plans.values():
        indices_by_doc[doc_key].update(indices)

    chunks_by_doc: dict[_DocKey, dict[int, SearchResult]] = {}
    for doc_key, doc_indices in indices_by_doc.items():
        namespace, corpus_id, document_id = doc_key
        chunks = await ctx.vector_store.get_chunks_by_index(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            chunk_indices=tuple(sorted(doc_indices)),
        )
        requested_indices = set(doc_indices)
        validated_chunks: dict[int, SearchResult] = {}
        for chunk in chunks:
            chunk_index = _requested_chunk_index(chunk, doc_key, requested_indices)
            if chunk_index is not None:
                validated_chunks[chunk_index] = chunk
        chunks_by_doc[doc_key] = validated_chunks
    return chunks_by_doc


def _requested_chunk_index(
    chunk: SearchResult,
    doc_key: _DocKey,
    requested_indices: set[int],
) -> int | None:
    chunk_index = chunk.chunk_index
    if isinstance(chunk_index, bool) or chunk_index is None:
        return None
    namespace, corpus_id, document_id = doc_key
    if chunk_index not in requested_indices:
        return None
    if (
        chunk.namespace != namespace
        or chunk.corpus_id != corpus_id
        or chunk.document_id != document_id
    ):
        return None
    return chunk_index


def _expand_result(
    result: SearchResult,
    plan: tuple[_DocKey, int, tuple[int, ...]],
    chunks_by_doc: dict[_DocKey, dict[int, SearchResult]],
    claimed: set[_ChunkKey],
) -> SearchResult:
    doc_key, center_index, indices = plan
    chunks = chunks_by_doc.get(doc_key, {})
    before: list[str] = []
    after: list[str] = []
    included_indices: list[int] = [center_index]
    for index in indices:
        if index == center_index:
            continue
        chunk_key = (*doc_key, index)
        if chunk_key in claimed:
            continue
        chunk = chunks.get(index)
        if chunk is None:
            continue
        if index < center_index:
            before.append(chunk.text)
        else:
            after.append(chunk.text)
        included_indices.append(index)

    for index in included_indices:
        claimed.add((*doc_key, index))
    if not before and not after:
        return result

    return replace(
        result,
        metadata={
            **result.metadata,
            CONTEXT_EXPANSION_BEFORE_METADATA_KEY: before,
            CONTEXT_EXPANSION_AFTER_METADATA_KEY: after,
        },
    )


def _doc_key(result: SearchResult, query: PipelineQuery) -> _DocKey | None:
    namespace = (result.namespace or query.namespace).strip()
    corpus_id = result.corpus_id
    if corpus_id is None and len(query.corpus_ids) == 1:
        corpus_id = query.corpus_ids[0]
    document_id = result.document_id
    if not namespace or not corpus_id or not corpus_id.strip():
        return None
    if not document_id or not document_id.strip():
        return None
    return namespace, corpus_id.strip(), document_id.strip()


__all__ = ["NeighborExpandPostprocess"]
