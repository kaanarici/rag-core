"""Retrieve stage: query-plan-aware embedding and vector-store search.

Emits embed events on the pipeline's event sink.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import EmbedCompleted, EmbedRequested
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)
from rag_core.search.embedding_cache_diagnostics import (
    embed_query_with_cache_observation,
)
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.planning import (
    QueryPlanPreparer,
    QueryPlanSparseVectorPolicy,
    default_query_plan_for_store,
    rerank_candidate_pool_limit,
    validate_query_plan_for_store,
)
from rag_core.search.query_plan import (
    DenseChannel,
    QueryPlan,
    SparseChannel,
    UnsupportedQueryStage,
    flatten_prefetches,
)
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL, primary_sparse_channel
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    SparseEmbedder,
    provider_name,
)
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import SearchResult, SparseVector

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


class HybridRetrieve:
    """Embed only the query channels required by the resolved query plan."""

    async def retrieve(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]:
        sink = ctx.event_sink
        embedding = ctx.embedding_provider
        sparse = ctx.sparse_embedder

        if (
            query.query_plan is None
            and isinstance(ctx.vector_store, QueryPlanPreparer)
            and not query.state.collection_ensured
        ):
            await ctx.vector_store.ensure_collection()
            query.state.collection_ensured = True
        capabilities = ctx.vector_store.capabilities.query_plan
        # When a real provider reranker will run, fetch a wider candidate pool
        # than the caller's final limit so the reranker can promote documents
        # ranked below it. The final result count is unchanged: the rerank stage
        # and the pipeline both slice back to query.limit. Only the default plan
        # is widened; an explicit caller plan stays authoritative.
        retrieve_limit = _retrieve_pool_limit(query, reranker=ctx.reranker)
        plan: QueryPlan | None
        if query.query_plan is not None:
            plan = query.query_plan
            _canonicalize_plan_limit(query, plan)
        else:
            plan = default_query_plan_for_store(
                store=ctx.vector_store,
                capabilities=capabilities,
                result_limit=retrieve_limit,
            )
        if plan is not None:
            validate_query_plan_for_store(
                plan,
                capabilities=capabilities,
                provider_name=provider_name(ctx.vector_store),
                store=ctx.vector_store,
            )
            if isinstance(ctx.vector_store, QueryPlanPreparer):
                await ctx.vector_store.prepare_query_plan(plan)

        needs_dense = _plan_uses_dense(plan)
        needs_sparse = _plan_uses_sparse(plan, store=ctx.vector_store)
        if (
            needs_sparse
            and query.query_sparse_vectors is None
            and sparse is None
        ):
            raise UnsupportedQueryStage(
                "This Engine was configured for dense retrieval only. "
                "Set IngestConfig(build_sparse_index=True) and rebuild the "
                "index before requesting lexical or hybrid search."
            )
        dense_task: asyncio.Task[tuple[list[float], object]] | None = None
        sparse_task: asyncio.Task[dict[str, SparseVector]] | None = None

        if not needs_dense:
            dense_vec = []
        elif query.query_vector is not None:
            dense_vec = query.query_vector
        else:
            dense_query_text = query.dense_query_text or query.query
            dense_task = asyncio.create_task(
                _embed_dense_query(embedding, dense_query_text, sink)
            )

        if query.query_sparse_vectors is not None:
            sparse_vectors = query.query_sparse_vectors
        elif not needs_sparse:
            sparse_vectors = {}
        elif sparse is None:
            raise UnsupportedQueryStage(
                "This Engine was configured for dense retrieval only. "
                "Set IngestConfig(build_sparse_index=True) and rebuild the "
                "index before requesting lexical or hybrid search."
            )
        else:
            sparse_task = asyncio.create_task(
                _embed_sparse_query_async(sparse, query.query, sink)
            )

        try:
            if dense_task is not None:
                dense_result = await dense_task
                dense_vec = dense_result[0]
            if sparse_task is not None:
                sparse_vectors = await sparse_task
        except BaseException:
            # CancelledError must reach here too: without this, cancelling a
            # search leaves the sibling embedding task running provider work.
            pending = [
                task
                for task in (dense_task, sparse_task)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise
        if needs_sparse:
            if plan is not None:
                missing_channels = _missing_sparse_query_vectors(plan, sparse_vectors)
                if missing_channels:
                    raise UnsupportedQueryStage(
                        "Sparse query vectors were not produced for required channels: "
                        f"{', '.join(sorted(missing_channels))}"
                    )
            elif not sparse_vectors:
                raise UnsupportedQueryStage(
                    "Sparse query vectors were not produced for the requested search"
                )
        if needs_dense and not dense_vec:
            if query.query_vector is not None:
                dense_vec = query.query_vector
            else:
                dense_query_text = query.dense_query_text or query.query
                dense_result = await _embed_dense_query(embedding, dense_query_text, sink)
                dense_vec = dense_result[0]

        if needs_sparse:
            try:
                primary_sparse = primary_sparse_channel(
                    sparse_vectors,
                    missing_message="No sparse query vector generated",
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        else:
            primary_sparse = SparseVector(indices=[], values=[])
        vector_query = SearchQuery(
            dense_vector=dense_vec,
            sparse_vector=primary_sparse,
            sparse_vectors=sparse_vectors,
            namespace=query.namespace,
            collections=query.collections,
            content_types=query.content_types,
            document_ids=query.document_ids,
            limit=retrieve_limit,
            query_plan=plan,
            metadata_filter=query.metadata_filter,
            lexical_query=query.query.strip() or None,
        )
        query.query_plan = plan
        query.emit_query_plan_trace(plan, query.limit)
        return await ctx.vector_store.search(vector_query)


async def _embed_dense_query(
    embedding: EmbeddingProvider,
    query: str,
    sink: "EventSink | None",
) -> tuple[list[float], object]:
    dense_provider = provider_name(embedding)
    dense_model = getattr(embedding, "model_name", "")
    emit_event(
        sink,
        EmbedRequested(
            provider=dense_provider,
            model=dense_model,
            text_count=1,
            role=DENSE_RETRIEVAL_CHANNEL,
        ),
    )
    dense_started_ms = now_ms()
    dense_vec, dense_cache = await embed_query_with_cache_observation(
        embedding, query
    )
    emit_event(
        sink,
        EmbedCompleted(
            provider=dense_provider,
            model=dense_model,
            text_count=1,
            role=DENSE_RETRIEVAL_CHANNEL,
            duration_ms=now_ms() - dense_started_ms,
            cache_hits=dense_cache.hits,
            cache_misses=dense_cache.misses,
            cache_writes=dense_cache.writes,
            cache_bypasses=dense_cache.bypasses,
        ),
    )
    return dense_vec, dense_cache


async def _embed_sparse_query_async(
    sparse_embedder: SparseEmbedder,
    query: str,
    sink: "EventSink | None",
) -> dict[str, SparseVector]:
    sparse_provider = provider_name(sparse_embedder)
    sparse_model = getattr(sparse_embedder, "model_name", "")
    emit_event(
        sink,
        EmbedRequested(
            provider=sparse_provider,
            model=sparse_model,
            text_count=1,
            role=SPARSE_RETRIEVAL_CHANNEL,
        ),
    )
    sparse_started_ms = now_ms()
    sparse_vectors = await asyncio.to_thread(sparse_embedder.embed_query_multi, query)
    emit_event(
        sink,
        EmbedCompleted(
            provider=sparse_provider,
            model=sparse_model,
            text_count=1,
            role=SPARSE_RETRIEVAL_CHANNEL,
            duration_ms=now_ms() - sparse_started_ms,
        ),
    )
    return sparse_vectors


def _plan_uses_sparse(plan: QueryPlan | None, *, store: object) -> bool:
    if isinstance(store, QueryPlanSparseVectorPolicy):
        return store.query_plan_needs_sparse_vectors(plan)
    if plan is None:
        return True
    return any(
        isinstance(prefetch.channel, SparseChannel)
        for prefetch in flatten_prefetches(plan.prefetches)
    )


def _plan_uses_dense(plan: QueryPlan | None) -> bool:
    if plan is None or plan.rerank is not None:
        return True
    return any(
        isinstance(prefetch.channel, DenseChannel)
        for prefetch in flatten_prefetches(plan.prefetches)
    )


def _canonicalize_plan_limit(query: PipelineQuery, plan: QueryPlan | None) -> None:
    if plan is None or plan.final_limit == query.limit:
        return
    query.limit = plan.final_limit


def _retrieve_pool_limit(query: PipelineQuery, *, reranker: object | None) -> int:
    """Candidate count the retrieve stage should fetch.

    For an explicit caller plan, honor its ``final_limit``. For the default
    plan, widen to a rerank candidate pool when a real provider reranker will
    run so it can promote documents ranked below the final limit; otherwise use
    the caller's limit. The final result count is unchanged either way.
    """
    if query.query_plan is not None:
        return query.query_plan.final_limit
    if not (query.rerank and reranker is not None):
        return query.limit
    requested = (
        query.rerank_budget.candidate_count
        if query.rerank_budget is not None
        else None
    )
    return rerank_candidate_pool_limit(final_limit=query.limit, requested=requested)


def _missing_sparse_query_vectors(
    plan: QueryPlan,
    sparse_vectors: dict[str, SparseVector],
) -> set[str]:
    available = set(sparse_vectors)
    if sparse_vectors:
        available.add(PRIMARY_SPARSE_CHANNEL)
    required = {
        prefetch.channel.using_query_vector
        for prefetch in flatten_prefetches(plan.prefetches)
        if isinstance(prefetch.channel, SparseChannel)
    }
    return required - available
