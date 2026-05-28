"""Default Retrieve stage: capability-aware query embedding and vector-store search.

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
    QUERY_PLAN_PRESET_DENSE_ONLY,
    QueryPlanPreparer,
    default_query_plan_for_store,
    query_plan_preset,
    resolve_prefetch_limit,
    validate_query_plan_for_store,
)
from rag_core.search.query_plan import DenseChannel, Prefetch, QueryPlan, SparseChannel
from rag_core.search.sparse_channels import PRIMARY_SPARSE_CHANNEL, primary_sparse_channel
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    QueryPlanCapabilities,
    SparseEmbedder,
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
        plan = query.query_plan or default_query_plan_for_store(
            store=ctx.vector_store,
            capabilities=capabilities,
            result_limit=query.limit,
        )
        _canonicalize_plan_limit(query, plan)
        if plan is not None:
            validate_query_plan_for_store(
                plan,
                capabilities=capabilities,
                provider_name=_provider_name(ctx.vector_store),
                store=ctx.vector_store,
            )
            if isinstance(ctx.vector_store, QueryPlanPreparer):
                await ctx.vector_store.prepare_query_plan(plan)

        needs_dense = _plan_uses_dense(plan)
        needs_sparse = _plan_uses_sparse(plan)
        dense_task: asyncio.Task[tuple[list[float], object]] | None = None
        sparse_task: asyncio.Task[dict[str, SparseVector]] | None = None

        if not needs_dense:
            dense_vec = []
        elif query.query_vector is not None:
            dense_vec = query.query_vector
        else:
            dense_task = asyncio.create_task(
                _embed_dense_query(embedding, query.query, sink)
            )

        if query.query_sparse_vectors is not None:
            sparse_vectors = query.query_sparse_vectors
        elif not needs_sparse:
            sparse_vectors = {}
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
        except Exception:
            for task in (dense_task, sparse_task):
                if task is not None and not task.done():
                    task.cancel()
            raise
        resolved_plan = _reconcile_implicit_default_plan(
            plan=plan,
            query=query,
            capabilities=capabilities,
            sparse_vectors=sparse_vectors,
        )
        if resolved_plan != plan:
            plan = resolved_plan
            if plan is not None:
                validate_query_plan_for_store(
                    plan,
                    capabilities=capabilities,
                    provider_name=_provider_name(ctx.vector_store),
                    store=ctx.vector_store,
                )
                if isinstance(ctx.vector_store, QueryPlanPreparer):
                    await ctx.vector_store.prepare_query_plan(plan)
        needs_dense = _plan_uses_dense(plan)
        needs_sparse = _plan_uses_sparse(plan)
        if needs_dense and not dense_vec:
            if query.query_vector is not None:
                dense_vec = query.query_vector
            else:
                dense_result = await _embed_dense_query(embedding, query.query, sink)
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
            corpus_ids=query.corpus_ids,
            content_types=query.content_types,
            document_ids=query.document_ids,
            limit=query.limit,
            query_plan=plan,
            metadata_filter=query.metadata_filter,
            lexical_query=query.query.strip() or None,
        )
        query.query_plan = plan
        query.emit_query_plan_trace(plan, query.limit)
        return await ctx.vector_store.search(vector_query)


def _provider_name(provider: object) -> str:
    name = getattr(provider, "provider_name", None)
    if isinstance(name, str) and name:
        return name
    return type(provider).__name__


def _embed_sparse_query(
    sparse_embedder: SparseEmbedder,
    query: str,
) -> dict[str, SparseVector]:
    return sparse_embedder.embed_query_multi(query)


async def _embed_dense_query(
    embedding: EmbeddingProvider,
    query: str,
    sink: "EventSink | None",
) -> tuple[list[float], object]:
    dense_provider = _provider_name(embedding)
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
    sparse_provider = _provider_name(sparse_embedder)
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
    sparse_vectors = await asyncio.to_thread(_embed_sparse_query, sparse_embedder, query)
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


def _plan_uses_sparse(plan: QueryPlan | None) -> bool:
    if plan is None:
        return True
    return any(_prefetch_uses_sparse(prefetch) for prefetch in plan.prefetches)


def _plan_uses_dense(plan: QueryPlan | None) -> bool:
    if plan is None:
        return True
    if plan.rerank is not None:
        return True
    return any(_prefetch_uses_dense(prefetch) for prefetch in plan.prefetches)


def _prefetch_uses_sparse(prefetch: Prefetch) -> bool:
    if isinstance(prefetch.channel, SparseChannel):
        return True
    return any(_prefetch_uses_sparse(nested) for nested in prefetch.nested)


def _prefetch_uses_dense(prefetch: Prefetch) -> bool:
    if isinstance(prefetch.channel, DenseChannel):
        return True
    return any(_prefetch_uses_dense(nested) for nested in prefetch.nested)


def _canonicalize_plan_limit(query: PipelineQuery, plan: QueryPlan | None) -> None:
    if plan is None or plan.final_limit == query.limit:
        return
    query.limit = plan.final_limit


def _reconcile_implicit_default_plan(
    *,
    plan: QueryPlan | None,
    query: PipelineQuery,
    capabilities: QueryPlanCapabilities,
    sparse_vectors: dict[str, SparseVector],
) -> QueryPlan | None:
    if plan is None or query.query_plan is not None:
        return plan
    missing_channels = _missing_sparse_query_vectors(plan, sparse_vectors)
    if not missing_channels:
        return plan
    available_plan = _plan_with_available_sparse_channels(
        plan,
        sparse_vectors=sparse_vectors,
    )
    if available_plan is not None:
        return available_plan
    if capabilities.dense:
        return query_plan_preset(QUERY_PLAN_PRESET_DENSE_ONLY, limit=query.limit)
    sparse_channel = next(iter(sparse_vectors), "")
    if not sparse_channel:
        return plan
    sparse_limit = resolve_prefetch_limit(result_limit=query.limit)
    return QueryPlan(
        prefetches=(
            Prefetch(
                channel=SparseChannel(
                    vector_field=sparse_channel,
                    using_query_vector=sparse_channel,
                ),
                limit=sparse_limit,
            ),
        ),
        final_limit=query.limit,
    )


def _plan_with_available_sparse_channels(
    plan: QueryPlan,
    *,
    sparse_vectors: dict[str, SparseVector],
) -> QueryPlan | None:
    available = set(sparse_vectors)
    if sparse_vectors:
        available.add(PRIMARY_SPARSE_CHANNEL)
    prefetches = tuple(
        filtered
        for prefetch in plan.prefetches
        if (filtered := _prefetch_with_available_sparse_channels(prefetch, available))
        is not None
    )
    if not prefetches:
        return None
    return QueryPlan(
        prefetches=prefetches,
        fuse=plan.fuse if len(prefetches) > 1 else None,
        rerank=plan.rerank,
        boost=plan.boost,
        final_limit=plan.final_limit,
        search_profile=plan.search_profile if prefetches == plan.prefetches else None,
    )


def _prefetch_with_available_sparse_channels(
    prefetch: Prefetch,
    available_sparse_vectors: set[str],
) -> Prefetch | None:
    channel = prefetch.channel
    if (
        isinstance(channel, SparseChannel)
        and channel.using_query_vector not in available_sparse_vectors
    ):
        return None
    nested = tuple(
        filtered
        for child in prefetch.nested
        if (
            filtered := _prefetch_with_available_sparse_channels(
                child,
                available_sparse_vectors,
            )
        )
        is not None
    )
    if nested == prefetch.nested:
        return prefetch
    return Prefetch(channel=channel, limit=prefetch.limit, nested=nested)


def _missing_sparse_query_vectors(
    plan: QueryPlan,
    sparse_vectors: dict[str, SparseVector],
) -> set[str]:
    available = set(sparse_vectors)
    if sparse_vectors:
        available.add(PRIMARY_SPARSE_CHANNEL)
    required: set[str] = set()
    for prefetch in plan.prefetches:
        _collect_sparse_query_vector_names(prefetch, required)
    return required - available


def _collect_sparse_query_vector_names(prefetch: Prefetch, names: set[str]) -> None:
    if isinstance(prefetch.channel, SparseChannel):
        names.add(prefetch.channel.using_query_vector)
    for nested in prefetch.nested:
        _collect_sparse_query_vector_names(nested, names)
