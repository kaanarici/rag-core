"""Tier-widening fail-closed seam tests for ``DeleteFilter`` and ``CollectionPolicy``.

Covers four guarantees for tiered deployments that run one ``Engine`` per
workspace/collection tier:

1. ``DeleteFilter.__post_init__`` rejects empty / blank ``document_id`` or
   ``collection`` strings so a formatting bug cannot silently widen a
   per-document delete to a collection-wide or namespace-wide one.
2. The retrieval facade (``Engine.search`` / ``Engine.context``)
   refuses ``collections=None`` and ``collections=[]`` because silent widening is
   forbidden.
3. ``CollectionPolicy`` validates namespace binding, allowed collections, rerank
   and lexical-sidecar capability flags from the engine seam before any
   provider call.
4. Explicit ``delete_collection`` and ``delete_namespace`` facade helpers exist
   so callers reach collection-wide or namespace-wide deletes deliberately, not
   by accident.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from rag_core import Engine
from rag_core.search.policy import CollectionPolicy, CollectionPolicyViolation
from rag_core.search.request_models import DeleteFilter

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


# --- DeleteFilter fail-closed guards ---------------------------------------


def test_delete_filter_rejects_empty_document_id() -> None:
    """A formatting bug producing ``document_id=''`` must not silently widen."""
    with pytest.raises(ValueError, match="DeleteFilter.document_id"):
        DeleteFilter(namespace="ns", collection="c", document_id="")


def test_delete_filter_rejects_whitespace_document_id() -> None:
    with pytest.raises(ValueError, match="DeleteFilter.document_id"):
        DeleteFilter(namespace="ns", collection="c", document_id="   ")


def test_delete_filter_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="DeleteFilter.collection"):
        DeleteFilter(namespace="ns", collection="", document_id="doc-1")


def test_delete_filter_requires_collection_or_document() -> None:
    """Bare ``DeleteFilter(namespace=...)`` would clear the namespace."""
    with pytest.raises(ValueError, match="at least one of collection or document_id"):
        DeleteFilter(namespace="ns")


def test_delete_filter_accepts_collection_only_for_explicit_collection_wide_delete() -> None:
    """The deliberate collection-wide path stays open via explicit None."""
    f = DeleteFilter(namespace="ns", collection="c")
    assert f.document_id is None
    assert f.collection == "c"


def test_delete_filter_accepts_document_with_collection_for_per_doc_delete() -> None:
    f = DeleteFilter(namespace="ns", collection="c", document_id="doc-1")
    assert f.document_id == "doc-1"


# --- CollectionPolicy enforcement ----------------------------------------------


def test_collection_policy_validates_bound_namespace() -> None:
    policy = CollectionPolicy(bound_namespace="workspace-alpha")
    policy.validate_namespace("workspace-alpha")  # ok
    with pytest.raises(CollectionPolicyViolation, match="bound to namespace"):
        policy.validate_namespace("workspace-beta")


def test_collection_policy_validates_allowed_collections() -> None:
    policy = CollectionPolicy(allowed_collections=frozenset({"public", "licensed"}))
    policy.validate_collections(["public"])  # ok
    policy.validate_collections(["licensed", "public"])  # ok
    with pytest.raises(CollectionPolicyViolation, match="refused collection"):
        policy.validate_collections(["restricted"])


def test_collection_policy_refuses_none_collections_when_allowlist_set() -> None:
    """Per slice: ``collections=None`` against a bound allowlist fails closed."""
    policy = CollectionPolicy(allowed_collections=frozenset({"public"}))
    with pytest.raises(CollectionPolicyViolation, match="silently widens"):
        policy.validate_collections(None)


def test_collection_policy_disallows_rerank_on_restricted_tier() -> None:
    policy = CollectionPolicy(allow_rerank=False)
    with pytest.raises(CollectionPolicyViolation, match="rerank"):
        policy.validate_search(
            namespace="ws",
            collections=["restricted"],
            rerank=True,
            use_lexical_search=False,
        )


def test_collection_policy_disallows_lexical_sidecar_on_restricted_tier() -> None:
    policy = CollectionPolicy(allow_lexical_sidecar=False)
    with pytest.raises(CollectionPolicyViolation, match="lexical sidecar"):
        policy.validate_search(
            namespace="ws",
            collections=["restricted"],
            rerank=False,
            use_lexical_search=True,
        )


def test_collection_policy_validates_allowed_query_plan_presets() -> None:
    policy = CollectionPolicy(allowed_query_plan_presets=frozenset({"dense_only"}))
    policy.validate_search(
        namespace="ws",
        collections=["c"],
        rerank=False,
        use_lexical_search=False,
        query_plan_preset="dense_only",
    )
    with pytest.raises(CollectionPolicyViolation, match="query_plan_preset"):
        policy.validate_search(
            namespace="ws",
            collections=["c"],
            rerank=False,
            use_lexical_search=False,
            query_plan_preset="hybrid_full",
        )


def test_collection_policy_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="bound_namespace"):
        CollectionPolicy(bound_namespace="   ")
    with pytest.raises(ValueError, match="allowed_collections"):
        CollectionPolicy(allowed_collections=frozenset({""}))


# --- Facade-level fail-closed collections -----------------------------------


def _make_core() -> tuple[Engine, RecordingVectorStore]:
    store = RecordingVectorStore()
    config = make_test_config(
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    core = Engine(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    return core, store


def _make_core_with_policy(policy: CollectionPolicy) -> tuple[Engine, RecordingVectorStore]:
    store = RecordingVectorStore()
    base = make_test_config(
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    core = Engine(
        replace(base, collection_policy=policy),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    return core, store


def test_search_facade_refuses_none_collections() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="collection or collections"):
                await core.search(
                    query="q",
                    namespace="ns",
                    collections=None,
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_search_facade_refuses_empty_collections() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="collections must not be empty"):
                await core.search(query="q", namespace="ns", collections=[])
        finally:
            await core.close()

    asyncio.run(_run())


def test_retrieve_context_facade_refuses_none_collections() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="collection or collections"):
                await core.context(
                    query="q",
                    namespace="ns",
                    collections=None,
                )
        finally:
            await core.close()

    asyncio.run(_run())


# --- CollectionPolicy wired through the engine ---------------------------------


def test_pipeline_runner_refuses_cross_namespace_request_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CollectionPolicy(bound_namespace="workspace-alpha")
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CollectionPolicyViolation, match="bound to namespace"):
                await core.search(
                    query="q",
                    namespace="workspace-beta",
                    collections=["public"],
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_pipeline_runner_refuses_disallowed_collection_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CollectionPolicy(allowed_collections=frozenset({"public"}))
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CollectionPolicyViolation, match="refused collection"):
                await core.search(
                    query="q",
                    namespace="ws",
                    collections=["restricted"],
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_pipeline_runner_refuses_rerank_when_policy_forbids_it() -> None:
    async def _run() -> None:
        policy = CollectionPolicy(allow_rerank=False)
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CollectionPolicyViolation, match="rerank"):
                await core.search(
                    query="q",
                    namespace="ws",
                    collections=["restricted"],
                    rerank=True,
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_pipeline_runner_refuses_disallowed_search_profile_under_policy() -> None:
    """An explicit caller plan is fenced by its search-profile name."""
    from rag_core.search.pipeline_runner import (
        SearchExecutionOptions,
        SearchPipelineRunner,
        SearchRequest,
    )
    from rag_core.search.query_plan_presets import search_profile

    async def _run() -> None:
        policy = CollectionPolicy(allowed_query_plan_presets=frozenset({"fast"}))
        runner = SearchPipelineRunner(
            FakeEmbeddingProvider(),
            FakeSparseEmbedder(),
            RecordingVectorStore(),
            collection_policy=policy,
        )
        plan = search_profile("balanced", limit=5)
        with pytest.raises(CollectionPolicyViolation, match="query_plan_preset"):
            await runner.search(
                SearchRequest(
                    query="q",
                    collections=["c"],
                    namespace="ws",
                    execution=SearchExecutionOptions(query_plan=plan),
                )
            )

    asyncio.run(_run())


def test_indexer_delete_refuses_cross_namespace_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CollectionPolicy(bound_namespace="workspace-alpha")
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CollectionPolicyViolation, match="bound to namespace"):
                await core.delete_document(
                    document_id="doc-1",
                    namespace="workspace-beta",
                    collection="public",
                )
        finally:
            await core.close()

    asyncio.run(_run())


# --- delete_collection / delete_namespace helpers ------------------------------


def test_delete_collection_helper_routes_through_explicit_seam() -> None:
    async def _run() -> None:
        core, store = _make_core()
        try:
            await core.add_bytes(
                file_bytes=b"alpha",
                filename="a.txt",
                mime_type="text/plain",
                namespace="ws",
                collection="c",
                path="/a.txt",
            )
            await core.delete_collection(namespace="ws", collection="c")
        finally:
            await core.close()

        # The recording store sees a collection-wide DeleteFilter with no
        # document_id. Exactly what the new explicit helper produces.
        assert any(
            call.collection == "c" and call.document_id is None
            for call in store.delete_calls
        ), f"expected explicit collection-wide delete; got {store.delete_calls!r}"

    asyncio.run(_run())


def test_delete_namespace_is_explicitly_reserved() -> None:
    """The namespace-wide path is not a silent default; it must be claimed."""
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(NotImplementedError, match="tenant offboarding"):
                await core.delete_namespace(namespace="ws")
        finally:
            await core.close()

    asyncio.run(_run())


def test_indexer_delete_document_refuses_empty_document_id() -> None:
    """The indexer seam refuses blank ``document_id`` even before DeleteFilter."""
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="document_id is required"):
                await core.delete_document(
                    document_id="",
                    namespace="ws",
                    collection="c",
                )
        finally:
            await core.close()

    asyncio.run(_run())
