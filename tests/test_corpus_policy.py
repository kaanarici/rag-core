"""Fail-closed seam tests for ``DeleteFilter`` and ``CollectionPolicy``.

Covers four guarantees:

1. ``DeleteFilter.__post_init__`` rejects empty / blank ``document_id`` or
   ``collection`` strings so a formatting bug cannot silently widen a
   per-document delete to a collection-wide or tenant-wide one.
2. The retrieval facade (``Engine.search`` / ``Engine.context``)
   refuses ``collections=None`` and ``collections=[]`` because silent widening is
   forbidden.
3. ``CollectionPolicy`` validates tenant_id binding and allowed collections
   from the engine seam before any provider call.
4. Explicit ``delete_collection`` and ``delete_tenant`` facade helpers exist
   so callers reach collection-wide or tenant-wide deletes deliberately, not
   by accident.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from rag_core.core import Engine
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
        DeleteFilter(tenant_id="ns", collection="c", document_id="")


def test_delete_filter_rejects_whitespace_document_id() -> None:
    with pytest.raises(ValueError, match="DeleteFilter.document_id"):
        DeleteFilter(tenant_id="ns", collection="c", document_id="   ")


def test_delete_filter_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="DeleteFilter.collection"):
        DeleteFilter(tenant_id="ns", collection="", document_id="doc-1")


def test_delete_filter_requires_collection_or_document() -> None:
    """Bare ``DeleteFilter(tenant_id=...)`` would clear the tenant_id."""
    with pytest.raises(ValueError, match="at least one of collection or document_id"):
        DeleteFilter(tenant_id="ns")


def test_delete_filter_accepts_collection_only_for_explicit_collection_wide_delete() -> None:
    """The deliberate collection-wide path stays open via explicit None."""
    f = DeleteFilter(tenant_id="ns", collection="c")
    assert f.document_id is None
    assert f.collection == "c"


def test_delete_filter_accepts_document_with_collection_for_per_doc_delete() -> None:
    f = DeleteFilter(tenant_id="ns", collection="c", document_id="doc-1")
    assert f.document_id == "doc-1"


# --- CollectionPolicy enforcement ----------------------------------------------


def test_collection_policy_validates_bound_tenant_id() -> None:
    policy = CollectionPolicy(bound_tenant_id="workspace-alpha")
    policy.validate_tenant_id("workspace-alpha")  # ok
    with pytest.raises(CollectionPolicyViolation, match="bound to tenant_id"):
        policy.validate_tenant_id("workspace-beta")


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


def test_collection_policy_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="bound_tenant_id"):
        CollectionPolicy(bound_tenant_id="   ")
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
                    tenant_id="ns",
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
                await core.search(query="q", tenant_id="ns", collections=[])
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
                    tenant_id="ns",
                    collections=None,
                )
        finally:
            await core.close()

    asyncio.run(_run())


# --- CollectionPolicy wired through the engine ---------------------------------


def test_pipeline_runner_refuses_cross_tenant_request_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CollectionPolicy(bound_tenant_id="workspace-alpha")
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CollectionPolicyViolation, match="bound to tenant_id"):
                await core.search(
                    query="q",
                    tenant_id="workspace-beta",
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
                    tenant_id="ws",
                    collections=["restricted"],
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_indexer_delete_refuses_cross_tenant_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CollectionPolicy(bound_tenant_id="workspace-alpha")
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CollectionPolicyViolation, match="bound to tenant_id"):
                await core.delete_document(
                    document_id="doc-1",
                    tenant_id="workspace-beta",
                    collection="public",
                )
        finally:
            await core.close()

    asyncio.run(_run())


# --- delete_collection / delete_tenant helpers ------------------------------


def test_delete_collection_helper_routes_through_explicit_seam() -> None:
    async def _run() -> None:
        core, store = _make_core()
        try:
            await core.add_bytes(
                file_bytes=b"alpha",
                filename="a.txt",
                mime_type="text/plain",
                tenant_id="ws",
                collection="c",
                path="/a.txt",
            )
            await core.delete_collection(tenant_id="ws", collection="c")
        finally:
            await core.close()

        # The recording store sees a collection-wide DeleteFilter with no
        # document_id. Exactly what the new explicit helper produces.
        assert any(
            call.collection == "c" and call.document_id is None
            for call in store.delete_calls
        ), f"expected explicit collection-wide delete; got {store.delete_calls!r}"

    asyncio.run(_run())


def test_delete_tenant_is_explicitly_reserved() -> None:
    """The tenant-wide path is not a silent default; it must be claimed."""
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(NotImplementedError, match="tenant offboarding"):
                await core.delete_tenant(tenant_id="ws")
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
                    tenant_id="ws",
                    collection="c",
                )
        finally:
            await core.close()

    asyncio.run(_run())
