"""Tier-widening fail-closed seam tests for ``DeleteFilter`` and ``CorpusPolicy``.

Covers four guarantees for tiered deployments that run one ``RAGCore`` per
workspace/corpus tier:

1. ``DeleteFilter.__post_init__`` rejects empty / blank ``document_id`` or
   ``corpus_id`` strings so a formatting bug cannot silently widen a
   per-document delete to a corpus-wide or namespace-wide one.
2. The retrieval facade (``RAGCore.search`` / ``RAGCore.retrieve_context``)
   refuses ``corpus_ids=None`` and ``corpus_ids=[]`` because silent widening is
   forbidden.
3. ``CorpusPolicy`` validates namespace binding, allowed corpus ids, rerank
   and lexical-sidecar capability flags from the engine seam before any
   provider call.
4. Explicit ``delete_corpus`` and ``delete_namespace`` facade helpers exist
   so callers reach corpus-wide or namespace-wide deletes deliberately, not
   by accident.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from rag_core import RAGCore
from rag_core.search.policy import CorpusPolicy, CorpusPolicyViolation
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
        DeleteFilter(namespace="ns", corpus_id="c", document_id="")


def test_delete_filter_rejects_whitespace_document_id() -> None:
    with pytest.raises(ValueError, match="DeleteFilter.document_id"):
        DeleteFilter(namespace="ns", corpus_id="c", document_id="   ")


def test_delete_filter_rejects_empty_corpus_id() -> None:
    with pytest.raises(ValueError, match="DeleteFilter.corpus_id"):
        DeleteFilter(namespace="ns", corpus_id="", document_id="doc-1")


def test_delete_filter_requires_corpus_or_document() -> None:
    """Bare ``DeleteFilter(namespace=...)`` would clear the namespace."""
    with pytest.raises(ValueError, match="at least one of corpus_id or document_id"):
        DeleteFilter(namespace="ns")


def test_delete_filter_accepts_corpus_only_for_explicit_corpus_wide_delete() -> None:
    """The deliberate corpus-wide path stays open via explicit None."""
    f = DeleteFilter(namespace="ns", corpus_id="c")
    assert f.document_id is None
    assert f.corpus_id == "c"


def test_delete_filter_accepts_document_with_corpus_for_per_doc_delete() -> None:
    f = DeleteFilter(namespace="ns", corpus_id="c", document_id="doc-1")
    assert f.document_id == "doc-1"


# --- CorpusPolicy enforcement ----------------------------------------------


def test_corpus_policy_validates_bound_namespace() -> None:
    policy = CorpusPolicy(bound_namespace="workspace-alpha")
    policy.validate_namespace("workspace-alpha")  # ok
    with pytest.raises(CorpusPolicyViolation, match="bound to namespace"):
        policy.validate_namespace("workspace-beta")


def test_corpus_policy_validates_allowed_corpus_ids() -> None:
    policy = CorpusPolicy(allowed_corpus_ids=frozenset({"public", "licensed"}))
    policy.validate_corpus_ids(["public"])  # ok
    policy.validate_corpus_ids(["licensed", "public"])  # ok
    with pytest.raises(CorpusPolicyViolation, match="refused corpus_id"):
        policy.validate_corpus_ids(["restricted"])


def test_corpus_policy_refuses_none_corpus_ids_when_allowlist_set() -> None:
    """Per slice: ``corpus_ids=None`` against a bound allowlist fails closed."""
    policy = CorpusPolicy(allowed_corpus_ids=frozenset({"public"}))
    with pytest.raises(CorpusPolicyViolation, match="silently widens"):
        policy.validate_corpus_ids(None)


def test_corpus_policy_disallows_rerank_on_restricted_tier() -> None:
    policy = CorpusPolicy(allow_rerank=False)
    with pytest.raises(CorpusPolicyViolation, match="rerank"):
        policy.validate_search(
            namespace="ws",
            corpus_ids=["restricted"],
            rerank=True,
            use_lexical_search=False,
        )


def test_corpus_policy_disallows_lexical_sidecar_on_restricted_tier() -> None:
    policy = CorpusPolicy(allow_lexical_sidecar=False)
    with pytest.raises(CorpusPolicyViolation, match="lexical sidecar"):
        policy.validate_search(
            namespace="ws",
            corpus_ids=["restricted"],
            rerank=False,
            use_lexical_search=True,
        )


def test_corpus_policy_validates_allowed_query_plan_presets() -> None:
    policy = CorpusPolicy(allowed_query_plan_presets=frozenset({"dense_only"}))
    policy.validate_search(
        namespace="ws",
        corpus_ids=["c"],
        rerank=False,
        use_lexical_search=False,
        query_plan_preset="dense_only",
    )
    with pytest.raises(CorpusPolicyViolation, match="query_plan_preset"):
        policy.validate_search(
            namespace="ws",
            corpus_ids=["c"],
            rerank=False,
            use_lexical_search=False,
            query_plan_preset="hybrid_full",
        )


def test_corpus_policy_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="bound_namespace"):
        CorpusPolicy(bound_namespace="   ")
    with pytest.raises(ValueError, match="allowed_corpus_ids"):
        CorpusPolicy(allowed_corpus_ids=frozenset({""}))


# --- Facade-level fail-closed corpus_ids -----------------------------------


def _make_core() -> tuple[RAGCore, RecordingVectorStore]:
    store = RecordingVectorStore()
    config = make_test_config(
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    core = RAGCore(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    return core, store


def _make_core_with_policy(policy: CorpusPolicy) -> tuple[RAGCore, RecordingVectorStore]:
    store = RecordingVectorStore()
    base = make_test_config(
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    core = RAGCore(
        replace(base, corpus_policy=policy),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    return core, store


def test_search_facade_refuses_none_corpus_ids() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="explicit corpus_ids"):
                await core.search(
                    query="q",
                    namespace="ns",
                    corpus_ids=None,  # type: ignore[arg-type]
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_search_facade_refuses_empty_corpus_ids() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="non-empty corpus_ids"):
                await core.search(query="q", namespace="ns", corpus_ids=[])
        finally:
            await core.close()

    asyncio.run(_run())


def test_retrieve_context_facade_refuses_none_corpus_ids() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="explicit corpus_ids"):
                await core.retrieve_context(
                    query="q",
                    namespace="ns",
                    corpus_ids=None,  # type: ignore[arg-type]
                )
        finally:
            await core.close()

    asyncio.run(_run())


# --- CorpusPolicy wired through the engine ---------------------------------


def test_pipeline_runner_refuses_cross_namespace_request_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CorpusPolicy(bound_namespace="workspace-alpha")
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CorpusPolicyViolation, match="bound to namespace"):
                await core.search(
                    query="q",
                    namespace="workspace-beta",
                    corpus_ids=["public"],
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_pipeline_runner_refuses_disallowed_corpus_id_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CorpusPolicy(allowed_corpus_ids=frozenset({"public"}))
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CorpusPolicyViolation, match="refused corpus_id"):
                await core.search(
                    query="q",
                    namespace="ws",
                    corpus_ids=["restricted"],
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_pipeline_runner_refuses_rerank_when_policy_forbids_it() -> None:
    async def _run() -> None:
        policy = CorpusPolicy(allow_rerank=False)
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CorpusPolicyViolation, match="rerank"):
                await core.search(
                    query="q",
                    namespace="ws",
                    corpus_ids=["restricted"],
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
        policy = CorpusPolicy(allowed_query_plan_presets=frozenset({"fast"}))
        runner = SearchPipelineRunner(
            FakeEmbeddingProvider(),
            FakeSparseEmbedder(),
            RecordingVectorStore(),
            corpus_policy=policy,
        )
        plan = search_profile("balanced", limit=5)
        with pytest.raises(CorpusPolicyViolation, match="query_plan_preset"):
            await runner.search(
                SearchRequest(
                    query="q",
                    corpus_ids=["c"],
                    namespace="ws",
                    execution=SearchExecutionOptions(query_plan=plan),
                )
            )

    asyncio.run(_run())


def test_indexer_delete_refuses_cross_namespace_under_bound_policy() -> None:
    async def _run() -> None:
        policy = CorpusPolicy(bound_namespace="workspace-alpha")
        core, _ = _make_core_with_policy(policy)
        try:
            with pytest.raises(CorpusPolicyViolation, match="bound to namespace"):
                await core.delete_document(
                    document_id="doc-1",
                    namespace="workspace-beta",
                    corpus_id="public",
                )
        finally:
            await core.close()

    asyncio.run(_run())


# --- delete_corpus / delete_namespace helpers ------------------------------


def test_delete_corpus_helper_routes_through_explicit_seam() -> None:
    async def _run() -> None:
        core, store = _make_core()
        try:
            await core.ingest_bytes(
                file_bytes=b"alpha",
                filename="a.txt",
                mime_type="text/plain",
                namespace="ws",
                corpus_id="c",
                path="/a.txt",
            )
            await core.delete_corpus(namespace="ws", corpus_id="c")
        finally:
            await core.close()

        # The recording store sees a corpus-wide DeleteFilter with no
        # document_id. Exactly what the new explicit helper produces.
        assert any(
            call.corpus_id == "c" and call.document_id is None
            for call in store.delete_calls
        ), f"expected explicit corpus-wide delete; got {store.delete_calls!r}"

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
                    corpus_id="c",
                )
        finally:
            await core.close()

    asyncio.run(_run())
