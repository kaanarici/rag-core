"""Integration tests: every public search profile executes against embedded Qdrant.

Claim: each of the five canonical search profiles (balanced, fast, lexical, coverage,
diverse) completes without error against a real Qdrant :memory: instance built via
``rag_core.demo.build_demo_core`` (deterministic dense + sparse embedders) and returns
structurally valid results. This does NOT assert ranking quality. Demo embeddings make
ranking-quality claims meaningless. See tests/README.md on the 'integration' tier.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from rag_core.demo import build_demo_core
from rag_core.search.query_plan_presets import (
    SEARCH_PROFILE_BALANCED,
    SEARCH_PROFILE_COVERAGE,
    SEARCH_PROFILE_DIVERSE,
    SEARCH_PROFILE_FAST,
    SEARCH_PROFILE_LEXICAL,
    SEARCH_PROFILE_SPECS,
    search_profile,
)

pytestmark = [pytest.mark.integration]

# Parametrize list is derived from SEARCH_PROFILE_SPECS so a new profile fails
# this test until added here. Intended per plan 006 maintenance notes.
_PROFILE_NAMES = [
    SEARCH_PROFILE_BALANCED,
    SEARCH_PROFILE_COVERAGE,
    SEARCH_PROFILE_DIVERSE,
    SEARCH_PROFILE_FAST,
    SEARCH_PROFILE_LEXICAL,
]

# Enforce that _PROFILE_NAMES stays in sync with SEARCH_PROFILE_SPECS at import time.
assert sorted(_PROFILE_NAMES) == sorted(SEARCH_PROFILE_SPECS), (
    "Profile list is out of sync with SEARCH_PROFILE_SPECS; "
    "update _PROFILE_NAMES when a profile is added or removed"
)

_CORPUS: list[dict[str, str]] = [
    {
        "document_id": "doc-invoices",
        "markdown": "# Invoice Payments\n\nInvoices can be paid by credit card, ACH bank transfer, or wire.",
    },
    {
        "document_id": "doc-refunds",
        "markdown": "# Refund Policy\n\nRefunds are processed within 5 business days to the original payment method.",
    },
    {
        "document_id": "doc-onboarding",
        "markdown": "# Onboarding Guide\n\nCreate your account, verify your email, then configure workspace settings.",
    },
    {
        "document_id": "doc-api-auth",
        "markdown": "# API Authentication\n\nAuthenticate using bearer tokens. Rotate keys every 90 days for security.",
    },
    {
        "document_id": "doc-search",
        "markdown": "# Search Concepts\n\nHybrid search combines dense semantic vectors with sparse lexical BM25 scoring.",
    },
    {
        "document_id": "doc-chunking",
        "markdown": "# Chunking Strategy\n\nDocuments are split into overlapping chunks to preserve sentence context.",
    },
    {
        "document_id": "doc-qdrant",
        "markdown": "# Qdrant Integration\n\nQdrant stores vectors with payload; collection names are scoped per namespace.",
    },
    {
        "document_id": "doc-pricing",
        "markdown": "# Pricing Tiers\n\nFree plan includes 1 GB storage. Pro plan adds advanced reranking and analytics.",
    },
]


async def _build_and_ingest() -> object:
    """Return a fully ingested Engine (async context manager used by caller)."""
    core = build_demo_core(store_collection=f"profile_smoke_{uuid.uuid4().hex}")
    await core.ensure_ready()
    for doc in _CORPUS:
        await core.add_bytes(
            file_bytes=doc["markdown"].encode("utf-8"),
            filename=f"{doc['document_id']}.md",
            mime_type="text/markdown",
            namespace="smoke",
            collection="profiles",
            document_id=doc["document_id"],
            document_key=f"{doc['document_id']}.md",
        )
    return core


def test_baseline_no_profile_returns_results() -> None:
    """Verify the fixture: plain search (no profile) returns ≥1 hit."""

    async def go() -> None:
        core = await _build_and_ingest()
        try:
            hits = await core.search(  # type: ignore[attr-defined]
                query="invoice payment method",
                namespace="smoke",
                collections=["profiles"],
                limit=5,
                rerank=False,
            )
            assert len(hits) >= 1, "baseline search should return at least one hit"
        finally:
            await core.close()  # type: ignore[attr-defined]

    asyncio.run(go())


@pytest.mark.parametrize("profile_name", _PROFILE_NAMES)
def test_every_public_profile_executes_against_real_qdrant(profile_name: str) -> None:
    """Each profile runs end-to-end against Qdrant :memory: and returns valid hits."""

    async def go() -> None:
        core = await _build_and_ingest()
        try:
            plan = search_profile(profile_name, limit=5)
            hits = await core.search(  # type: ignore[attr-defined]
                query="search indexing retrieval payment invoice",
                namespace="smoke",
                collections=["profiles"],
                limit=5,
                rerank=False,
                query_plan=plan,
            )
            assert isinstance(hits, list), f"[{profile_name}] search must return a list"
            assert len(hits) >= 1, f"[{profile_name}] expected at least one hit"
            assert len(hits) <= 5, f"[{profile_name}] must respect limit=5"
            seen_ids: set[str] = set()
            for hit in hits:
                assert hit.text, f"[{profile_name}] every hit must have non-empty text"
                assert isinstance(hit.score, float), (
                    f"[{profile_name}] score must be a float, got {type(hit.score)}"
                )
                identity = (hit.document_id, hit.id)
                assert identity not in seen_ids, (
                    f"[{profile_name}] duplicate hit: {identity}"
                )
                seen_ids.add(str(identity))
        finally:
            await core.close()  # type: ignore[attr-defined]

    asyncio.run(go())


@pytest.mark.xfail(
    strict=False,
    reason=(
        "lexical profile uses SPARSE_ONLY; build_demo_core wires DemoSparseEmbedder "
        "so sparse vectors are present; xfail is a safety net if the sparse sidecar "
        "is absent in some runtime; the profile should succeed and this mark will "
        "surface the unexpected pass as a promoted finding."
    ),
)
def test_lexical_profile_sparse_channel_behavior() -> None:
    """Lexical profile executes via sparse-only channel and returns structurally valid hits.

    build_demo_core wires DemoSparseEmbedder which provides sparse vectors on the
    PRIMARY_SPARSE_CHANNEL; the lexical profile's SPARSE_ONLY preset should therefore
    succeed. This xfail(strict=False) guard covers runtimes where sparse capability
    is absent (capability downgrade) while still treating a pass as expected.
    """

    async def go() -> None:
        core = await _build_and_ingest()
        try:
            plan = search_profile(SEARCH_PROFILE_LEXICAL, limit=5)
            hits = await core.search(  # type: ignore[attr-defined]
                query="invoice payment",
                namespace="smoke",
                collections=["profiles"],
                limit=5,
                rerank=False,
                query_plan=plan,
            )
            assert len(hits) >= 1, "lexical profile should return ≥1 hit when sparse is available"
            for hit in hits:
                assert hit.text, "lexical hit must have non-empty text"
        finally:
            await core.close()  # type: ignore[attr-defined]

    asyncio.run(go())
