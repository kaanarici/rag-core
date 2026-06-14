"""Prove citation locators survive the real ingest→Qdrant→context retrieval path.

These tests use real local Qdrant (``:memory:``) and the demo embedder. The
claim under test is locator/citation INTEGRITY across the full round trip:
chunk spans → Qdrant payload → search-result conversion → context packing, not
ranking quality. Queries use vocabulary unique to the target section so the
deterministic demo embeddings still retrieve it; ground truth (section titles,
line numbers, char spans) is computed from the test-constructed source text, not
copied from observed output.
"""

from __future__ import annotations

import uuid

import pytest

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.demo import build_demo_core
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.documents.contextualizer import ChunkContextRequest
from rag_core.search.pipeline import (
    HybridRetrieve,
    IdentityFuse,
    PassThroughRerank,
    RetrievalPipeline,
)
from rag_core.search.pipeline.stages.neighbor_expand import NeighborExpandPostprocess
from rag_core.search.pipeline_runner import SearchPipelineRunner

pytestmark = [pytest.mark.integration]


# Markdown with controlled section titles and body sentences carrying
# vocabulary unique to each section so the demo embedder can target them.
_GUIDE_MD = """# Onboarding Handbook

## Reimbursement Workflow

Employees submit gnarlwick expense forms through the portal each fortnight.
Approved reimbursements settle within five business days.

## Hardware Procurement

Request a flibberjet laptop replacement by filing a procurement ticket.
Standard issue machines ship preconfigured with the security baseline.
"""

# Python source large enough that the code chunker emits multiple code chunks
# (>2000-char budget), proving per-chunk line locators rather than a single
# prechunked blob. The target function `quibblesplat_average` is placed first so
# its unique vocabulary lands in chunk 0. The chunk the code chunker can locate
# verbatim. Functions are single-blank-line separated to keep that chunk a
# byte-true slice of the source.
def _make_tool_py() -> str:
    def fn(name: str, token: str, steps: int) -> str:
        body = "\n".join(
            f"    # {token} note {i}: keeps this function within the code-chunk budget"
            for i in range(steps)
        )
        return (
            f"def {name}(amounts):\n"
            f"{body}\n"
            f"    if not amounts:\n"
            f"        return 0\n"
            f"    return sum(amounts) / len(amounts)\n"
        )

    return "\n".join(
        [
            fn("quibblesplat_average", "quibblesplat", 20),
            fn("zorptangle_total", "zorptangle", 20),
            fn("flibberjet_scale", "flibberjet", 20),
        ]
    )


_TOOL_PY = _make_tool_py()

_NEIGHBOR_GUIDE_MD = """# Neighbor Handbook

## Alpha Preparation

The prelude orbit sentence explains setup before the target policy.

## Beta Target

The amberglint retrieval sentence names the exact mid-document policy.

## Gamma Followup

The followup lattice sentence explains what happens after the target policy.
"""


class _ContextOnlyContextualizer:
    contextualizer_id = "test:qdrant-clean-citation-context"

    async def contextualize(self, request: ChunkContextRequest) -> str:
        return "zephyrcleanmarker context only"


async def _ingest(core: object, *, filename: str, mime_type: str, body: str) -> object:
    return await core.ingest_bytes(  # type: ignore[attr-defined]
        file_bytes=body.encode("utf-8"),
        filename=filename,
        mime_type=mime_type,
        namespace="citations",
        corpus_id="docs",
        document_key=filename,
    )


async def _ingest_fixture(core: object) -> tuple[int, int]:
    guide = await _ingest(
        core, filename="guide.md", mime_type="text/markdown", body=_GUIDE_MD
    )
    tool = await _ingest(
        core, filename="tool.py", mime_type="text/x-python", body=_TOOL_PY
    )
    return guide.chunk_count, tool.chunk_count  # type: ignore[attr-defined]


def _assert_span_sanity(snippet: object, original: str) -> None:
    """Every returned span must resolve to a real slice of the source text."""
    locator = snippet.locator  # type: ignore[attr-defined]
    start = locator.start_offset
    end = locator.end_offset
    if start is None or end is None:
        return
    assert 0 <= start < end <= len(original), (
        f"span out of bounds: start={start} end={end} len={len(original)}"
    )
    snippet_text = snippet.text  # type: ignore[attr-defined]
    assert snippet_text in original[start:end] or original[start:end] in snippet_text, (
        f"chunk text not contained in its own span:\n"
        f"  span slice={original[start:end]!r}\n  snippet={snippet_text!r}"
    )


def test_citations_survive_real_round_trip() -> None:
    import asyncio

    async def go() -> None:
        async with build_demo_core(
            collection=f"citations_{uuid.uuid4().hex}"
        ) as core:
            guide_chunks, tool_chunks = await _ingest_fixture(core)
            assert guide_chunks > 0
            assert tool_chunks > 0

            pack = await core.retrieve_context(
                query="gnarlwick reimbursement fortnight expense forms",
                namespace="citations",
                corpus_ids=["docs"],
                limit=5,
                rerank=False,
            )

            assert pack.snippets, "no snippets retrieved for the reimbursement query"

            # Citation ids are sequential S1..Sn in the prompt projection and each
            # maps to a real ingested filename in the app-facing projection.
            prompt_snippets = pack.to_prompt_payload()["snippets"]
            assert isinstance(prompt_snippets, list)
            prompt_ids = [s["citation_id"] for s in prompt_snippets]
            assert prompt_ids == [f"S{i + 1}" for i in range(len(prompt_snippets))]

            ingested_keys = {"guide.md", "tool.py"}
            for source in pack.citations:
                assert source.document_key in ingested_keys

            prompt_text = pack.as_prompt_text()
            for index, snippet in enumerate(pack.snippets):
                tag = f"[S{index + 1}]"
                assert prompt_text.count(tag) == 1, (
                    f"{tag} should appear exactly once per source block"
                )
            summary = pack.prompt_citation_summary
            for index in range(len(pack.snippets)):
                assert f"[S{index + 1}]" in summary

            # The reimbursement section is the deterministic top hit; verify the
            # markdown locator carries the right section title and chunk index,
            # and that previews/text are byte-true substrings of the source.
            guide_snippets = [
                s for s in pack.snippets if s.source.document_key == "guide.md"
            ]
            assert guide_snippets, "reimbursement query did not retrieve guide.md"
            reimbursement = next(
                (
                    s
                    for s in guide_snippets
                    if "gnarlwick" in s.text
                ),
                None,
            )
            assert reimbursement is not None, (
                "unique reimbursement vocabulary did not land in any guide chunk"
            )
            assert reimbursement.source.section_path is not None
            assert (
                "Reimbursement Workflow"
                in reimbursement.source.section_path
            )
            assert reimbursement.source.chunk_index is not None
            assert reimbursement.text in _GUIDE_MD

            for snippet in pack.snippets:
                original = _GUIDE_MD if snippet.source.document_key == "guide.md" else _TOOL_PY
                assert snippet.text in original
                _assert_span_sanity(snippet, original)

    asyncio.run(go())


def test_qdrant_contextualized_retrieval_keeps_clean_citation_text() -> None:
    import asyncio

    marker = "zephyrcleanmarker"
    clean_phrase = "The clean reimbursement paragraph names baseline operations."
    clean_document = f"# Clean Policy\n\n{clean_phrase}"

    async def go() -> None:
        core = RAGCore(
            RAGCoreConfig(
                qdrant=QdrantConfig(
                    location=":memory:",
                    collection=f"clean_context_{uuid.uuid4().hex}",
                    dimension_aware_collection=False,
                ),
                embedding=EmbeddingConfig(
                    provider="demo",
                    model="demo-dense-v1",
                    dimensions=64,
                ),
            ),
            embedding_provider=DemoEmbeddingProvider(),
            sparse_embedder=DemoSparseEmbedder(),
            chunk_contextualizer=_ContextOnlyContextualizer(),
        )
        try:
            await core.ingest_bytes(
                file_bytes=clean_document.encode("utf-8"),
                filename="clean.md",
                mime_type="text/markdown",
                namespace="citations",
                corpus_id="docs",
                document_key="clean.md",
            )

            hits = await core.search(
                query=marker,
                namespace="citations",
                corpus_ids=["docs"],
                limit=3,
                rerank=False,
            )
            pack = await core.retrieve_context(
                query=marker,
                namespace="citations",
                corpus_ids=["docs"],
                limit=3,
                rerank=False,
            )
        finally:
            await core.close()

        assert hits, "context-only query did not retrieve the Qdrant document"
        assert pack.snippets, "context-only query did not build Qdrant context"
        assert clean_phrase in hits[0].text
        assert marker not in hits[0].text
        assert clean_phrase in pack.snippets[0].text
        assert marker not in pack.snippets[0].text
        assert marker not in pack.as_prompt_text()
        for preview in pack.source_previews:
            assert marker not in preview.as_text()
            assert marker not in repr(preview.to_payload())
        for citation in pack.citations:
            assert marker not in repr(citation.to_payload())

    asyncio.run(go())


def test_retrieve_context_extrema_order_keeps_real_round_trip_citation_spans() -> None:
    import asyncio

    async def go() -> None:
        async with build_demo_core(
            collection=f"citations_extrema_{uuid.uuid4().hex}"
        ) as core:
            await _ingest_fixture(core)

            pack = await core.retrieve_context(
                query="gnarlwick reimbursement fortnight expense forms",
                namespace="citations",
                corpus_ids=["docs"],
                limit=5,
                rerank=False,
            )

            assert len(pack.snippets) >= 3
            assert [snippet.rank for snippet in pack.snippets] == list(
                range(1, len(pack.snippets) + 1)
            )
            prompt_text = pack.as_prompt_text(context_order="extrema")
            assert prompt_text.startswith("[S1]")
            assert prompt_text.rsplit("\n\n", 1)[-1].startswith("[S2]")
            for snippet in pack.snippets:
                original = _GUIDE_MD if snippet.source.document_key == "guide.md" else _TOOL_PY
                assert snippet.text in original
                _assert_span_sanity(snippet, original)

    asyncio.run(go())


def test_code_chunk_locators_match_source_lines() -> None:
    import asyncio

    async def go() -> None:
        async with build_demo_core(
            collection=f"citations_{uuid.uuid4().hex}"
        ) as core:
            await _ingest_fixture(core)

            pack = await core.retrieve_context(
                query="quibblesplat average amounts",
                namespace="citations",
                corpus_ids=["docs"],
                limit=5,
                rerank=False,
            )

            code_snippets = [
                s for s in pack.snippets if s.source.document_key == "tool.py"
            ]
            assert code_snippets, "code query did not retrieve tool.py"
            target = next(
                (s for s in code_snippets if "quibblesplat_average" in s.text), None
            )
            assert target is not None, (
                "unique function vocabulary did not land in any code chunk"
            )

            locator = target.locator
            assert locator.line_start is not None and locator.line_end is not None, (
                "code chunk exposes no line locator fields"
            )

            source_lines = _TOOL_PY.splitlines()
            # 1-based, inclusive line range sliced from the original source.
            located = "\n".join(
                source_lines[locator.line_start - 1 : locator.line_end]
            )
            assert "def quibblesplat_average" in located, (
                f"located lines {locator.line_start}-{locator.line_end} do not "
                f"contain the target function:\n{located!r}"
            )

            _assert_span_sanity(target, _TOOL_PY)

    asyncio.run(go())


def test_neighbor_expansion_keeps_original_citation_span() -> None:
    import asyncio

    async def go() -> None:
        async with build_demo_core(
            collection=f"neighbor_{uuid.uuid4().hex}"
        ) as core:
            core._search = SearchPipelineRunner(
                embedding_provider=core._embedding,
                sparse_embedder=core._sparse,
                vector_store=core._store,
                event_sink=core._event_sink,
                pipeline=RetrievalPipeline(
                    retrieve=HybridRetrieve(),
                    fuse=IdentityFuse(),
                    rerank=PassThroughRerank(),
                    postprocesses=(NeighborExpandPostprocess(window=1),),
                ),
            )
            ingested = await _ingest(
                core,
                filename="neighbor.md",
                mime_type="text/markdown",
                body=_NEIGHBOR_GUIDE_MD,
            )
            assert getattr(ingested, "chunk_count") >= 4

            pack = await core.retrieve_context(
                query="amberglint retrieval sentence",
                namespace="citations",
                corpus_ids=["docs"],
                limit=1,
                rerank=False,
            )

            assert len(pack.snippets) == 1
            snippet = pack.snippets[0]
            assert "prelude orbit sentence" in snippet.text
            assert "amberglint retrieval sentence" in snippet.text
            assert "followup lattice sentence" in snippet.text
            assert snippet.locator.start_offset is not None
            assert snippet.locator.end_offset is not None
            original_slice = _NEIGHBOR_GUIDE_MD[
                snippet.locator.start_offset : snippet.locator.end_offset
            ]
            assert "amberglint retrieval sentence" in original_slice
            assert "prelude orbit sentence" not in original_slice
            assert "followup lattice sentence" not in original_slice
            assert pack.source_previews[0].locator_label is not None
            assert pack.source_previews[0].locator_label.endswith(
                f"chunk {snippet.source.chunk_index}"
            )

    asyncio.run(go())
