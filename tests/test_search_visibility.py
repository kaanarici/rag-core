"""Uncertain revisions fail explicitly within a bounded amount of query work."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from rag_core import SearchVisibilityError
from rag_core.search.document_versions import Candidate, committed_search
from rag_core.search.policy import DEFAULT_POLICY
from rag_core.search.request_models import SearchQuery
from tests.support import make_search_result


@pytest.mark.parametrize("progress,expected_reads", [(True, 8), (False, 2)])
def test_unpublished_versions_have_a_distinguishable_bounded_failure(
    progress: bool, expected_reads: int,
) -> None:
    async def run() -> None:
        calls = 0
        head_reads = 0

        async def search(query: SearchQuery) -> list[Candidate]:
            nonlocal calls
            calls += 1
            revision = f"{calls if progress else 1:032x}"
            return [(make_search_result(
                text="unfinished", tenant_id="t", collection="c", document_id="d",
            ), revision)]

        async def heads(ids: Sequence[str]) -> dict[str, dict[str, object]]:
            nonlocal head_reads
            head_reads += 1
            return {}

        with pytest.raises(SearchVisibilityError) as failure:
            await committed_search(
                SearchQuery(dense_vector=[1.0], tenant_id="t", collections=["c"], limit=1),
                search, heads, DEFAULT_POLICY,
            )
        assert failure.value.attempts == calls == head_reads == expected_reads
        assert failure.value.excluded_versions == (8 if progress else 1)

    asyncio.run(run())
