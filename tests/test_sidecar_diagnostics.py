from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Sequence, cast

from rag_core.events.sink import EventSink
from rag_core.events.sinks import EventBuffer
from rag_core.events.sinks import JsonlSink
from rag_core.events.types import SidecarApplied
from rag_core.search.pipeline.stages.sidecar_postprocess import SidecarPostprocess
from rag_core.search.pipeline.stages.sidecar_postprocess import SidecarPrefetchTransform
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.filters import Term
from rag_core.search.request_models import SearchSidecarQuery
from rag_core.search.vector_models import SearchResult


def test_sidecar_applied_preserves_positional_constructor_shape() -> None:
    event = SidecarApplied(
        "provider",
        3,
        5,
        2,
        3,
        4,
        12.5,
        False,
        "RuntimeError",
    )

    assert event.provider == "provider"
    assert event.input_count == 3
    assert event.provider_result_count == 5
    assert event.accepted_count == 2
    assert event.dropped_count == 3
    assert event.result_count == 4
    assert event.duration_ms == 12.5
    assert event.succeeded is False
    assert event.fallback_reason == "RuntimeError"
    assert event.event_type == "sidecar.applied"

    event_with_type = SidecarApplied(
        "provider",
        3,
        5,
        2,
        3,
        4,
        12.5,
        False,
        "RuntimeError",
        "sidecar.applied",
    )
    assert event_with_type.event_type == "sidecar.applied"


class _StaticSidecar:
    provider_name = "test-sidecar"

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[SearchSidecarQuery] = []

    def upsert_records(self, records: Sequence[object]) -> None:
        return None

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None:
        return None

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        self.calls.append(query)
        return list(self._results)


class _FailingSidecar(_StaticSidecar):
    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        self.calls.append(query)
        raise RuntimeError("private query text should not escape")


class _CancelledSidecar(_StaticSidecar):
    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        self.calls.append(query)
        raise asyncio.CancelledError()


class _DelayedSidecar(_StaticSidecar):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        self.calls.append(query)
        self.started.set()
        try:
            await asyncio.Event().wait()
            return []
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _CleanupFailingSidecar(_StaticSidecar):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        self.calls.append(query)
        self.started.set()
        try:
            await asyncio.Event().wait()
            return []
        except asyncio.CancelledError as exc:
            raise RuntimeError("private cleanup failure") from exc


def test_sidecar_trace_counts_filtered_and_merged_results() -> None:
    events = EventBuffer()
    sidecar = _StaticSidecar(
        [
            _result(
                "sidecar-ok",
                text="private sidecar text",
                document_id="doc-allowed",
                metadata={"team": "support"},
            ),
            _result(
                "sidecar-wrong-doc",
                document_id="doc-blocked",
                metadata={"team": "support"},
            ),
            _result(
                "sidecar-wrong-team",
                document_id="doc-allowed",
                metadata={"team": "sales"},
            ),
        ]
    )

    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=sidecar,
        event_sink=events,
        prefetch=True,
    )

    assert [hit.id for hit in result] == ["sidecar-ok", "vector"]
    assert sidecar.calls[0].query == "private query text"
    assert applied.provider == "test-sidecar"
    assert applied.input_count == 1
    assert applied.provider_result_count == 3
    assert applied.accepted_count == 1
    assert applied.dropped_count == 2
    assert applied.result_count == 2
    assert applied.succeeded is True
    assert "private query text" not in str(applied)
    assert "private sidecar text" not in str(applied)


def test_sidecar_drops_scoped_results_without_namespace() -> None:
    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=_StaticSidecar(
            [
                _result(
                    "sidecar-no-namespace",
                    namespace=None,
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                )
            ]
        ),
        event_sink=EventBuffer(),
        prefetch=True,
    )

    assert [hit.id for hit in result] == ["vector"]
    assert applied.accepted_count == 0
    assert applied.dropped_count == 1


def test_sidecar_over_return_cannot_crowd_out_vector_hits() -> None:
    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=_StaticSidecar(
            [
                _result(
                    "sidecar-a",
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                ),
                _result(
                    "sidecar-b",
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                ),
                _result(
                    "sidecar-c",
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                ),
            ]
        ),
        event_sink=EventBuffer(),
        prefetch=True,
        query_limit=2,
    )

    assert [hit.id for hit in result] == ["sidecar-a", "vector"]
    assert applied.provider_result_count == 3
    assert applied.accepted_count == 1
    assert applied.dropped_count == 2
    assert applied.result_count == 2


def test_sidecar_trace_counts_empty_provider_success() -> None:
    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=_StaticSidecar([]),
        event_sink=EventBuffer(),
        prefetch=False,
    )

    assert [hit.id for hit in result] == ["vector"]
    assert applied.provider_result_count == 0
    assert applied.accepted_count == 0
    assert applied.dropped_count == 0
    assert applied.result_count == 1
    assert applied.succeeded is True


def test_sidecar_trace_counts_all_provider_results_dropped_by_filters() -> None:
    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=_StaticSidecar(
            [
                _result(
                    "sidecar-wrong-doc",
                    text="private wrong document",
                    document_id="doc-blocked",
                    metadata={"team": "support"},
                ),
                _result(
                    "sidecar-wrong-team",
                    text="private wrong team",
                    document_id="doc-allowed",
                    metadata={"team": "sales"},
                ),
            ]
        ),
        event_sink=EventBuffer(),
        prefetch=True,
    )

    assert [hit.id for hit in result] == ["vector"]
    assert applied.provider_result_count == 2
    assert applied.accepted_count == 0
    assert applied.dropped_count == 2
    assert applied.result_count == 1
    assert applied.succeeded is True
    assert "private query text" not in str(applied)
    assert "private wrong document" not in str(applied)
    assert "private wrong team" not in str(applied)


def test_sidecar_prefetch_is_cancelled_when_query_opts_out_after_transform() -> None:
    async def run() -> None:
        sidecar = _DelayedSidecar()
        query = _query()
        ctx = _context(sidecar=sidecar, event_sink=EventBuffer())
        vector_results = [_result("vector", document_id="doc-vector")]

        query = await SidecarPrefetchTransform().transform(query, ctx)
        task = _prefetched_task(query)
        await asyncio.wait_for(sidecar.started.wait(), timeout=1)
        query.use_lexical_search = False

        result = await SidecarPostprocess().postprocess(vector_results, query, ctx)

        assert [hit.id for hit in result] == ["vector"]
        assert query.state.sidecar_prefetch is None
        assert task.done()
        assert task.cancelled()
        assert sidecar.cancelled.is_set()

    asyncio.run(run())


def test_sidecar_prefetch_replacement_cancels_previous_task() -> None:
    async def run() -> None:
        stale_sidecar = _DelayedSidecar()
        replacement_sidecar = _StaticSidecar([])
        query = _query()
        ctx = _context(sidecar=stale_sidecar, event_sink=EventBuffer())

        query = await SidecarPrefetchTransform().transform(query, ctx)
        stale_task = _prefetched_task(query)
        await asyncio.wait_for(stale_sidecar.started.wait(), timeout=1)

        query = await SidecarPrefetchTransform().transform(
            query,
            _context(sidecar=replacement_sidecar, event_sink=EventBuffer()),
        )

        assert stale_task.done()
        assert stale_task.cancelled()
        assert stale_sidecar.cancelled.is_set()
        assert _prefetched_task(query) is not stale_task

        await SidecarPostprocess().postprocess(
            [_result("vector", document_id="doc-vector")],
            query,
            _context(sidecar=replacement_sidecar, event_sink=EventBuffer()),
        )

    asyncio.run(run())


def test_sidecar_prefetch_cancel_logs_cleanup_failure_without_private_text(
    caplog: Any,
) -> None:
    async def run() -> None:
        sidecar = _CleanupFailingSidecar()
        query = _query()
        ctx = _context(sidecar=sidecar, event_sink=EventBuffer())
        vector_results = [_result("vector", document_id="doc-vector")]

        query = await SidecarPrefetchTransform().transform(query, ctx)
        task = _prefetched_task(query)
        await asyncio.wait_for(sidecar.started.wait(), timeout=1)
        query.use_lexical_search = False

        result = await SidecarPostprocess().postprocess(vector_results, query, ctx)

        assert [hit.id for hit in result] == ["vector"]
        assert task.done()
        assert not task.cancelled()
        assert isinstance(task.exception(), RuntimeError)

    caplog.set_level(
        logging.WARNING,
        logger="rag_core.search.pipeline.stages.sidecar_postprocess",
    )
    asyncio.run(run())
    assert "RuntimeError" in caplog.text
    assert "private cleanup failure" not in caplog.text


def test_sidecar_trace_serializes_without_query_or_result_text(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    result, _ = _run_sidecar(
        [_result("vector", text="private vector text", document_id="doc-vector")],
        sidecar=_StaticSidecar(
            [
                _result(
                    "sidecar-ok",
                    text="private sidecar text",
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                )
            ]
        ),
        event_sink=JsonlSink(path),
        prefetch=True,
    )

    assert [hit.id for hit in result] == ["sidecar-ok", "vector"]
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "sidecar.applied"
    assert payload["provider_result_count"] == 1
    assert payload["accepted_count"] == 1
    serialized = json.dumps(payload)
    assert "private query text" not in serialized
    assert "private sidecar text" not in serialized
    assert "private vector text" not in serialized


def test_sidecar_trace_records_failure_without_exception_message(
    caplog: Any,
) -> None:
    events = EventBuffer()
    caplog.set_level(
        logging.WARNING,
        logger="rag_core.search.pipeline.stages.sidecar_postprocess",
    )
    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=_FailingSidecar([]),
        event_sink=events,
        prefetch=True,
    )

    assert [hit.id for hit in result] == ["vector"]
    assert applied.provider == "test-sidecar"
    assert applied.provider_result_count == 0
    assert applied.accepted_count == 0
    assert applied.dropped_count == 0
    assert applied.result_count == 1
    assert applied.succeeded is False
    assert applied.fallback_reason == "RuntimeError"
    assert "private query text" not in str(applied)
    assert "private query text" not in caplog.text


def test_sidecar_cancelled_error_falls_back_to_vector_results() -> None:
    events = EventBuffer()
    result, applied = _run_sidecar(
        [_result("vector", document_id="doc-vector")],
        sidecar=_CancelledSidecar([]),
        event_sink=events,
        prefetch=True,
    )

    assert [hit.id for hit in result] == ["vector"]
    assert applied.succeeded is False
    assert applied.fallback_reason == "CancelledError"


def _run_sidecar(
    vector_results: list[SearchResult],
    *,
    sidecar: _StaticSidecar,
    event_sink: EventSink,
    prefetch: bool,
    query_limit: int = 20,
) -> tuple[list[SearchResult], SidecarApplied]:
    async def run() -> tuple[list[SearchResult], SidecarApplied]:
        query = _query(limit=query_limit)
        ctx = _context(sidecar=sidecar, event_sink=event_sink)
        if prefetch:
            query = await SidecarPrefetchTransform().transform(query, ctx)
        results = await SidecarPostprocess().postprocess(vector_results, query, ctx)
        if isinstance(event_sink, EventBuffer):
            applied = [
                event for event in event_sink.events if isinstance(event, SidecarApplied)
            ]
            assert len(applied) == 1
            return results, applied[0]
        return results, SidecarApplied()

    return asyncio.run(run())


def _prefetched_task(query: PipelineQuery) -> asyncio.Task[list[SearchResult]]:
    prefetched = query.state.sidecar_prefetch
    assert prefetched is not None
    return prefetched.task


def _query(*, limit: int = 20) -> PipelineQuery:
    return PipelineQuery(
        query="private query text",
        namespace="ns",
        corpus_ids=["corpus"],
        limit=limit,
        content_types=["document"],
        document_ids=["doc-allowed"],
        metadata_filter=Term(field="team", value="support"),
    )


def _context(
    *,
    sidecar: _StaticSidecar,
    event_sink: EventSink,
) -> PipelineContext:
    return PipelineContext(
        embedding_provider=cast(Any, object()),
        sparse_embedder=cast(Any, object()),
        vector_store=cast(Any, object()),
        sidecar=sidecar,
        event_sink=event_sink,
    )


def _result(
    result_id: str,
    *,
    text: str = "result text",
    namespace: str | None = "ns",
    corpus_id: str = "corpus",
    document_id: str,
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    return SearchResult(
        id=result_id,
        text=text,
        score=1.0,
        content_type="document",
        source_type="file",
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        metadata=dict(metadata or {}),
    )
