"""Public configuration and lifecycle regressions; fake retrieval is plumbing."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from rag_core import Config
from rag_core import cli
from rag_core.cli.config import config_from_args
from rag_core.cli.parser import _build_parser
from rag_core.config.env_config import build_config_from_env
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.core import Engine
from rag_core.events import (
    AuditContext,
    EventBuffer,
    SearchStarted,
    SearchCompleted,
    summarize_search_trace,
    summarize_search_trace_payloads,
)
from rag_core.events.types import Event, SearchPlanned, SearchStageCompleted
from rag_core.search.runner import SearchRequest, SearchRunner
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    FakeReranker,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ:
        if key.startswith("RAG_CORE_"):
            monkeypatch.delenv(key)


def test_cli_and_environment_decode_identical_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {
        "RAG_CORE_QDRANT_LOCATION": ":memory:",
        "RAG_CORE_HYBRID": "true",
        "RAG_CORE_EMBEDDING_PROVIDER": "local",
        "RAG_CORE_EMBEDDING_BATCH_SIZE": "7",
        "RAG_CORE_EMBEDDING_BASE_URL": "https://embedding.example.com/v1",
        "RAG_CORE_MANIFEST_DIR": str(tmp_path / "manifest"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    expected = build_config_from_env()
    actual = config_from_args(_build_parser().parse_args(["doctor"]))
    assert actual == expected
    assert actual.ingest.processing_version == DEFAULT_PROCESSING_VERSION
    assert actual.ingest.manifest_directory == tmp_path / "manifest"
    assert actual.embedding.batch_size == 7


@pytest.mark.parametrize(
    "key,value",
    [
        ("RAG_CORE_HYBRID", "maybe"),
        ("RAG_CORE_HYBRID", " "),
        ("RAG_CORE_EMBEDDING_BATCH_SIZE", "bad"),
        ("RAG_CORE_EMBEDDING_DIMENSIONS", "bad"),
        ("RAG_CORE_EMBEDDING_BATCH_SIZE", " "),
    ],
)
def test_bad_environment_is_reported_after_parser_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str,
    value: str,
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as helped:
        cli.main(["search", "--help"])
    assert helped.value.code == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as failed:
        cli.main(["doctor"])
    assert failed.value.code == 2
    error = capsys.readouterr().err
    assert key in error and "Traceback" not in error


def test_cli_overrides_bad_environment_and_replaces_inherited_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CORE_HYBRID", "bad")
    monkeypatch.setenv("RAG_CORE_QDRANT_URL", "https://inherited.example.com")
    args = _build_parser().parse_args(
        ["doctor", "--no-hybrid", "--qdrant-location", ":memory:"]
    )
    config = config_from_args(args)
    assert config.hybrid is False
    assert config.vector_store.qdrant.location == ":memory:"
    assert config.vector_store.qdrant.url is None


def test_zero_batch_is_rejected_instead_of_becoming_default() -> None:
    args = _build_parser().parse_args(["doctor", "--embedding-batch-size", "0"])
    with pytest.raises(ValueError, match="batch_size"):
        config_from_args(args)


def test_folder_cli_config_defaults_and_explicit_values(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    args = _build_parser().parse_args(
        ["search", "billing", str(folder), "--persist-dir", str(tmp_path / "index")]
    )
    config = config_from_args(args, path=folder)
    assert config.hybrid and config.embedding.provider == "local"
    assert config.ingest.processing_version == DEFAULT_PROCESSING_VERSION
    args.embedding_batch_size = 3
    args.processing_version = "app-v2"
    explicit = config_from_args(args, path=folder)
    assert (
        explicit.embedding.batch_size == 3
        and explicit.ingest.processing_version == "app-v2"
    )
    args.mode = "hybrid"
    args.hybrid = False
    with pytest.raises(ValueError, match="require --hybrid"):
        config_from_args(args, path=folder)


def test_engine_closes_store_when_bootstrap_fails() -> None:
    class BrokenStore(RecordingVectorStore):
        closed = False

        async def ensure_collection(self) -> None:
            raise RuntimeError("setup failed")

        async def close(self) -> None:
            self.closed = True

    async def run() -> None:
        store = BrokenStore()
        engine = Engine(
            make_test_config(),
            vector_store=store,
            embedding_provider=FakeEmbeddingProvider(),
        )
        with pytest.raises(RuntimeError, match="setup failed"):
            async with engine:
                pytest.fail("body must not run")
        assert store.closed

    asyncio.run(run())


def test_cli_owns_and_closes_jsonl_sink_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_core.events.sinks import JsonlSink

    closed: list[bool] = []

    class TrackingSink(JsonlSink):
        def close(self) -> None:
            super().close()
            closed.append(True)

    monkeypatch.setattr("rag_core.events.sinks.JsonlSink", TrackingSink)
    trace = tmp_path / "events.jsonl"
    assert (
        cli.main(
            ["doctor", "--embedding-provider", "demo", "--events-jsonl", str(trace)]
        )
        == 0
    )
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli.main(["search", "billing", "--events-jsonl", str(trace)])
    assert len(closed) == 2


def test_eval_defaults_to_real_local_engine_and_labels_demo_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Inject a deterministic embedder to inspect orchestration without downloading models.
    # The assertion below checks selected config, not semantic quality.
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "billing.md").write_text("Invoices use card or ACH.")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "query": "invoice payments",
                "tenant_id": "app",
                "collections": ["docs"],
                "expected_ids": ["billing.md"],
            }
        )
        + "\n"
    )
    seen: list[Config] = []

    def engine(config: Config, **kwargs: Any) -> Engine:
        seen.append(config)
        return Engine(
            config,
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            **kwargs,
        )

    monkeypatch.setattr(cli, "Engine", engine)
    cli.main(
        [
            "eval",
            str(folder),
            str(cases),
            "--persist-dir",
            str(tmp_path / "index"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert seen[0].embedding.provider == "local" and seen[0].hybrid
    assert "demo" not in payload["run"]["embedding_note"]
    cli.main(["eval", str(folder), str(cases), "--demo", "--json"])
    demo = json.loads(capsys.readouterr().out)
    assert demo["run"]["embedding_model"] == "demo-dense-v1"
    assert "not semantic" in demo["run"]["embedding_note"]


@pytest.mark.parametrize("payloads", [False, True])
def test_single_trace_refuses_interleaved_searches(payloads: bool) -> None:
    from dataclasses import asdict

    events: list[Event] = [
        SearchStarted(search_id="a"),
        SearchStarted(search_id="b"),
        SearchCompleted(search_id="a"),
        SearchCompleted(search_id="b"),
    ]
    with pytest.raises(ValueError, match="multiple search_ids"):
        if payloads:
            summarize_search_trace_payloads([asdict(event) for event in events])
        else:
            summarize_search_trace(events)


def test_context_and_plan_events_retain_real_budget_and_audit_context() -> None:
    async def run() -> None:
        store = RecordingVectorStore()
        store.search_results = [
            make_search_result(document_id="bill", text="Invoices use card")
        ]
        buffer = EventBuffer()
        runner = SearchRunner(
            FakeEmbeddingProvider(), FakeSparseEmbedder(), store, FakeReranker(), buffer
        )
        from rag_core.search.request_models import RerankBudget

        req = SearchRequest(
            query="invoice",
            tenant_id="app",
            collections=["docs"],
            limit=2,
            mode="hybrid",
            rerank_budget=RerankBudget(candidate_count=7, timeout_seconds=1.2),
            audit_context=AuditContext(actor="owner", request_id="req-1"),
        )
        await runner.context(req)
        plan = next(
            event for event in buffer.events if isinstance(event, SearchPlanned)
        )
        assert (
            plan.candidate_limit == 7
            and plan.rerank_enabled
            and plan.rerank_timeout_ms == 1200
        )
        context = next(
            event
            for event in buffer.events
            if isinstance(event, SearchStageCompleted) and event.stage == "context_pack"
        )
        assert (
            context.request_id == "req-1"
            and context.actor == "owner"
            and context.search_id == plan.search_id
        )
        assert summarize_search_trace(buffer.events).to_payload()["schema_version"] == 2

    asyncio.run(run())


@pytest.mark.integration
def test_mcp_stdio_remains_streaming_with_jsonl_tracing(tmp_path: Path) -> None:
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    trace = tmp_path / "mcp-events.jsonl"

    async def run() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "rag_core",
                "mcp",
                "--tenant-id",
                "app",
                "--collection",
                "docs",
                "--qdrant-location",
                ":memory:",
                "--embedding-provider",
                "demo",
                "--events-jsonl",
                str(trace),
            ],
            env=dict(
                os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")
            ),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [tool.name for tool in tools.tools] == ["search_user_documents"]
                result = await session.call_tool(
                    "search_user_documents", {"query": "invoice payments"}
                )
                assert not result.isError

    asyncio.run(asyncio.wait_for(run(), 15))
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(event["event_type"] == "search.completed" for event in events)
    assert any(event.get("stage") == "context_pack" for event in events)


def test_prepare_events_keep_each_concurrent_ingest_correlation() -> None:
    from rag_core.events.types import ParseCompleted, ChunkProduced

    async def run() -> None:
        buffer = EventBuffer()
        engine = Engine(
            make_test_config(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=RecordingVectorStore(),
            event_sink=buffer,
        )
        async with engine:
            await asyncio.gather(
                *[
                    engine.add_bytes(
                        file_bytes=b"Invoices accept card.",
                        filename=f"bill-{i}.md",
                        mime_type="text/markdown",
                        tenant_id="app",
                        collection="docs",
                        document_key=f"bill-{i}",
                        audit_context=AuditContext(
                            actor=f"owner-{i}",
                            request_id=f"req-{i}",
                            ingest_id=f"ing-{i}",
                        ),
                    )
                    for i in range(2)
                ]
            )
        prepared = [
            event
            for event in buffer.events
            if isinstance(event, (ParseCompleted, ChunkProduced))
        ]
        assert len(prepared) == 4
        assert {
            (event.actor, event.request_id, event.ingest_id) for event in prepared
        } == {("owner-0", "req-0", "ing-0"), ("owner-1", "req-1", "ing-1")}

    asyncio.run(run())


def test_shipped_model_name_survives_trace_redaction() -> None:
    from rag_core.config.embedding_config import LOCAL_EMBEDDING_MODEL
    from rag_core.events.sink_payloads import event_to_jsonl_dict
    from rag_core.events.types import EmbedCompleted

    assert (
        event_to_jsonl_dict(EmbedCompleted(model=LOCAL_EMBEDDING_MODEL))["model"]
        == LOCAL_EMBEDDING_MODEL
    )
    assert (
        event_to_jsonl_dict(EmbedCompleted(model="namespace/sk-proj-abcdefghijklmnop"))[
            "model"
        ]
        == "unknown"
    )


@pytest.mark.integration
def test_add_returns_failure_status_for_partial_ingest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "billing.txt").write_text("Invoices accept card.")
    (folder / "broken.pdf").write_bytes(b"invalid PDF")
    status = cli.main(
        [
            "add",
            str(folder),
            "--tenant-id",
            "app",
            "--collection",
            "docs",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert status == 1
    assert result["succeeded_count"] == 1 and result["failed_count"] == 1


@pytest.mark.integration
def test_folder_search_with_explicit_store_has_safe_manifest_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "billing.txt").write_text("Invoices accept card.")
    args = [
        "search",
        "invoice",
        str(folder),
        "--qdrant-location",
        ":memory:",
        "--embedding-provider",
        "demo",
        "--json",
    ]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["indexed_count"] == 1
    assert cli.main(args + ["--manifest-dir", str(tmp_path / "manifest")]) == 0
    assert json.loads(capsys.readouterr().out)["indexed_count"] == 1


@pytest.mark.parametrize(
    "mode,limit,expected", [("dense", 5, 5), ("hybrid", 5, 20), ("lexical", 7, 7)]
)
def test_trace_records_resolved_channel_budget_without_reranker(
    mode: Any, limit: int, expected: int
) -> None:
    async def run() -> None:
        store = RecordingVectorStore()
        buffer = EventBuffer()
        runner = SearchRunner(
            FakeEmbeddingProvider(), FakeSparseEmbedder(), store, event_sink=buffer
        )
        await runner.search(
            SearchRequest(
                query="invoice",
                tenant_id="app",
                collections=["docs"],
                limit=limit,
                mode=mode,
                rerank=False,
            )
        )
        plan = next(
            event for event in buffer.events if isinstance(event, SearchPlanned)
        )
        assert plan.candidate_limit == expected
        assert plan.candidate_limit == store.search_calls[0].candidate_limit
        assert plan.limit == limit

    asyncio.run(run())


def test_model_slashes_do_not_relax_other_trace_labels() -> None:
    from rag_core.events.sink_payloads import event_to_jsonl_dict
    from rag_core.events.types import EmbedCompleted
    from rag_core.events.trace_summary_models import safe_search_id
    from rag_core.events.trace_payload_fields import safe_trace_label

    assert safe_search_id("tenant/a") == ""
    assert (
        safe_trace_label("https://example.com/private-project", stage=False)
        == "unknown"
    )
    event = event_to_jsonl_dict(
        EmbedCompleted(
            provider="namespace/provider", model="https://example.com/private-project"
        )
    )
    assert event["provider"] == event["model"] == "unknown"
