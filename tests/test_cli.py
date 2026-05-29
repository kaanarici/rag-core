"""CLI surface tests.

The CLI is intentionally thin — most behavior lives behind ``RAGCore``,
``local_ingest``. These tests focus on the seams the CLI
owns: argparse wiring, validation before runtime setup, structured JSON
output shape, and the exit-code contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from rag_core.cli import main
from rag_core.config import DEFAULT_RERANKER_PROVIDER
from rag_core._engine.core_lifecycle import compute_content_sha256
from rag_core.core_models import CorpusManifestEntry, IngestedDocument, RAGCoreConfig
from rag_core.documents.contextualizer_provider_names import NOOP_CONTEXTUALIZER_ID
from rag_core.events.sinks import DEFAULT_EVENT_SINK_PROVIDER
from rag_core.manifest_persistence import write_entry
from rag_core.sources import document_key as local_document_key
from rag_core.search.context_pack import build_context_pack
from rag_core.search.planning import search_profile
from rag_core.search.providers.cache_provider_names import NO_CACHE_PROVIDER
from rag_core.search.providers.diagnostic_support import (
    READINESS_PACKAGE_AND_ENV,
    SUPPORT_DEFAULT,
    SUPPORT_FIRST_PARTY_OPTIONAL,
    SUPPORT_FIRST_PARTY_UTILITY,
)
from rag_core.search.providers.sparse import SPLADE_LOAD_UNKNOWN_UNTIL_RUN
from rag_core.search.providers.vector_store_diagnostics import (
    VECTOR_STORE_RUNTIME_FAILED,
    VECTOR_STORE_RUNTIME_HEALTHY,
    VECTOR_STORE_RUNTIME_NOT_REQUESTED,
)
from rag_core.search.types import And, Term
from tests.support import make_search_result


_SearchHandler = Callable[..., Any]
_RetrieveContextHandler = Callable[..., Any]
_IngestHandler = Callable[..., Any]


class _FakeRAGCore:
    """One configurable stand-in for ``RAGCore`` across CLI tests.

    Tests register the behavior they care about via ``search_handler`` /
    ``ingest_handler``. The default handlers return a single hit for
    ``doc-1`` and a single-chunk ``IngestedDocument`` respectively, which
    is enough for the happy-path metrics and serialization tests.
    """

    search_handler: _SearchHandler | None = None
    retrieve_context_handler: _RetrieveContextHandler | None = None
    ingest_handler: _IngestHandler | None = None
    instances: list["_FakeRAGCore"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False
        self.event_sink = kwargs.get("event_sink")
        self.search_calls: list[dict[str, Any]] = []
        self.retrieve_context_calls: list[dict[str, Any]] = []
        self.ingest_calls: list[dict[str, Any]] = []
        type(self)._last_instance = self  # type: ignore[attr-defined]
        type(self).instances.append(self)

    async def ensure_ready(self) -> None:
        return None

    async def search(self, **kwargs: Any) -> list[object]:
        self.search_calls.append(kwargs)
        handler = type(self).search_handler
        if handler is not None:
            return cast(list[object], handler(self, **kwargs))
        return [make_search_result(document_id="doc-1")]

    async def retrieve_context(self, **kwargs: Any) -> object:
        self.retrieve_context_calls.append(kwargs)
        handler = type(self).retrieve_context_handler
        if handler is not None:
            return handler(self, **kwargs)
        return build_context_pack(
            [
                make_search_result(
                    id="private-hit-1",
                    document_id="internal-doc-id",
                    document_key="private/billing.md",
                    title=None,
                    text="billing context",
                )
            ],
            query=cast(str, kwargs.get("query", "")),
        )

    async def ingest_file(self, path: Path, **kwargs: Any) -> IngestedDocument:
        self.ingest_calls.append({"path": path, **kwargs})
        handler = type(self).ingest_handler
        if handler is not None:
            return cast(IngestedDocument, handler(self, path, **kwargs))
        return IngestedDocument(
            document_id=f"doc-{path.stem}",
            corpus_id=kwargs.get("corpus_id", ""),
            namespace=kwargs.get("namespace", ""),
            chunk_count=1,
            filename=path.name,
            mime_type="text/markdown",
            document_key=kwargs.get("document_key"),
            ingest_state="created",
        )

    async def close(self) -> None:
        self.closed = True


class _UnexpectedRAGCore:
    """Tracks construction; tests assert validation fired before this ran."""

    constructed: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        type(self).constructed = True


class _FakeOpenAIError(Exception):
    __module__ = "openai"


def _search_calls(fake_core: type[_FakeRAGCore]) -> list[dict[str, Any]]:
    return [
        call
        for instance in fake_core.instances
        for call in instance.search_calls
    ]


@pytest.fixture
def fake_core(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRAGCore]:
    from rag_core import cli as cli_module

    fake = type("_FakeRAGCoreLocal", (_FakeRAGCore,), {"instances": []})
    monkeypatch.setattr(cli_module, "RAGCore", fake)
    return fake


@pytest.fixture
def unexpected_core(monkeypatch: pytest.MonkeyPatch) -> type[_UnexpectedRAGCore]:
    from rag_core import cli as cli_module

    sentinel = type("_UnexpectedRAGCoreLocal", (_UnexpectedRAGCore,), {"constructed": False})
    monkeypatch.setattr(cli_module, "RAGCore", sentinel)
    return sentinel




def _expect_cli_error(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert message in error
    assert "Traceback" not in error


@pytest.mark.parametrize("command", ["search", "retrieve-context"])
def test_query_commands_reject_non_positive_limit_before_runtime_setup(
    command: str,
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _expect_cli_error(
        [
            command,
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--limit",
            "0",
        ],
        "--limit must be positive",
        capsys,
    )

    assert unexpected_core.constructed is False


def test_doctor_json_reports_planned_core(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "doctor",
            "--json",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["collection_name"] == "rag_core_chunks__text_embedding_3_small_1536d"
    assert "retrieval" not in payload
    assert payload["search"]["default_search_profile"] == "balanced"
    assert payload["embedding"]["model"] == "text-embedding-3-small"
    assert payload["embedding"]["dimensions"] == 1536
    assert payload["reranker"]["effective"] == DEFAULT_RERANKER_PROVIDER
    assert payload["vector_store"]["configured"] == "qdrant"
    assert {"qdrant", "turbopuffer", "memory"}.issubset(
        payload["vector_store"]["registered"]
    )
    qdrant = payload["vector_store"]["providers"]["qdrant"]
    assert qdrant["support_level"] == SUPPORT_DEFAULT
    assert qdrant["configured"] is True
    assert qdrant["package_available"] is True
    assert qdrant["credential_required"] is False
    assert qdrant["api_key_configured"] is False
    assert qdrant["runtime_validated"] is False
    assert qdrant["runtime_validation"] == VECTOR_STORE_RUNTIME_NOT_REQUESTED
    assert qdrant["dimension_aware_collection"] is True
    assert qdrant["per_point_delete"] is True
    assert qdrant["document_record_lookup"] is True
    turbopuffer = payload["vector_store"]["providers"]["turbopuffer"]
    assert turbopuffer["support_level"] == SUPPORT_FIRST_PARTY_OPTIONAL
    assert turbopuffer["configured"] is False
    assert isinstance(turbopuffer["package_available"], bool)
    assert turbopuffer["credential_required"] is True
    assert turbopuffer["api_key_configured"] is False
    assert turbopuffer["runtime_validated"] is False
    assert turbopuffer["runtime_validation"] == VECTOR_STORE_RUNTIME_NOT_REQUESTED
    memory = payload["vector_store"]["providers"]["memory"]
    assert memory["support_level"] == SUPPORT_FIRST_PARTY_UTILITY
    assert memory["configured"] is False
    assert memory["package_available"] is True
    assert memory["credential_required"] is False
    assert memory["check_store_supported"] is False
    assert memory["query_plan"]["hybrid_rrf"] is True
    provider_categories = payload["providers"]
    assert provider_categories["sparse"]["configured"] == "fastembed"
    fastembed = provider_categories["sparse"]["providers"]["fastembed"]
    assert fastembed["readiness_scope"] == READINESS_PACKAGE_AND_ENV
    assert fastembed["channels"]["splade"]["live_ready"] is None
    assert fastembed["channels"]["splade"]["load_status"] == SPLADE_LOAD_UNKNOWN_UNTIL_RUN
    assert provider_categories["ocr"]["providers"]["mistral"]["support_level"] == (
        SUPPORT_FIRST_PARTY_OPTIONAL
    )
    assert provider_categories["contextualizer"]["configured"] == NOOP_CONTEXTUALIZER_ID
    assert provider_categories["embedding_cache"]["configured"] == NO_CACHE_PROVIDER
    assert provider_categories["chunk_context_cache"]["configured"] == NO_CACHE_PROVIDER
    assert provider_categories["search_sidecar"]["configured"] is None
    assert "" not in provider_categories["search_sidecar"]["providers"]
    assert provider_categories["event_sink"]["configured"] == DEFAULT_EVENT_SINK_PROVIDER
    assert "pdf_inspector" in payload


def test_cli_demo_embedding_provider_defaults_to_demo_model() -> None:
    from rag_core.cli_parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "ingest",
            "examples/demo_corpus",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "64",
        ]
    )

    config = RAGCoreConfig.from_cli(args)

    assert config.embedding.provider == "demo"
    assert config.embedding.model == "demo-dense-v1"
    assert config.embedding.dimensions == 64


def test_doctor_json_reports_turbopuffer_env_without_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "secret-key")
    monkeypatch.setenv("TURBOPUFFER_REGION", "aws-us-west-2")
    monkeypatch.setenv("TURBOPUFFER_BASE_URL", "https://example.invalid")

    exit_code = main(
        [
            "doctor",
            "--json",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "secret-key" not in output
    payload = json.loads(output)
    turbopuffer = payload["vector_store"]["providers"]["turbopuffer"]
    assert turbopuffer["api_key_configured"] is True
    assert turbopuffer["region"] == "aws-us-west-2"
    assert turbopuffer["base_url_configured"] is True
    assert turbopuffer["configured"] is False


def test_doctor_json_reports_turbopuffer_selection_without_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "doctor",
            "--json",
            "--vector-store",
            "turbopuffer",
            "--turbopuffer-namespace",
            "prod-docs",
            "--turbopuffer-api-key",
            "secret-key",
            "--turbopuffer-region",
            "aws-us-west-2",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "secret-key" not in output
    payload = json.loads(output)
    assert payload["collection_name"] == "prod-docs"
    assert payload["vector_store"]["configured"] == "turbopuffer"
    assert payload["vector_store"]["providers"]["qdrant"]["configured"] is False
    turbopuffer = payload["vector_store"]["providers"]["turbopuffer"]
    assert turbopuffer["configured"] is True
    assert turbopuffer["namespace"] == "prod-docs"
    assert turbopuffer["api_key_configured"] is True


def test_doctor_json_redacts_qdrant_url_sensitive_parts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_url = "https://user:pass@cluster.example.com:6333/store?api_key=secret#frag"

    exit_code = main(
        [
            "doctor",
            "--json",
            "--qdrant-url",
            raw_url,
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "user" not in output
    assert "pass" not in output
    assert "secret" not in output
    assert "frag" not in output
    payload = json.loads(output)
    expected = "https://cluster.example.com:6333/store?redacted"
    assert payload["qdrant"]["url"] == expected
    assert payload["vector_store"]["providers"]["qdrant"]["url"] == expected


def test_doctor_json_redacts_pdf_inspector_binary_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_path = "/private/tmp/pdf-inspector-secret/bin"
    monkeypatch.setenv("PDF_INSPECTOR_BINARY_PATH", raw_path)

    exit_code = main(
        [
            "doctor",
            "--json",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert raw_path not in output
    payload = json.loads(output)
    assert payload["pdf_inspector"]["binary_path"] == "configured"
    assert payload["pdf_inspector"]["binary_path_configured"] is True


def test_doctor_json_summarizes_qdrant_local_store_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store_path = tmp_path / "private" / "qdrant-store"

    exit_code = main(
        [
            "doctor",
            "--json",
            "--qdrant-location",
            str(store_path),
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert str(store_path) not in output
    payload = json.loads(output)
    assert payload["qdrant"]["location"] == "local_path_configured"
    assert (
        payload["vector_store"]["providers"]["qdrant"]["location"]
        == "local_path_configured"
    )


def test_doctor_check_store_creates_local_collection(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "doctor",
            "--json",
            "--check-store",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["store_health"]["healthy"] is True
    assert payload["store_health"]["collection"] == "rag_core_chunks__text_embedding_3_small_1536d"
    qdrant = payload["vector_store"]["providers"]["qdrant"]
    assert qdrant["runtime_validated"] is True
    assert qdrant["runtime_validation"] == VECTOR_STORE_RUNTIME_HEALTHY


def test_doctor_check_store_exits_nonzero_when_health_is_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rag_core import cli_doctor
    from rag_core.cli_doctor_store import DoctorStoreOutcome

    async def _unhealthy_store(*args: object, **kwargs: object) -> DoctorStoreOutcome:
        return DoctorStoreOutcome(
            health={"healthy": False, "error": "store offline"},
            fix_summary={"status": "ok"},
        )

    monkeypatch.setattr(cli_doctor, "exercise_doctor_store", _unhealthy_store)

    exit_code = main(
        [
            "doctor",
            "--json",
            "--check-store",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["store_health"]["healthy"] is False
    qdrant = payload["vector_store"]["providers"]["qdrant"]
    assert qdrant["runtime_validated"] is False
    assert qdrant["runtime_validation"] == VECTOR_STORE_RUNTIME_FAILED


def test_doctor_default_output_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "doctor",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Collection:" in output
    assert "Processing Version:" in output
    assert "Embedding:" in output
    assert "Vector Store:" in output
    assert not output.lstrip().startswith("{")


def test_demo_json_runs_without_external_services(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["demo", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunk_count"] > 0
    assert payload["hits"]


def test_local_search_indexes_folder_and_returns_hits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    billing = docs / "billing.md"
    billing.write_text(
        "Invoices can be paid by card or ACH every month.", encoding="utf-8"
    )
    (docs / "support.txt").write_text(
        "Support tickets are reviewed on weekdays.", encoding="utf-8"
    )

    exit_code = main(["local-search", str(docs), "How do I pay invoices?", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed_count"] == 2
    assert payload["corpus_id"] == "docs"
    assert payload["hits"][0]["document_key"] == local_document_key(docs, billing)
    assert "invoices" in payload["hits"][0]["text"].lower()


def test_local_search_runs_against_checked_in_examples_folder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "local-search",
            "examples",
            "corpus lifecycle",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed_count"] >= 3
    assert payload["skipped_count"] == 0
    assert payload["skipped_failed"] == []
    assert payload["corpus_id"] == "examples"
    examples = Path("examples")
    assert payload["hits"][0]["document_key"] == local_document_key(
        examples,
        examples / "demo_corpus" / "corpus_lifecycle.md",
    )


def test_local_search_writes_events_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(
        "Invoices can be paid by card or ACH every month.", encoding="utf-8"
    )
    events = tmp_path / "traces" / "events.jsonl"

    exit_code = main(
        [
            "local-search",
            str(docs),
            "How do I pay invoices?",
            "--events-jsonl",
            str(events),
            "--json",
        ]
    )

    assert exit_code == 0
    event_payloads = [
        json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
    ]
    event_types = {payload["event_type"] for payload in event_payloads}
    assert {
        "ingest.started",
        "parse.completed",
        "chunk.produced",
        "index.upserted",
        "ingest.completed",
        "search.started",
        "search.planned",
        "search.stage.completed",
        "search.completed",
    }.issubset(event_types)
    parse_payload = next(p for p in event_payloads if p["event_type"] == "parse.completed")
    assert parse_payload["quality_verdict"]
    assert parse_payload["char_count"] > 0
    assert parse_payload["page_count"] >= 1


@pytest.mark.parametrize(
    ("extra_files", "indexed", "skipped"),
    (
        ([("archive.bin", b"\x00\x01")], 1, 1),
        (
            [("~$draft.docx", b"not a real docx"), ("scan.png", b"\x89PNG\r\n\x1a\n")],
            1,
            2,
        ),
    ),
)
def test_local_search_skips_unsupported_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_files: list[tuple[str, bytes]],
    indexed: int,
    skipped: int,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "readme.txt").write_text("rag retrieval smoke text", encoding="utf-8")
    for name, content in extra_files:
        (docs / name).write_bytes(content)

    exit_code = main(["local-search", str(docs), "retrieval", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed_count"] == indexed
    assert payload["skipped_count"] == skipped
    assert payload["skipped_failed"] == []


def test_local_search_reports_empty_supported_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "readme.txt").write_text("rag retrieval smoke text", encoding="utf-8")
    (docs / "empty.md").write_text("", encoding="utf-8")

    exit_code = main(["local-search", str(docs), "retrieval", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed_count"] == 1
    assert payload["skipped_count"] == 1
    assert payload["skipped_empty_count"] == 1
    assert payload["skipped_unsupported_count"] == 0


def test_local_search_human_output_explains_skips_and_truncation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("rag retrieval smoke text", encoding="utf-8")
    (docs / "b.txt").write_text("second retrieval smoke text", encoding="utf-8")
    (docs / "empty.md").write_text("", encoding="utf-8")
    (docs / "archive.bin").write_bytes(b"\x00\x01")

    exit_code = main(["local-search", str(docs), "retrieval", "--max-files", "1"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Indexed: 1 files" in output
    assert "Skipped: 2 files (unsupported=1, empty=1, failed=0)" in output
    assert "Truncated: yes; rerun with --max-files" in output
    assert "Top hits:" in output


def test_local_search_missing_file_reports_cli_error(capsys: pytest.CaptureFixture[str]) -> None:
    _expect_cli_error(["local-search", "does-not-exist", "query"], "file not found", capsys)


def test_local_eval_indexes_folder_and_returns_redacted_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(
        "Invoices can be paid by card or ACH every month.", encoding="utf-8"
    )
    (docs / "support.txt").write_text(
        "Support tickets are reviewed on weekdays.", encoding="utf-8"
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "case_id": "billing-payment-methods",
                "query": "How do customers pay invoices?",
                "namespace": "acme",
                "corpus_ids": ["help-center"],
                "expected_ids": ["billing.md"],
                "expected_context_contains": ["card", "ACH"],
                "forbidden_context_contains": ["# Metadata", "# Content"],
                "forbidden_private_identifiers": ["local:", "content_sha256"],
                "expected_citation_count_min": 1,
                "expected_source_count_min": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "local-eval",
            str(docs),
            str(cases),
            "--min-recall-at-5",
            "1",
            "--min-mrr",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 1
    assert payload["metrics"]["recall_at_5"] == 1.0
    assert payload["metrics"]["mrr"] == 1.0
    assert payload["metrics"]["context_recall"] == 1.0
    assert payload["metrics"]["prompt_safety_pass_rate"] == 1.0
    assert payload["quality_gate"]["passed"] is True
    assert payload["run"]["mode"] == "local_eval"
    assert payload["run"]["namespace"] == "acme"
    assert payload["run"]["corpus_id"] == "help-center"
    assert payload["run"]["indexed_count"] == 2
    assert payload["cases"][0]["case_label"] == "case-1"
    assert "case_id" not in payload["cases"][0]
    assert "query" not in payload["cases"][0]
    assert "expected_ids" not in payload["cases"][0]
    assert "expected_context_contains" not in payload["cases"][0]
    assert payload["cases"][0]["context_contains_pass"] is True
    assert payload["cases"][0]["prompt_safety_pass"] is True
    assert payload["cases"][0]["forbidden_leak_count"] == 0


def test_local_eval_quality_gate_failure_sets_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(
        "Invoices can be paid by card or ACH every month.", encoding="utf-8"
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "query": "How do customers pay invoices?",
                "namespace": "acme",
                "corpus_ids": ["help-center"],
                "expected_ids": ["missing.md"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "local-eval",
            str(docs),
            str(cases),
            "--min-recall-at-5",
            "1",
            "--json",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_gate"]["passed"] is False
    assert payload["quality_gate"]["failures"][0]["metric"] == "recall_at_5"


def test_local_eval_rejects_invalid_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text("Invoices can be paid by card.", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "query": "How do customers pay invoices?",
                "namespace": "acme",
                "corpus_ids": ["help-center"],
                "expected_ids": ["billing.md"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _expect_cli_error(
        ["local-eval", str(docs), str(cases), "--min-mrr", "2"],
        "--min-mrr must be between 0 and 1",
        capsys,
    )


def test_manifest_json_previews_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing docs stay easy to find", encoding="utf-8")

    exit_code = main(
        [
            "manifest",
            str(file_path),
            "--namespace",
            "acme",
            "--corpus-id",
            "help-center",
            "--metadata",
            "source=seed",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_entry"]["filename"] == "guide.txt"
    assert payload["manifest_entry"]["parser"] == "local:text"
    assert payload["manifest_entry"]["metadata"]["source"] == "seed"
    assert payload["document"]["ingest_state"] == "preview"


@pytest.mark.parametrize(
    ("target", "namespace", "extra_argv", "message"),
    (
        ("missing", "acme", [], "file not found"),
        ("dir", "acme", [], "manifest path must be a file"),
        ("good", "acme", ["--metadata", "broken"], "metadata entries must use KEY=VALUE"),
        ("good", "../escape", [], "single non-empty path segment"),
        ("image", "acme", [], "no supported file matched"),
    ),
    ids=("missing-file", "directory", "bad-metadata", "bad-scope", "unsupported"),
)
def test_manifest_validation_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    namespace: str,
    extra_argv: list[str],
    message: str,
) -> None:
    good = tmp_path / "good.txt"
    good.write_text("billing docs stay easy to find", encoding="utf-8")
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    target_path = {
        "missing": "does-not-exist.txt",
        "dir": str(tmp_path),
        "good": str(good),
        "image": str(image),
    }[target]

    _expect_cli_error(
        [
            "manifest",
            target_path,
            "--namespace",
            namespace,
            "--corpus-id",
            "help-center",
            *extra_argv,
        ],
        message,
        capsys,
    )



@pytest.mark.parametrize(
    "argv",
    (
        [
            "search",
            "billing",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            "--search-profile",
            "balanced",
            "--query-plan-preset",
            "hybrid_rrf",
        ],
    ),
)
def test_search_profile_rejects_query_plan_conflict_before_runtime_setup(
    argv: list[str],
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _expect_cli_error(argv, "not allowed with argument --search-profile", capsys)
    assert unexpected_core.constructed is False


def test_query_metadata_filter_passes_exact_terms_to_core(
    fake_core: type[_FakeRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--limit",
            "7",
            "--qdrant-location",
            ":memory:",
            "--metadata-filter",
            "team=support",
            "--metadata-filter",
            "tier=enterprise",
            "--content-type",
            "document",
            "--content-type",
            "code",
            "--document-id",
            "doc-1",
            "--document-id",
            "doc-2",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["document_id"] == "doc-1"
    instance = cast(_FakeRAGCore, fake_core._last_instance)  # type: ignore[attr-defined]
    call = instance.search_calls[0]
    assert call["limit"] == 7
    assert call["query_plan"] is None
    assert call["content_types"] == ["document", "code"]
    assert call["document_ids"] == ["doc-1", "doc-2"]
    assert call["metadata_filter"] == And(
        filters=(
            Term(field="team", value="support"),
            Term(field="tier", value="enterprise"),
        )
    )


def test_query_search_profile_passes_query_plan_to_core(
    fake_core: type[_FakeRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_plan = search_profile("lexical", limit=7)

    exit_code = main(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--limit",
            "7",
            "--qdrant-location",
            ":memory:",
            "--search-profile",
            "lexical",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["document_id"] == "doc-1"
    instance = cast(_FakeRAGCore, fake_core._last_instance)  # type: ignore[attr-defined]
    assert instance.search_calls[0]["query_plan"] == expected_plan


def test_retrieve_context_json_uses_prompt_safe_context_text(
    fake_core: type[_FakeRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "retrieve-context",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_text"].startswith("[S1] file")
    assert "billing context" in payload["context_text"]
    assert "internal-doc-id" not in payload["context_text"]
    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    source = snippets[0]["source"]
    assert isinstance(source, dict)
    assert "source_id" in source
    assert "result_id" in source


def test_query_event_sink_failure_does_not_emit_success_payload(
    fake_core: type[_FakeRAGCore],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_payload(_: object) -> dict[str, object]:
        raise RuntimeError("sink failed")

    def handler(self: _FakeRAGCore, **kw: Any) -> list[object]:
        from rag_core.events import SearchStarted

        assert self.event_sink is not None
        self.event_sink.emit(
            SearchStarted(query_length=7, corpus_ids=("help",), limit=1)
        )
        return [make_search_result(document_id="doc-1")]

    fake_core.search_handler = handler
    monkeypatch.setattr("rag_core.events.sinks.event_to_jsonl_dict", fail_payload)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "search",
                "billing",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
                "--events-jsonl",
                str(tmp_path / "events.jsonl"),
                "--json",
            ]
        )

    assert exc_info.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "events JSONL sink failed to write" in output.err


def test_query_bad_metadata_filter_reports_cli_error(
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _expect_cli_error(
        [
            "search",
            "billing",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            "--metadata-filter",
            "broken",
        ],
        "metadata entries must use KEY=VALUE",
        capsys,
    )
    assert unexpected_core.constructed is False


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--content-type", "--content-type values must be non-empty"),
        ("--document-id", "--document-id values must be non-empty"),
    ],
)
def test_query_scope_filters_reject_blank_values_before_runtime_setup(
    flag: str,
    message: str,
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _expect_cli_error(
        [
            "search",
            "billing",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            flag,
            " ",
        ],
        message,
        capsys,
    )
    assert unexpected_core.constructed is False


def test_query_provider_bootstrap_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rag_core import cli as cli_module

    class _ProviderFailingCore:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _FakeOpenAIError(
                "The api_key client option must be set by passing api_key or OPENAI_API_KEY"
            )

    monkeypatch.setattr(cli_module, "RAGCore", _ProviderFailingCore)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "search",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "provider setup failed before search" in error
    assert "provider=openai" in error
    assert "api_key client option" not in error
    assert "OPENAI_API_KEY" in error
    assert "Traceback" not in error


def test_retrieve_context_provider_bootstrap_error_uses_retrieve_context_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rag_core import cli as cli_module

    class _ProviderFailingCore:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _FakeOpenAIError(
                "The api_key client option must be set by passing api_key or OPENAI_API_KEY"
            )

    monkeypatch.setattr(cli_module, "RAGCore", _ProviderFailingCore)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "retrieve-context",
                "billing policy",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "provider setup failed before retrieve-context" in error
    assert "provider setup failed before search" not in error
    assert "provider=openai" in error
    assert "api_key client option" not in error
    assert "OPENAI_API_KEY" in error


def test_ingest_bad_manifest_scope_reports_before_runtime_setup(
    tmp_path: Path,
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("hello", encoding="utf-8")

    _expect_cli_error(
        [
            "ingest",
            str(docs),
            "--namespace",
            "../escape",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
        ],
        "single non-empty path segment",
        capsys,
    )
    assert unexpected_core.constructed is False


def test_ingest_unsupported_file_reports_supported_file_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                str(image),
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "no supported file matched" in error
    assert "Images require an injected OCR provider" in error
    assert "Traceback" not in error


def test_ingest_json_reports_per_file_failures_and_continues(
    tmp_path: Path,
    fake_core: type[_FakeRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    good = docs / "a.md"
    good.write_text("alpha", encoding="utf-8")
    bad = docs / "b.txt"
    bad.write_text("broken", encoding="utf-8")
    events = tmp_path / "traces" / "ingest.jsonl"
    good_hash = compute_content_sha256(b"alpha")
    bad_hash = compute_content_sha256(b"broken")

    def handler(self: _FakeRAGCore, path: Path, **kw: Any) -> IngestedDocument:
        if path == bad:
            raise ValueError("failed to parse")
        return IngestedDocument(
            document_id="doc-a",
            corpus_id=kw["corpus_id"],
            namespace=kw["namespace"],
            chunk_count=1,
            filename=path.name,
            mime_type="text/markdown",
            document_key=kw.get("document_key"),
            ingest_state="created",
        )

    fake_core.ingest_handler = handler

    exit_code = main(
        [
            "ingest",
            str(docs),
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            "--manifest-dir",
            str(tmp_path / "manifest"),
            "--metadata",
            "source=seed",
            "--force-reindex",
            "--events-jsonl",
            str(events),
            "--json",
        ]
    )

    assert exit_code == 1
    stdout = capsys.readouterr().out
    payloads = [
        json.loads(line) for line in stdout.splitlines() if line.strip()
    ]
    assert [p["ok"] for p in payloads] == [True, False]
    success, failure = payloads
    assert success["path"] == "<local-file>"
    assert success["filename"] == "a.md"
    assert "content_sha256" not in success
    assert "document_key" not in success
    assert success["chunk_count"] == 1
    assert success["ingest_state"] == "created"
    assert success["manifest_status"] == "missing"
    assert failure["path"] == "<local-file>"
    assert failure["filename"] == "b.txt"
    assert "content_sha256" not in failure
    assert "document_key" not in failure
    assert failure["error"] == "failed to parse"
    assert str(good) not in stdout
    assert str(bad) not in stdout
    assert good_hash not in json.dumps(payloads)
    assert bad_hash not in json.dumps(payloads)

    instance = cast(_FakeRAGCore, fake_core._last_instance)  # type: ignore[attr-defined]
    paths_seen = [call["path"] for call in instance.ingest_calls]
    assert paths_seen == [good, bad]
    assert instance.ingest_calls[0]["metadata"] == {"source": "seed"}
    assert instance.ingest_calls[0]["force_reindex"] is True
    assert instance.closed is True

    event_payloads = [
        json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [p["event_type"] for p in event_payloads]
    assert event_types == [
        "ingest.batch.started",
        "ingest.batch.progress",
        "ingest.batch.progress",
        "ingest.batch.completed",
    ]
    assert event_payloads[0]["planned_count"] == 2
    assert event_payloads[1]["status"] == "succeeded"
    assert "content_sha256" not in event_payloads[1]
    assert event_payloads[2]["status"] == "failed"
    assert "content_sha256" not in event_payloads[2]
    assert event_payloads[3]["succeeded_count"] == 1
    assert event_payloads[3]["failed_count"] == 1


def test_ingest_surfaces_provider_bootstrap_as_batch_setup_error(
    tmp_path: Path,
    fake_core: type[_FakeRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    bad = docs / "b.txt"
    bad.write_text("broken", encoding="utf-8")

    def handler(self: _FakeRAGCore, path: Path, **kw: Any) -> IngestedDocument:
        raise _FakeOpenAIError(
            "raw api_key client option OPENAI_API_KEY private-token"
        )

    fake_core.ingest_handler = handler

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                str(docs),
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
                "--json",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out == ""
    assert "provider failed during ingest" in captured.err
    assert "provider setup failed before ingest" not in captured.err
    assert "provider=openai" in captured.err
    assert "OPENAI_API_KEY" not in captured.err
    assert "raw api_key" not in captured.err
    assert "private-token" not in captured.err
    instance = cast(_FakeRAGCore, fake_core._last_instance)  # type: ignore[attr-defined]
    assert instance.closed is True


def test_ingest_json_redacts_provider_exception_text_in_per_file_failure(
    tmp_path: Path,
    fake_core: type[_FakeRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    bad = docs / "b.txt"
    bad.write_text("broken", encoding="utf-8")

    def handler(self: _FakeRAGCore, path: Path, **kw: Any) -> IngestedDocument:
        raise _FakeOpenAIError(f"rate limit for {bad} private-token")

    fake_core.ingest_handler = handler

    exit_code = main(
        [
            "ingest",
            str(docs),
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
            "--json",
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "private-token" not in output
    [failure] = [json.loads(line) for line in output.splitlines() if line.strip()]
    assert failure["ok"] is False
    assert str(bad) not in failure["error"]
    assert "provider=openai" in failure["error"]
    assert "error_type=_FakeOpenAIError" in failure["error"]
    assert "rate limit" not in failure["error"]


def test_ingest_plan_json_reports_fingerprints_and_manifest_reconciliation(
    tmp_path: Path,
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    unchanged = docs / "unchanged.md"
    unchanged.write_text("same", encoding="utf-8")
    changed = docs / "changed.md"
    changed.write_text("new", encoding="utf-8")
    manifest_dir = tmp_path / "manifest"
    unchanged_hash = compute_content_sha256(b"same")
    changed_hash = compute_content_sha256(b"new")
    unchanged_key = local_document_key(docs, unchanged)
    changed_key = local_document_key(docs, changed)

    for entry in (
        CorpusManifestEntry(
            document_id="doc-unchanged",
            namespace="acme",
            corpus_id="help",
            document_key=unchanged_key,
            content_sha256=unchanged_hash,
            filename="unchanged.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
        CorpusManifestEntry(
            document_id="doc-changed",
            namespace="acme",
            corpus_id="help",
            document_key=changed_key,
            content_sha256="old-hash",
            filename="changed.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
        CorpusManifestEntry(
            document_id="doc-orphan",
            namespace="acme",
            corpus_id="help",
            document_key="removed.md",
            content_sha256="orphan-hash",
            filename="removed.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    ):
        write_entry(manifest_dir, entry)

    exit_code = main(
        [
            "ingest",
            str(docs),
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--manifest-dir",
            str(manifest_dir),
            "--plan-json",
        ]
    )

    assert exit_code == 0
    assert unexpected_core.constructed is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_count"] == 2
    docs_payload = payload["documents"]
    assert payload["path"] == "<local-source>"
    assert [d["path"] for d in docs_payload] == ["<local-file>", "<local-file>"]
    assert [d["filename"] for d in docs_payload] == ["changed.md", "unchanged.md"]
    assert all("document_key" not in document for document in docs_payload)
    assert all("content_sha256" not in document for document in docs_payload)
    assert docs_payload[0]["manifest_status"] == "changed"
    assert docs_payload[0]["content_sha256_available"] is True
    assert docs_payload[1]["manifest_status"] == "unchanged"
    assert docs_payload[1]["content_sha256_available"] is True
    assert payload["reconciliation"]["summary"] == {
        "changed_count": 1,
        "duplicate_count": 0,
        "missing_count": 0,
        "needs_reindex_count": 1,
        "orphaned_count": 1,
        "unchanged_count": 1,
    }
    items = [
        (item["item_index"], item["status"], item["has_source_content_sha256"])
        for item in payload["reconciliation"]["items"]
    ]
    assert items == [
        (0, "changed", True),
        (1, "unchanged", True),
        (2, "orphaned", False),
    ]
    assert changed_key not in json.dumps(payload)
    assert unchanged_key not in json.dumps(payload)
    assert changed_hash not in json.dumps(payload)
    assert unchanged_hash not in json.dumps(payload)


def test_ingest_plan_json_surfaces_duplicate_manifest_keys(
    tmp_path: Path,
    unexpected_core: type[_UnexpectedRAGCore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "same.md"
    source.write_text("same", encoding="utf-8")
    source_hash = compute_content_sha256(b"same")
    source_key = local_document_key(docs, source)
    manifest_dir = tmp_path / "manifest"
    for document_id in ("doc-a", "doc-b"):
        write_entry(
            manifest_dir,
            CorpusManifestEntry(
                document_id=document_id,
                namespace="acme",
                corpus_id="help",
                document_key=source_key,
                content_sha256=source_hash,
                filename="same.md",
                mime_type="text/markdown",
                chunk_count=1,
            ),
        )

    exit_code = main(
        [
            "ingest",
            str(docs),
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--manifest-dir",
            str(manifest_dir),
            "--plan-json",
        ]
    )

    assert exit_code == 0
    assert unexpected_core.constructed is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"][0]["manifest_status"] == "duplicate"
    assert payload["documents"][0]["manifest_reason"] == "duplicate_manifest_document_key"
    assert payload["reconciliation"]["summary"]["duplicate_count"] == 2


def test_doctor_fix_creates_collection_on_fresh_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "doctor",
            "--json",
            "--fix",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["store_health"]["healthy"] is True
    assert payload["fix"]["status"] == "ok"
    assert payload["fix"]["expected"] == 1536


def test_doctor_fix_reports_dimension_mismatch_without_mutating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Real dimension drift requires a persisted collection across two runs,
    # which in-memory Qdrant does not provide. Simulate the adapter's exact
    # failure message (owned by qdrant_collection.py) to exercise the CLI's
    # structured reaction without forking a second storage adapter.
    from rag_core.core import RAGCore as _RAGCore

    async def _fail_with_mismatch(self: Any) -> None:
        raise ValueError(
            "Existing collection rag_core_chunks uses 1536 dimensions, "
            "but the current embedding provider uses 768. "
            "Use a different collection name or reindex with a matching embedding configuration."
        )

    monkeypatch.setattr(_RAGCore, "ensure_ready", _fail_with_mismatch)

    exit_code = main(
        [
            "doctor",
            "--json",
            "--fix",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
            "--embedding-dimensions",
            "768",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    fix = payload["fix"]
    assert fix["status"] == "dimension_mismatch"
    assert fix["expected"] == 768
    assert fix["actual"] == 1536
    assert "reindex" in fix["message"].lower()


def test_doctor_fix_human_diff_line_on_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_core.core import RAGCore as _RAGCore

    async def _fail_with_mismatch(self: Any) -> None:
        raise ValueError(
            "Existing collection rag_core_chunks uses 3072 dimensions, "
            "but the current embedding provider uses 1536. ..."
        )

    monkeypatch.setattr(_RAGCore, "ensure_ready", _fail_with_mismatch)

    exit_code = main(
        [
            "doctor",
            "--fix",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Fix: dimension_mismatch" in output
    assert "expected=1536" in output
    assert "actual=3072" in output


def test_doctor_fix_human_output_includes_summary_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "doctor",
            "--fix",
            "--qdrant-location",
            ":memory:",
            "--embedding-model",
            "text-embedding-3-small",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Fix: ok" in output
