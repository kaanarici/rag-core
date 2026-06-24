from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from rag_core.cli.parser import _build_parser
from rag_core.cli.parsers.serve import JOB_RETENTION_SECONDS_ENV
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_LIMIT_MAX,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
)
from rag_core.runtime.app import create_app
from rag_core.runtime_defaults import DEFAULT_RUNTIME_JOB_DB_PATH_ENV
from rag_core.runtime.requests import (
    DEFAULT_RUNTIME_CONTEXT_LIMIT,
    DEFAULT_RUNTIME_SEARCH_LIMIT,
    parse_retrieval_request,
)
from rag_core.search.context_pack import build_context_pack
from rag_core.search.providers.embedding import create_embedding_provider
from rag_core.search.providers.memory_store import InMemoryVectorStore
from rag_core.search.request_models import RerankResult
from tests.support import make_search_result

pytestmark = [pytest.mark.integration]


_OPENAPI_PATH = Path("docs/self-host/openapi.yaml")


def _openapi_methods_by_path() -> dict[str, set[str]]:
    openapi = _OPENAPI_PATH.read_text(encoding="utf-8")
    paths: dict[str, set[str]] = {}
    current_path: str | None = None
    for line in openapi.splitlines():
        if line.startswith("  /") and line.rstrip().endswith(":"):
            current_path = line.strip()[:-1]
            paths[current_path] = set()
        elif current_path is not None and line.startswith("    "):
            method = line.strip().rstrip(":").upper()
            if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                paths[current_path].add(method)
    return paths


def _make_runtime_client(job_db_path: Path) -> TestClient:
    config = Config(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=4,
        ),
    )
    core = Engine(
        config,
        embedding_provider=DemoEmbeddingProvider(dimensions=4),
        sparse_embedder=DemoSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    def core_factory(cfg: Config) -> Engine:
        assert cfg is config
        return core

    app = create_app(
        config=config,
        core_factory=core_factory,
        job_db_path=job_db_path,
        ingest_roots=(job_db_path.parent,),
    )
    return TestClient(app)


@pytest.fixture
def runtime_client(tmp_path: Path) -> TestClient:
    return _make_runtime_client(tmp_path / "jobs.sqlite3")


class _FailingIngestCore:
    async def ensure_ready(self) -> None:
        return None

    async def add_file(
        self,
        path: Path,
        *,
        namespace: str,
        collection: str,
        **_: Any,
    ) -> object:
        raise RuntimeError(
            f"ingest refused for {path.name} in {namespace}/{collection}"
        )

    async def close(self) -> None:
        return None


class _ContextOrderCore:
    def __init__(self) -> None:
        self.retrieve_context_calls: list[dict[str, object]] = []

    async def ensure_ready(self) -> None:
        return None

    async def context(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int,
        content_types: list[str] | None,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        max_chars: int | None,
        max_tokens: int | None,
        audit_context: object | None,
    ) -> object:
        self.retrieve_context_calls.append(
            {
                "query": query,
                "namespace": namespace,
                "collections": collections,
                "limit": limit,
                "content_types": content_types,
                "document_ids": document_ids,
                "rerank": rerank,
                "use_lexical_search": use_lexical_search,
                "max_chars": max_chars,
                "max_tokens": max_tokens,
                "audit_context": audit_context,
            }
        )
        return build_context_pack(
            [
                make_search_result(id="hit-1", text="first", document_id="doc-1"),
                make_search_result(id="hit-2", text="second", document_id="doc-2"),
                make_search_result(id="hit-3", text="third", document_id="doc-3"),
            ],
            query=query,
            max_snippets=limit,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

    async def close(self) -> None:
        return None


def _openapi_schema_block(schema_name: str) -> str:
    openapi = _OPENAPI_PATH.read_text(encoding="utf-8")
    match = re.search(
        rf"^    {re.escape(schema_name)}:\n(?P<body>(?:      .*\n|        .*\n|          .*\n|            .*\n|              .*\n|                .*\n)*)",
        openapi,
        flags=re.MULTILINE,
    )
    assert match is not None, f"schema {schema_name!r} missing from OpenAPI"
    return match.group("body")


def _assert_schema_mentions(schema_name: str, snippets: tuple[str, ...]) -> None:
    block = _openapi_schema_block(schema_name)
    for snippet in snippets:
        assert snippet in block, f"{schema_name} missing {snippet!r}"


def test_demo_embedding_provider_is_registered_for_serve() -> None:
    provider = create_embedding_provider(provider="demo", dimensions=8)
    assert provider.model_name == "demo-dense-v1"
    assert provider.dimensions == 8


def test_serve_cli_job_db_path_flag_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "env-jobs.sqlite3"
    flag_path = tmp_path / "flag-jobs.sqlite3"

    monkeypatch.setenv(DEFAULT_RUNTIME_JOB_DB_PATH_ENV, str(env_path))

    env_args = _build_parser().parse_args(["serve"])
    flag_args = _build_parser().parse_args(["serve", "--job-db-path", str(flag_path)])

    assert env_args.job_db_path == env_path
    assert flag_args.job_db_path == flag_path


def test_serve_cli_job_retention_seconds_flag_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(JOB_RETENTION_SECONDS_ENV, "60.5")

    env_args = _build_parser().parse_args(["serve"])
    flag_args = _build_parser().parse_args(
        ["serve", "--job-retention-seconds", "5"]
    )

    assert env_args.job_retention_seconds == 60.5
    assert flag_args.job_retention_seconds == 5.0


def test_serve_cli_job_retention_seconds_rejects_non_positive() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["serve", "--job-retention-seconds", "0"])


def test_openapi_paths_match_runtime_routes(runtime_client: TestClient) -> None:
    route_methods: dict[str, set[str]] = {}
    app = cast(Any, runtime_client.app)
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if isinstance(path, str) and methods:
            route_methods[path] = {
                method
                for method in methods
                if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}
            }

    assert route_methods == _openapi_methods_by_path()


def test_openapi_declares_runtime_request_and_response_shapes() -> None:
    _assert_schema_mentions(
        "ApiError",
        (
            "- busy",
            "- payload_too_large",
            "max_bytes:",
        ),
    )
    _assert_schema_mentions(
        "IngestRequest",
        (
            "additionalProperties: false",
            "required: [path, collection]",
            "path:",
            'pattern: "\\\\S"',
            "namespace:",
            "collection:",
        ),
    )
    _assert_schema_mentions(
        "IngestJobCreated",
        (
            "required: [job_id, status]",
            "job_id:",
            "status:",
            "enum: [pending, running, completed, failed]",
        ),
    )
    _assert_schema_mentions(
        "IngestJobStatus",
        (
            "required: [job_id, status, path, namespace, collection]",
            "enum: [pending, running, completed, failed]",
            "result:",
            "error:",
            "required: [error_type, error_code]",
            "error_type:",
            "error_code:",
        ),
    )
    _assert_schema_mentions(
        "SearchEndpointRequest",
        (
            "additionalProperties: false",
            "required: [query]",
            "query:",
            "collection:",
            "collections:",
            'pattern: "\\\\S"',
            "limit:",
            "minimum: 1",
            "content_types:",
            "document_ids:",
            f"default: {DEFAULT_RUNTIME_SEARCH_LIMIT}",
            "rerank:",
            "use_lexical_search:",
            "Controls configured lexical/exact-match expansion only",
        ),
    )
    _assert_schema_mentions(
        "ContextRetrievalRequest",
        (
            "additionalProperties: false",
            "required: [query]",
            "query:",
            "collection:",
            "collections:",
            'pattern: "\\\\S"',
            "limit:",
            "minimum: 1",
            "content_types:",
            "document_ids:",
            f"default: {DEFAULT_RUNTIME_CONTEXT_LIMIT}",
            "rerank:",
            "use_lexical_search:",
            "Controls configured lexical/exact-match expansion only",
            "max_chars:",
            "max_tokens:",
            "context_order:",
            "enum: [rank, extrema]",
        ),
    )
    for schema_name in ("SearchEndpointRequest", "ContextRetrievalRequest"):
        block = _openapi_schema_block(schema_name)
        assert "query_plan:" not in block
        assert "search_profile:" not in block

    openapi = _OPENAPI_PATH.read_text(encoding="utf-8")
    assert "#/components/schemas/SearchRequest" not in openapi
    _assert_schema_mentions(
        "SearchHit",
        (
            "required: [id, text, score, content_type, source_type]",
            "metadata:",
            "chunk_index:",
        ),
    )
    _assert_schema_mentions(
        "ContextResponse",
        (
            "- query",
            "- context_text",
            "- snippets",
            "- citations",
            "- source_previews",
            "- citation_summary",
            "- dropped_count",
            "- max_snippets",
            "- max_chars",
            "- max_tokens",
            "- token_estimate",
            "- char_count",
            "- truncated",
            "Prompt-safe text from",
            "App-facing context snippets",
            "App-facing source references",
            "App-facing source previews",
            "App-facing citation summary",
            "not the length of ``context_text``",
        ),
    )
    _assert_schema_mentions(
        "ReadinessResponse",
        (
            "degraded:",
            "event_sink:",
        ),
    )
    _assert_schema_mentions(
        "IngestRequest",
        ("Absolute path",),
    )
    assert "XRequestId" in openapi
    assert "/v1/ingest/{job_id}/events:" in openapi
    assert "#/components/schemas/IngestJobStatus" in openapi
    assert "closes after terminal status" in openapi
    event_stream_match = re.search(
        r"            text/event-stream:\n(?P<body>(?:              .*\n|                .*\n|                  .*\n)*)",
        openapi,
    )
    assert event_stream_match is not None
    event_stream_block = event_stream_match.group("body")
    assert "type: string" in event_stream_block
    assert "event: status" in event_stream_block
    assert "data: {\"job_id\"" in event_stream_block
    assert "#/components/schemas/IngestJobStatus" not in event_stream_block

    _assert_schema_mentions(
        "SearchEndpointRequest",
        (
            "minItems: 1",
            f"maximum: {SEARCH_USER_DOCUMENTS_LIMIT_MAX}",
        ),
    )
    _assert_schema_mentions(
        "ContextRetrievalRequest",
        (
            "minItems: 1",
            f"maximum: {SEARCH_USER_DOCUMENTS_LIMIT_MAX}",
            f"minimum: {SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN}",
            f"maximum: {SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX}",
            f"minimum: {SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN}",
            f"maximum: {SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX}",
        ),
    )


def test_runtime_retrieval_request_defaults_match_route_surface() -> None:
    payload = {"query": "billing", "collection": "help"}

    search_request = parse_retrieval_request(
        payload,
        default_limit=DEFAULT_RUNTIME_SEARCH_LIMIT,
    )
    context_request = parse_retrieval_request(
        payload,
        default_limit=DEFAULT_RUNTIME_CONTEXT_LIMIT,
    )

    assert search_request.limit == DEFAULT_RUNTIME_SEARCH_LIMIT
    assert search_request.namespace == "default"
    assert search_request.collections == ("help",)
    assert search_request.content_types is None
    assert search_request.document_ids is None
    assert search_request.use_lexical_search is True
    assert search_request.max_chars is None
    assert search_request.max_tokens is None
    assert search_request.context_order == "rank"
    assert context_request.limit == DEFAULT_RUNTIME_CONTEXT_LIMIT
    assert context_request.content_types is None
    assert context_request.document_ids is None
    assert context_request.use_lexical_search is True
    assert context_request.max_chars is None
    assert context_request.max_tokens is None
    assert context_request.context_order == "rank"


def test_runtime_retrieval_request_accepts_optional_filters_and_context_budget() -> None:
    payload = {
        "query": "billing",
        "collection": " help ",
        "content_types": [" document "],
        "document_ids": [" doc-1 "],
        "rerank": False,
        "use_lexical_search": False,
        "max_chars": 1200,
        "max_tokens": 256,
        "context_order": "extrema",
    }

    search_request = parse_retrieval_request(
        payload,
        default_limit=DEFAULT_RUNTIME_SEARCH_LIMIT,
        allow_context_budget=True,
    )

    assert search_request.collections == ("help",)
    assert search_request.namespace == "default"
    assert search_request.content_types == ("document",)
    assert search_request.document_ids == ("doc-1",)
    assert search_request.use_lexical_search is False
    assert search_request.max_chars == 1200
    assert search_request.max_tokens == 256
    assert search_request.context_order == "extrema"


def test_runtime_retrieve_context_context_order_extrema_reorders_only_context_text(
    tmp_path: Path,
) -> None:
    config = Config.local()
    core = _ContextOrderCore()
    app = create_app(
        config=config,
        core_factory=lambda _: cast(Any, core),
        job_db_path=tmp_path / "jobs.sqlite3",
        ingest_roots=(tmp_path,),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/search/context",
            json={
                "query": "billing",
                "collection": "help",
                "limit": 3,
                "context_order": "extrema",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert [snippet["rank"] for snippet in payload["snippets"]] == [1, 2, 3]
    assert payload["context_text"].find("[S3]") < payload["context_text"].find("[S2]")
    assert "context_order" not in core.retrieve_context_calls[0]


def test_runtime_retrieve_context_default_and_explicit_rank_output_are_byte_identical(
    tmp_path: Path,
) -> None:
    config = Config.local()
    request_payload = {
        "query": "billing",
        "collection": "help",
        "limit": 3,
    }

    def _post(body: dict[str, object]) -> bytes:
        app = create_app(
            config=config,
            core_factory=lambda _: cast(Any, _ContextOrderCore()),
            job_db_path=tmp_path / f"{body.get('context_order', 'default')}.sqlite3",
            ingest_roots=(tmp_path,),
        )
        with TestClient(app) as client:
            response = client.post("/v1/search/context", json=body)
        assert response.status_code == 200
        return response.content

    assert _post(request_payload) == _post({**request_payload, "context_order": "rank"})


def test_runtime_health_and_runtime_endpoints(runtime_client: TestClient) -> None:
    health = runtime_client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True, "status": "ok", "live": True}

    ready = runtime_client.get("/health/ready")
    assert ready.status_code == 200
    ready_payload = ready.json()
    assert ready_payload["ready"] is True
    assert ready_payload["checks"]["core"]["status"] == "ok"
    assert ready_payload["checks"]["vector_store"]["healthy"] is True
    # Sink failure_count is non-blocking observability: clean run -> 0 +
    # no ``degraded`` flag, but the block is always present so an operator
    # can rely on its shape.
    assert ready_payload["event_sink"]["failure_count"] == 0
    assert ready_payload["event_sink"]["kind"] == "none"
    assert "degraded" not in ready_payload

    runtime = runtime_client.get("/v1/runtime")
    assert runtime.status_code == 200
    payload = runtime.json()
    assert "collection_name" in payload
    assert "retrieval" not in payload
    assert payload["search"]["default_search_profile"] == "balanced"
    assert payload["event_sink"] == {"kind": "none", "failure_count": 0}


def test_runtime_health_ready_unhealthy_body_is_sanitized(tmp_path: Path) -> None:
    """503 readiness body must not leak ``str(exc)``. Only ``error_type`` + ``error_code``."""

    leak_marker = "SDK leak: secret-tail-do-not-echo"

    class _UnhealthyCore:
        async def ensure_ready(self) -> None:
            raise RuntimeError(leak_marker)

        async def close(self) -> None:
            return None

    config = Config.local()
    app = create_app(
        config=config,
        core_factory=lambda _: cast(Any, _UnhealthyCore()),
        job_db_path=tmp_path / "jobs.sqlite3",
        ingest_roots=(tmp_path,),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    payload = response.json()
    assert payload["ready"] is False
    core_check = payload["checks"]["core"]
    assert core_check["status"] == "error"
    assert core_check["error_type"] == "RuntimeError"
    assert core_check["error_code"] == "unhealthy"
    assert "message" not in core_check
    # The exception text must not appear anywhere in the body.
    assert leak_marker not in response.text


def test_runtime_failed_ingest_job_row_has_no_str_exc(tmp_path: Path) -> None:
    """Sanity-check: the SQLite job row must not carry the raw ``str(exc)``."""

    import sqlite3

    job_db_path = tmp_path / "jobs.sqlite3"
    doc = tmp_path / "secret.md"
    doc.write_text("body\n", encoding="utf-8")
    config = Config.local()
    app = create_app(
        config=config,
        core_factory=lambda _: cast(Any, _FailingIngestCore()),
        job_db_path=job_db_path,
        ingest_roots=(tmp_path,),
    )
    with TestClient(app) as client:
        created = client.post(
            "/v1/ingest",
            json={"path": str(doc), "collection": "help"},
        )
        assert created.status_code == 202
        _wait_for_job(client, created.json()["job_id"], terminal_status="failed")

    connection = sqlite3.connect(job_db_path)
    try:
        rows = connection.execute("SELECT error FROM ingest_jobs").fetchall()
    finally:
        connection.close()
    assert rows, "expected at least one job row"
    raw_error = rows[0][0]
    assert raw_error is not None
    # The persisted error column must be the JSON-encoded sanitized dict.
    # ``ingest refused for`` is the literal substring of the SDK-like message
    # produced by ``_FailingIngestCore``; if it ever lands in the SQLite row,
    # the sanitization seam has regressed.
    assert "ingest refused for" not in raw_error
    assert "secret" not in raw_error
    import json as _json

    decoded = _json.loads(raw_error)
    assert decoded == {"error_type": "RuntimeError", "error_code": "ingest_failed"}


def test_runtime_api_error_shape(runtime_client: TestClient) -> None:
    missing_fields = runtime_client.post("/v1/search", json={"query": "billing"})
    assert missing_fields.status_code == 400
    body = missing_fields.json()
    assert body["error"]["code"] == "invalid_request"
    assert "message" in body["error"]

    invalid_json = runtime_client.post(
        "/v1/search",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert invalid_json.status_code == 400
    assert invalid_json.json()["error"]["code"] == "invalid_json"

    invalid_limit = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "limit": True,
        },
    )
    assert invalid_limit.status_code == 400
    assert invalid_limit.json()["error"]["details"] == {"field": "limit"}

    string_limit = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "limit": "5",
        },
    )
    assert string_limit.status_code == 400
    assert string_limit.json()["error"]["details"] == {"field": "limit"}

    invalid_rerank = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "rerank": "false",
        },
    )
    assert invalid_rerank.status_code == 400
    assert invalid_rerank.json()["error"]["details"] == {"field": "rerank"}

    invalid_use_lexical_search = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "use_lexical_search": "false",
        },
    )
    assert invalid_use_lexical_search.status_code == 400
    assert invalid_use_lexical_search.json()["error"]["details"] == {
        "field": "use_lexical_search"
    }

    invalid_document_ids = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "document_ids": [" "],
        },
    )
    assert invalid_document_ids.status_code == 400
    assert invalid_document_ids.json()["error"]["details"] == {"field": "document_ids"}

    invalid_content_types = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "content_types": [" "],
        },
    )
    assert invalid_content_types.status_code == 400
    assert invalid_content_types.json()["error"]["details"] == {"field": "content_types"}

    context_budget_on_search = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "max_chars": 1200,
        },
    )
    assert context_budget_on_search.status_code == 400
    assert context_budget_on_search.json()["error"]["details"] == {
        "fields": ["max_chars"]
    }

    context_order_on_search = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "context_order": "extrema",
        },
    )
    assert context_order_on_search.status_code == 400
    assert context_order_on_search.json()["error"]["details"] == {
        "fields": ["context_order"]
    }

    invalid_context_budget = runtime_client.post(
        "/v1/search/context",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "max_tokens": 0,
        },
    )
    assert invalid_context_budget.status_code == 400
    assert invalid_context_budget.json()["error"]["details"] == {"field": "max_tokens"}

    invalid_context_order = runtime_client.post(
        "/v1/search/context",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "context_order": "middle",
        },
    )
    assert invalid_context_order.status_code == 400
    assert invalid_context_order.json()["error"]["details"] == {
        "field": "context_order"
    }

    invalid_collections = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": [123],
        },
    )
    assert invalid_collections.status_code == 400
    assert invalid_collections.json()["error"]["details"] == {"field": "collections"}

    blank_collection_item = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help", " "],
        },
    )
    assert blank_collection_item.status_code == 400
    assert blank_collection_item.json()["error"]["details"] == {
        "field": "collections"
    }

    camel_case_search = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "corpusIds": ["help"],
        },
    )
    assert camel_case_search.status_code == 400
    assert camel_case_search.json()["error"]["details"] == {
        "fields": ["corpusIds"]
    }

    camel_case_ingest = runtime_client.post(
        "/v1/ingest",
        json={
            "path": "/tmp/nope.md",
            "namespace": "acme",
            "corpusId": "help",
        },
    )
    assert camel_case_ingest.status_code == 400
    assert camel_case_ingest.json()["error"]["details"] == {
        "fields": ["corpusId"]
    }

    unknown_with_canonical = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "corpusIds": ["help"],
        },
    )
    assert unknown_with_canonical.status_code == 400
    assert unknown_with_canonical.json()["error"]["details"] == {
        "fields": ["corpusIds"]
    }

    advanced_controls = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "query_plan": {"preset": "dense_only"},
            "search_profile": "fast",
        },
    )
    assert advanced_controls.status_code == 400
    assert advanced_controls.json()["error"]["details"] == {
        "fields": ["query_plan", "search_profile"]
    }

    blank_scope = runtime_client.post(
        "/v1/search",
        json={
            "query": "   ",
            "namespace": "  ",
            "collections": ["  "],
        },
    )
    assert blank_scope.status_code == 400
    assert blank_scope.json()["error"]["details"] == {
        "field": "collections"
    }

    missing_job = runtime_client.get("/v1/ingest/does-not-exist")
    assert missing_job.status_code == 404
    assert missing_job.json()["error"]["code"] == "not_found"
    assert missing_job.json()["error"]["details"]["job_id"] == "does-not-exist"


def test_runtime_ingest_rejects_paths_outside_ingest_roots(runtime_client: TestClient) -> None:
    response = runtime_client.post(
        "/v1/ingest",
        json={
            "path": "/tmp/outside-runtime-root.md",
            "namespace": "acme",
            "collection": "help",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]["field"] == "path"


def test_runtime_ingest_rejects_missing_paths_inside_ingest_roots(
    runtime_client: TestClient,
    tmp_path: Path,
) -> None:
    response = runtime_client.post(
        "/v1/ingest",
        json={
            "path": str(tmp_path / "missing.md"),
            "namespace": "acme",
            "collection": "help",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["message"] == "path does not exist"
    assert error["details"] == {"field": "path"}


def test_runtime_unknown_route_returns_api_error(runtime_client: TestClient) -> None:
    response = runtime_client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def _wait_for_job(
    client: TestClient,
    job_id: str,
    *,
    terminal_status: str = "completed",
    timeout_s: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(f"/v1/ingest/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        status = payload.get("status")
        if status == terminal_status:
            return dict(payload)
        if status in {"completed", "failed"}:
            pytest.fail(
                f"ingest job reached {status}, expected {terminal_status}: {payload}"
            )
        time.sleep(0.05)
    pytest.fail(f"ingest job did not reach {terminal_status} before timeout: {job_id}")


def test_runtime_ingest_search_and_retrieve_context_journey(
    runtime_client: TestClient,
    tmp_path: Path,
) -> None:
    doc = tmp_path / "billing.md"
    doc.write_text(
        "Invoices are due monthly. Customers can pay by card, ACH, or wire transfer.\n",
        encoding="utf-8",
    )

    created = runtime_client.post(
        "/v1/ingest",
        json={
            "path": str(doc),
            "namespace": "acme",
            "collection": "help",
        },
    )
    assert created.status_code == 202
    created_payload = created.json()
    assert set(created_payload) == {"job_id", "status"}
    assert created_payload["status"] == "pending"
    job_id = created_payload["job_id"]
    finished = _wait_for_job(runtime_client, job_id)
    assert {"job_id", "status", "path", "namespace", "collection", "result"}.issubset(
        finished
    )
    result = finished.get("result")
    assert isinstance(result, dict)
    assert result.get("chunk_count", 0) > 0
    document_id = result.get("document_id")
    assert isinstance(document_id, str)

    search = runtime_client.post(
        "/v1/search",
        json={
            "query": "How can invoices be paid?",
            "namespace": "acme",
            "collections": ["help"],
            "content_types": ["document"],
            "document_ids": [document_id],
            "limit": 5,
            "use_lexical_search": False,
        },
    )
    assert search.status_code == 200
    hits = search.json()
    assert isinstance(hits, list)
    assert hits
    assert "text" in hits[0]
    assert "document_id" in hits[0]
    assert {"id", "text", "score", "content_type", "source_type"}.issubset(hits[0])

    context = runtime_client.post(
        "/v1/search/context",
        json={
            "query": "invoice payment methods",
            "namespace": "acme",
            "collections": ["help"],
            "content_types": ["document"],
            "document_ids": [document_id],
            "limit": 5,
            "use_lexical_search": False,
            "max_chars": 300,
            "max_tokens": 100,
        },
    )
    assert context.status_code == 200
    pack = context.json()
    assert isinstance(pack.get("context_text"), str)
    assert pack["context_text"]
    assert "[S1]" in pack["context_text"]
    snippets = pack["snippets"]
    assert isinstance(snippets, list)
    assert snippets
    structured_source = snippets[0]["source"]
    assert isinstance(structured_source, dict)
    assert "source_id" in structured_source
    assert "result_id" in structured_source
    assert pack["max_tokens"] == 100


def test_runtime_ingest_job_persists_across_app_restart(tmp_path: Path) -> None:
    job_db_path = tmp_path / "jobs.sqlite3"
    doc = tmp_path / "restart.md"
    doc.write_text("Restart persistence keeps ingest job status available.\n", encoding="utf-8")

    first_client = _make_runtime_client(job_db_path)
    created = first_client.post(
        "/v1/ingest",
        json={
            "path": str(doc),
            "namespace": "acme",
            "collection": "help",
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    finished = _wait_for_job(first_client, job_id)
    first_client.close()

    restarted_client = _make_runtime_client(job_db_path)
    reloaded = restarted_client.get(f"/v1/ingest/{job_id}")
    restarted_client.close()

    assert reloaded.status_code == 200
    assert reloaded.json() == finished


def test_runtime_failed_ingest_job_exposes_error_after_restart(tmp_path: Path) -> None:
    job_db_path = tmp_path / "jobs.sqlite3"
    doc = tmp_path / "failure.md"
    doc.write_text("This file intentionally fails in the injected core.\n", encoding="utf-8")
    config = Config.local()

    def make_client() -> TestClient:
        return TestClient(
            create_app(
                config=config,
                core_factory=lambda _: cast(Any, _FailingIngestCore()),
                job_db_path=job_db_path,
                ingest_roots=(tmp_path,),
            )
        )

    first_client = make_client()
    created = first_client.post(
        "/v1/ingest",
        json={
            "path": str(doc),
            "namespace": "acme",
            "collection": "help",
        },
    )
    assert created.status_code == 202
    failed = _wait_for_job(
        first_client,
        created.json()["job_id"],
        terminal_status="failed",
    )
    first_client.close()

    restarted_client = make_client()
    reloaded = restarted_client.get(f"/v1/ingest/{failed['job_id']}")
    restarted_client.close()

    assert failed["status"] == "failed"
    # Public job body is sanitized: only error_type + a stable error_code.
    # The raw exception message is forbidden. It can carry SDK error strings
    # or licensed-source identifiers and must travel via the log/event sink.
    assert failed["error"] == {
        "error_type": "RuntimeError",
        "error_code": "ingest_failed",
    }
    raw_error_text = "ingest refused for failure.md"
    assert raw_error_text not in str(failed)
    assert "result" not in failed
    assert reloaded.status_code == 200
    assert reloaded.json() == failed


class _RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents), top_k))
        return [
            RerankResult(index=index, score=1.0 - (index * 0.1), text=document)
            for index, document in enumerate(documents[:top_k])
        ]


def test_runtime_search_rerank_true_reaches_injected_reranker(tmp_path: Path) -> None:
    reranker = _RecordingReranker()
    config = Config(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=4,
        ),
    )
    core = Engine(
        config,
        embedding_provider=DemoEmbeddingProvider(dimensions=4),
        sparse_embedder=DemoSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        reranker=reranker,
    )

    async def seed() -> None:
        await core.add_bytes(
            file_bytes=b"Billing invoices support ACH and card payments.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="rerank-http",
            collection="docs",
            document_id="billing",
            document_key="billing.txt",
        )

    asyncio.run(seed())

    client = TestClient(
        create_app(
            config=config,
            core_factory=lambda cfg: core,
            job_db_path=tmp_path / "jobs.sqlite3",
        )
    )
    response = client.post(
        "/v1/search",
        json={
            "query": "invoice payment methods",
            "namespace": "rerank-http",
            "collections": ["docs"],
            "limit": 5,
            "rerank": True,
        },
    )
    assert response.status_code == 200
    assert reranker.calls
    assert reranker.calls[0][0] == "invoice payment methods"


def test_runtime_search_returns_hit_list(runtime_client: TestClient) -> None:
    response = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "collections": ["help"],
            "limit": 5,
        },
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_serve_cli_starts_without_nested_event_loop(tmp_path: Path) -> None:
    pytest.importorskip("uvicorn")
    job_db_path = tmp_path / "serve-jobs.sqlite3"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rag_core",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8797",
            "--job-db-path",
            str(job_db_path),
            "--job-retention-seconds",
            "3600",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "64",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                urlopen("http://127.0.0.1:8797/health", timeout=0.5).close()
                break
            except URLError:
                pass
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise AssertionError(
                    f"serve exited early ({proc.returncode}): {stderr}"
                )
            time.sleep(0.2)
        else:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"serve never became healthy: {stderr}")
        assert job_db_path.exists()
    finally:
        proc.terminate()
        proc.wait(timeout=5)
