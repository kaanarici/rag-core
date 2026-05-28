from pathlib import Path

from rag_core.config import (
    QDRANT_COLLECTION_ENV,
    QDRANT_DIMENSION_AWARE_COLLECTION_ENV,
    QDRANT_LOCATION_ENV,
    QDRANT_URL_ENV,
    TURBOPUFFER_BASE_URL_ENV,
    TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV,
    TURBOPUFFER_DISTANCE_METRIC_ENV,
    TURBOPUFFER_NAMESPACE_ENV,
    TURBOPUFFER_REGION_ENV,
    VECTOR_STORE_ENV,
)
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_HOST,
    DEFAULT_RUNTIME_JOB_DB_PATH,
    DEFAULT_RUNTIME_JOB_DB_PATH_ENV,
    DEFAULT_RUNTIME_PORT,
)
from rag_core.runtime.jobs import (
    INGEST_JOB_STATUSES,
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
    INGEST_JOB_STATUS_PENDING,
    INGEST_JOB_STATUS_RUNNING,
)


def test_serve_ingest_root_docs_match_allowlist_behavior() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    self_host = (root / "docs" / "self-host.md").read_text(encoding="utf-8")
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    ci_self_host_smoke = (root / "scripts" / "ci_self_host_smoke.sh").read_text(
        encoding="utf-8"
    )
    normalized_self_host = " ".join(self_host.replace("\\\n", " ").split())
    parser = (root / "src" / "rag_core" / "cli_serve_parser.py").read_text(
        encoding="utf-8"
    )

    assert "limited to the working directory by default" in readme
    assert "only the configured roots are allowed" in readme
    assert "allowlist becomes the explicit roots you configured" in normalized_self_host
    assert "default Compose file uses demo embeddings and Qdrant" in self_host
    assert "Copy `.env.example` to `.env` for OpenAI embeddings" not in self_host
    assert "current working directory when omitted" in parser
    assert "allow other server-local roots" not in self_host
    assert "import os\n\nfrom starlette.middleware.base" in self_host
    assert 'os.environ["RAG_CORE_API_KEY"]' in self_host
    assert "HTTP probe for an already-running `rag-core serve`" in scripts_readme
    assert "HTTP against `docker compose`" not in scripts_readme
    assert "RAG_CORE_RUNTIME_JOB_DB_PATH" in self_host
    assert "--job-db-path" in self_host
    assert "RAG_CORE_RUNTIME_JOB_DB_PATH" in env_example
    assert "--job-db-path \"$JOB_DB_PATH\"" in ci_self_host_smoke
    assert "COPY examples/demo_corpus ./examples/demo_corpus" in dockerfile
    assert '"/app/examples/demo_corpus"' in dockerfile
    assert "- /app/examples/demo_corpus" in compose


def test_wheel_smoke_exercises_installed_runtime_extra() -> None:
    source = Path("scripts/wheel_smoke.py").read_text(encoding="utf-8")

    assert "def _installed_runtime_smoke(" in source
    assert 'extras=("runtime",)' in source
    assert '"serve"' in source
    assert '"--qdrant-location"' in source
    assert '":memory:"' in source
    assert '"--embedding-provider"' in source
    assert '"demo"' in source
    assert "scripts\" / \"self_host_smoke.sh" in source
    assert "installed runtime extra smoke passed" in source


def test_runtime_serve_defaults_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/runtime_defaults.py",
            "src/rag_core/cli_serve_parser.py",
            "src/rag_core/cli_serve.py",
            "src/rag_core/runtime/app.py",
        )
    }

    assert DEFAULT_RUNTIME_HOST == "127.0.0.1"
    assert DEFAULT_RUNTIME_PORT == 8787
    assert DEFAULT_RUNTIME_JOB_DB_PATH.as_posix() == ".rag-core/runtime/jobs.sqlite3"
    assert DEFAULT_RUNTIME_JOB_DB_PATH_ENV == "RAG_CORE_RUNTIME_JOB_DB_PATH"
    owner = sources["src/rag_core/runtime_defaults.py"]
    assert owner.count('DEFAULT_RUNTIME_HOST: Final[str] = "127.0.0.1"') == 1
    assert owner.count("DEFAULT_RUNTIME_PORT: Final[int] = 8787") == 1
    assert (
        owner.count(
            'DEFAULT_RUNTIME_JOB_DB_PATH: Final[Path] = Path(".rag-core/runtime/jobs.sqlite3")'
        )
        == 1
    )
    assert (
        owner.count(
            'DEFAULT_RUNTIME_JOB_DB_PATH_ENV: Final[str] = "RAG_CORE_RUNTIME_JOB_DB_PATH"'
        )
        == 1
    )
    for path, source in sources.items():
        if path == "src/rag_core/runtime_defaults.py":
            continue
        assert 'default="127.0.0.1"' not in source
        assert "default=8787" not in source
        assert '".rag-core/runtime/jobs.sqlite3"' not in source
        assert '"RAG_CORE_RUNTIME_JOB_DB_PATH"' not in source
    parser = sources["src/rag_core/cli_serve_parser.py"]
    assert "--job-db-path" in parser
    assert "DEFAULT_RUNTIME_JOB_DB_PATH_ENV" in parser
    assert "from rag_core.runtime." not in sources["src/rag_core/cli_serve_parser.py"]


def test_runtime_ingest_job_statuses_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/runtime/jobs.py",
            "src/rag_core/runtime/app.py",
        )
    }

    assert INGEST_JOB_STATUS_PENDING == "pending"
    assert INGEST_JOB_STATUS_RUNNING == "running"
    assert INGEST_JOB_STATUS_COMPLETED == "completed"
    assert INGEST_JOB_STATUS_FAILED == "failed"
    assert INGEST_JOB_STATUSES == (
        INGEST_JOB_STATUS_PENDING,
        INGEST_JOB_STATUS_RUNNING,
        INGEST_JOB_STATUS_COMPLETED,
        INGEST_JOB_STATUS_FAILED,
    )

    owner = sources["src/rag_core/runtime/jobs.py"]
    for definition in (
        'INGEST_JOB_STATUS_PENDING: Final[JobStatus] = "pending"',
        'INGEST_JOB_STATUS_RUNNING: Final[JobStatus] = "running"',
        'INGEST_JOB_STATUS_COMPLETED: Final[JobStatus] = "completed"',
        'INGEST_JOB_STATUS_FAILED: Final[JobStatus] = "failed"',
        "INGEST_JOB_STATUSES: Final[tuple[JobStatus, ...]] = (",
        "def parse_job_status(value: str) -> JobStatus:",
    ):
        assert owner.count(definition) == 1

    consumer = sources["src/rag_core/runtime/app.py"]
    for symbol in (
        "INGEST_JOB_STATUS_RUNNING",
        "INGEST_JOB_STATUS_COMPLETED",
        "INGEST_JOB_STATUS_FAILED",
    ):
        assert symbol in consumer
    for duplicate in (
        'status="running"',
        'status="completed"',
        'status="failed"',
        'status = "running"',
        'status = "completed"',
        'status = "failed"',
    ):
        assert duplicate not in consumer


def test_self_host_docs_cover_selectable_vector_store_config() -> None:
    root = Path(__file__).resolve().parents[1]
    self_host = (root / "docs" / "self-host.md").read_text(encoding="utf-8")
    config_sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/core_config_cli.py",
            "src/rag_core/config/embedding_config.py",
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/config/qdrant_config.py",
            "src/rag_core/config/reranker_config.py",
            "src/rag_core/config/vector_store_config.py",
        )
    )

    for term in (
        "RAG_CORE_VECTOR_STORE",
        "--vector-store",
        "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION",
        "--dimension-aware-collection",
        "RAG_CORE_TURBOPUFFER_NAMESPACE",
        "--turbopuffer-namespace",
        "TURBOPUFFER_API_KEY",
        "--turbopuffer-api-key",
        "TURBOPUFFER_REGION",
        "--turbopuffer-region",
        "TURBOPUFFER_BASE_URL",
        "--turbopuffer-base-url",
        "RAG_CORE_TURBOPUFFER_DISTANCE_METRIC",
        "--turbopuffer-distance-metric",
        "RAG_CORE_TURBOPUFFER_DELETE_CONTINUATION_LIMIT",
        "--turbopuffer-delete-continuation-limit",
        "RAG_CORE_EMBEDDING_BATCH_SIZE",
        "--embedding-batch-size",
        "RAG_CORE_RERANKER_MODEL",
        "--reranker-model",
    ):
        assert term in self_host
        assert term in config_sources

    assert "Physical TurboPuffer namespace" in self_host
    assert "request `namespace` tenancy" in self_host
    assert "--no-dimension-aware-collection" in self_host
    assert "BooleanOptionalAction" in config_sources
    assert "document embedding batch size" in self_host


def test_vector_store_env_names_have_config_owners() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/qdrant_config.py",
            "src/rag_core/config/vector_store_config.py",
            "src/rag_core/config/__init__.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/core_config_cli.py",
            "src/rag_core/core_runtime.py",
            "src/rag_core/search/providers/vector_store_diagnostics.py",
        )
    }

    assert VECTOR_STORE_ENV == "RAG_CORE_VECTOR_STORE"
    assert QDRANT_URL_ENV == "RAG_CORE_QDRANT_URL"
    assert QDRANT_LOCATION_ENV == "RAG_CORE_QDRANT_LOCATION"
    assert QDRANT_COLLECTION_ENV == "RAG_CORE_QDRANT_COLLECTION"
    assert (
        QDRANT_DIMENSION_AWARE_COLLECTION_ENV
        == "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION"
    )
    assert TURBOPUFFER_NAMESPACE_ENV == "RAG_CORE_TURBOPUFFER_NAMESPACE"
    assert TURBOPUFFER_REGION_ENV == "TURBOPUFFER_REGION"
    assert TURBOPUFFER_BASE_URL_ENV == "TURBOPUFFER_BASE_URL"
    assert TURBOPUFFER_DISTANCE_METRIC_ENV == "RAG_CORE_TURBOPUFFER_DISTANCE_METRIC"
    assert (
        TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV
        == "RAG_CORE_TURBOPUFFER_DELETE_CONTINUATION_LIMIT"
    )

    qdrant_owner = sources["src/rag_core/config/qdrant_config.py"]
    assert qdrant_owner.count('QDRANT_URL_ENV = "RAG_CORE_QDRANT_URL"') == 1
    assert qdrant_owner.count('QDRANT_LOCATION_ENV = "RAG_CORE_QDRANT_LOCATION"') == 1
    assert (
        qdrant_owner.count('QDRANT_COLLECTION_ENV = "RAG_CORE_QDRANT_COLLECTION"') == 1
    )
    assert (
        qdrant_owner.count(
            'QDRANT_DIMENSION_AWARE_COLLECTION_ENV = "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION"'
        )
        == 1
    )
    vector_owner = sources["src/rag_core/config/vector_store_config.py"]
    assert vector_owner.count('VECTOR_STORE_ENV = "RAG_CORE_VECTOR_STORE"') == 1
    assert (
        vector_owner.count(
            'TURBOPUFFER_NAMESPACE_ENV = "RAG_CORE_TURBOPUFFER_NAMESPACE"'
        )
        == 1
    )
    assert vector_owner.count('TURBOPUFFER_REGION_ENV = "TURBOPUFFER_REGION"') == 1
    assert vector_owner.count('TURBOPUFFER_BASE_URL_ENV = "TURBOPUFFER_BASE_URL"') == 1
    assert (
        vector_owner.count(
            'TURBOPUFFER_DISTANCE_METRIC_ENV = "RAG_CORE_TURBOPUFFER_DISTANCE_METRIC"'
        )
        == 1
    )
    assert vector_owner.count("TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV = (") == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path
        not in {
            "src/rag_core/config/qdrant_config.py",
            "src/rag_core/config/vector_store_config.py",
        }
    )
    for symbol in (
        "VECTOR_STORE_ENV",
        "QDRANT_URL_ENV",
        "QDRANT_LOCATION_ENV",
        "QDRANT_COLLECTION_ENV",
        "QDRANT_DIMENSION_AWARE_COLLECTION_ENV",
        "TURBOPUFFER_NAMESPACE_ENV",
        "TURBOPUFFER_REGION_ENV",
        "TURBOPUFFER_BASE_URL_ENV",
        "TURBOPUFFER_DISTANCE_METRIC_ENV",
        "TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV",
    ):
        assert symbol in consumers
    for duplicate in (
        '"RAG_CORE_VECTOR_STORE"',
        '"RAG_CORE_QDRANT_URL"',
        '"RAG_CORE_QDRANT_LOCATION"',
        '"RAG_CORE_QDRANT_COLLECTION"',
        '"RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION"',
        '"RAG_CORE_TURBOPUFFER_NAMESPACE"',
        '"TURBOPUFFER_REGION"',
        '"TURBOPUFFER_BASE_URL"',
        '"RAG_CORE_TURBOPUFFER_DISTANCE_METRIC"',
        '"RAG_CORE_TURBOPUFFER_DELETE_CONTINUATION_LIMIT"',
    ):
        assert duplicate not in consumers
