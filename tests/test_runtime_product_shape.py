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
from rag_core.runtime.jobs import (
    INGEST_JOB_STATUSES,
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
    INGEST_JOB_STATUS_PENDING,
    INGEST_JOB_STATUS_RUNNING,
    parse_job_status,
)
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_HOST,
    DEFAULT_RUNTIME_JOB_DB_PATH,
    DEFAULT_RUNTIME_JOB_DB_PATH_ENV,
    DEFAULT_RUNTIME_PORT,
)

from tests.support.source_graph import (
    defining_modules,
    iter_package_sources,
    modules_assigning_value,
    modules_importing,
    symbol_module,
    under_module,
)


def test_serve_ingest_root_docs_match_allowlist_behavior() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    self_host = (root / "docs-site" / "content" / "docs" / "self-host.mdx").read_text(
        encoding="utf-8"
    )
    scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    ci_self_host_smoke = (root / "scripts" / "ci_self_host_smoke.sh").read_text(
        encoding="utf-8"
    )
    normalized_self_host = " ".join(self_host.replace("\\\n", " ").split())
    parser = (
        root / "src" / "rag_core" / "cli" / "parsers" / "serve.py"
    ).read_text(encoding="utf-8")

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
    assert "RAG_CORE_RUNTIME_JOB_RETENTION_SECONDS" in self_host
    assert "--job-retention-seconds" in self_host
    assert "--job-retention-seconds" in parser
    assert "terminal statuses only" in self_host
    assert "not resumed after process restart" in normalized_self_host
    assert "Pending and running rows are never pruned by retention." in self_host
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
    assert DEFAULT_RUNTIME_HOST == "127.0.0.1"
    assert DEFAULT_RUNTIME_PORT == 8787
    assert DEFAULT_RUNTIME_JOB_DB_PATH.as_posix() == ".rag-core/runtime/jobs.sqlite3"
    assert DEFAULT_RUNTIME_JOB_DB_PATH_ENV == "RAG_CORE_RUNTIME_JOB_DB_PATH"

    # Each serve default is owned by exactly one module and no consumer re-types
    # the literal (e.g. ``default="127.0.0.1"`` / ``default=8787`` / the db path).
    # The literal's single owning module is the durable form of the old per-file
    # ``count()`` + ``literal not in source`` scrape over a hand-pinned list.
    assert modules_assigning_value("src/rag_core", value="127.0.0.1") == {
        "rag_core.runtime_defaults": ["DEFAULT_RUNTIME_HOST"]
    }
    assert modules_assigning_value("src/rag_core", value=8787) == {
        "rag_core.runtime_defaults": ["DEFAULT_RUNTIME_PORT"]
    }
    assert modules_assigning_value(
        "src/rag_core", value=".rag-core/runtime/jobs.sqlite3"
    ) == {"rag_core.runtime_defaults": ["DEFAULT_RUNTIME_JOB_DB_PATH"]}
    assert modules_assigning_value(
        "src/rag_core", value="RAG_CORE_RUNTIME_JOB_DB_PATH"
    ) == {"rag_core.runtime_defaults": ["DEFAULT_RUNTIME_JOB_DB_PATH_ENV"]}

    # The serve CLI parser stays importable by the base CLI: it may read defaults
    # from ``rag_core.runtime_defaults`` but must not pull in the heavy
    # ``rag_core.runtime`` package.
    assert (
        modules_importing(
            "src/rag_core/cli/parsers", predicate=under_module("rag_core.runtime")
        )
        == {}
    )

    # The serve parser still wires the job-db-path flag and reads the env name
    # from the owner constant rather than re-typing it.
    parsers = "\n".join(src for _, _, src in iter_package_sources("src/rag_core/cli/parsers"))
    assert "--job-db-path" in parsers
    assert "DEFAULT_RUNTIME_JOB_DB_PATH_ENV" in parsers


def test_runtime_ingest_job_statuses_have_single_owner() -> None:
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

    # Each status string is bound once, in the jobs module; no consumer re-types
    # ``status="running"`` / ``status = "completed"`` etc. instead of importing
    # the constant. Pinning the literal's owning module replaces the old
    # ``owner.count(...) == 1`` + ``status="..." not in consumer`` scrape.
    for literal, name in (
        ("pending", "INGEST_JOB_STATUS_PENDING"),
        ("running", "INGEST_JOB_STATUS_RUNNING"),
        ("completed", "INGEST_JOB_STATUS_COMPLETED"),
        ("failed", "INGEST_JOB_STATUS_FAILED"),
    ):
        assert modules_assigning_value("src/rag_core/runtime", value=literal) == {
            "rag_core.runtime.jobs": [name]
        }

    # The status parser is owned by the jobs module and defined nowhere else.
    assert symbol_module(parse_job_status) == "rag_core.runtime.jobs"
    assert defining_modules("src/rag_core", name="parse_job_status") == {
        "rag_core.runtime.jobs"
    }


def test_self_host_docs_cover_selectable_vector_store_config() -> None:
    root = Path(__file__).resolve().parents[1]
    self_host = (root / "docs-site" / "content" / "docs" / "self-host.mdx").read_text(
        encoding="utf-8"
    )
    config_sources = "\n".join(
        src
        for _, _, src in iter_package_sources(
            "src/rag_core/config",
            "src/rag_core/cli",
            "src/rag_core/_engine",
            "src/rag_core/search",
        )
    )

    for term in (
        "RAG_CORE_VECTOR_STORE",
        "--store",
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
        assert term in config_sources
        if term != "--store":
            assert term in self_host

    assert "Physical TurboPuffer namespace" in self_host
    assert "request `namespace` tenancy" in self_host
    assert "--no-dimension-aware-collection" in self_host
    assert "BooleanOptionalAction" in config_sources
    assert "document embedding batch size" in self_host


def test_vector_store_env_names_have_config_owners() -> None:
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

    # Each vector-store env literal is assigned in exactly one config module;
    # consumers import the named constant. Owning-module-of-the-literal replaces
    # the old per-file ``count()`` + ``literal not in consumers`` scrape.
    owners = {
        VECTOR_STORE_ENV: ("rag_core.config.vector_store_config", "VECTOR_STORE_ENV"),
        QDRANT_URL_ENV: ("rag_core.config.qdrant_config", "QDRANT_URL_ENV"),
        QDRANT_LOCATION_ENV: ("rag_core.config.qdrant_config", "QDRANT_LOCATION_ENV"),
        QDRANT_COLLECTION_ENV: (
            "rag_core.config.qdrant_config",
            "QDRANT_COLLECTION_ENV",
        ),
        QDRANT_DIMENSION_AWARE_COLLECTION_ENV: (
            "rag_core.config.qdrant_config",
            "QDRANT_DIMENSION_AWARE_COLLECTION_ENV",
        ),
        TURBOPUFFER_NAMESPACE_ENV: (
            "rag_core.config.vector_store_config",
            "TURBOPUFFER_NAMESPACE_ENV",
        ),
        TURBOPUFFER_REGION_ENV: (
            "rag_core.config.vector_store_config",
            "TURBOPUFFER_REGION_ENV",
        ),
        TURBOPUFFER_BASE_URL_ENV: (
            "rag_core.config.vector_store_config",
            "TURBOPUFFER_BASE_URL_ENV",
        ),
        TURBOPUFFER_DISTANCE_METRIC_ENV: (
            "rag_core.config.vector_store_config",
            "TURBOPUFFER_DISTANCE_METRIC_ENV",
        ),
        TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV: (
            "rag_core.config.vector_store_config",
            "TURBOPUFFER_DELETE_CONTINUATION_LIMIT_ENV",
        ),
    }
    for literal, (owner_module, owner_name) in owners.items():
        assert modules_assigning_value("src/rag_core", value=literal) == {
            owner_module: [owner_name]
        }
