from __future__ import annotations

from pathlib import Path

from rag_core.config import (
    DEFAULT_CLI_MANIFEST_DIRECTORY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INGEST_MAX_CONCURRENCY,
    DEFAULT_INGEST_SOURCE_TYPE,
    DEMO_EMBEDDING_MODEL,
    DEMO_EMBEDDING_PROVIDER,
    INGEST_SOURCE_TYPE_ARCHIVE,
    INGEST_SOURCE_TYPE_FILE,
    INGEST_SOURCE_TYPE_URL,
    STANDARD_INGEST_SOURCE_TYPES,
)
from rag_core.core_models import DEFAULT_PROCESSING_VERSION

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_embedding_defaults_have_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/embedding_config.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "src/rag_core/search/providers/embedding.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "tests/support/fakes.py",
        )
    }

    assert DEFAULT_EMBEDDING_PROVIDER == "openai"
    assert DEMO_EMBEDDING_PROVIDER == "demo"
    assert DEFAULT_EMBEDDING_MODEL == "text-embedding-3-large"
    assert DEMO_EMBEDDING_MODEL == "demo-dense-v1"
    assert (
        sources["src/rag_core/config/embedding_config.py"].count(
            'DEFAULT_EMBEDDING_PROVIDER = "openai"'
        )
        == 1
    )
    assert (
        sources["src/rag_core/config/embedding_config.py"].count(
            'DEMO_EMBEDDING_PROVIDER = "demo"'
        )
        == 1
    )
    assert (
        sources["src/rag_core/config/embedding_config.py"].count(
            'DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"'
        )
        == 1
    )
    assert (
        sources["src/rag_core/config/embedding_config.py"].count(
            'DEMO_EMBEDDING_MODEL = "demo-dense-v1"'
        )
        == 1
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/embedding_config.py":
            continue
        assert 'model: str = "text-embedding-3-large"' not in source
        assert 'model: str = "demo-dense-v1"' not in source
        assert 'provider: str = "openai"' not in source
        assert 'or "openai"' not in source
        assert "DEFAULT_CLI_EMBEDDING_MODEL" not in source
        assert "DEMO_CLI_EMBEDDING_MODEL" not in source





def test_processing_version_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "src/rag_core/_engine/core_runtime.py",
            "tests/support/fakes.py",
        )
    }

    assert DEFAULT_PROCESSING_VERSION == "rag_core_processing_v3"
    assert (
        sources["src/rag_core/config/ingest_config.py"].count(
            'DEFAULT_PROCESSING_VERSION = "rag_core_processing_v3"'
        )
        == 1
    )
    forbidden_default_copies = (
        'env_or_default(PROCESSING_VERSION_ENV, "rag_core_processing_v3")',
        'base_version = (configured_version or "").strip() or "rag_core_processing_v3"',
        'processing_version: str = "rag_core_processing_v3"',
        'default="rag_core_processing_v3"',
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/ingest_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source





def test_ingest_source_type_defaults_have_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/cli_archive.py",
            "src/rag_core/cli_url_ingest.py",
            "src/rag_core/_engine/core_archive_runner.py",
            "src/rag_core/_engine/core_builders.py",
            "src/rag_core/_engine/core_remote.py",
            "src/rag_core/_engine/core_runtime.py",
            "src/rag_core/facade/ingest_sources.py",
            "src/rag_core/local_ingest.py",
            "src/rag_core/remote_ingest_models.py",
            "src/rag_core/remote_sources.py",
            "src/rag_core/search/indexer_points.py",
            "src/rag_core/search/indexer_texts.py",
            "src/rag_core/_engine/core_ingest_identity.py",
            "src/rag_core/cli_doctor_output.py",
            "tests/support/fakes.py",
        )
    }

    assert INGEST_SOURCE_TYPE_FILE == "file"
    assert INGEST_SOURCE_TYPE_URL == "url"
    assert INGEST_SOURCE_TYPE_ARCHIVE == "archive"
    assert DEFAULT_INGEST_SOURCE_TYPE == "file"
    assert STANDARD_INGEST_SOURCE_TYPES == ("file", "url", "archive")
    owner = sources["src/rag_core/config/ingest_config.py"]
    assert owner.count('INGEST_SOURCE_TYPE_FILE = "file"') == 1
    assert owner.count('INGEST_SOURCE_TYPE_URL = "url"') == 1
    assert owner.count('INGEST_SOURCE_TYPE_ARCHIVE = "archive"') == 1
    assert "DEFAULT_INGEST_SOURCE_TYPE = INGEST_SOURCE_TYPE_FILE" in owner
    forbidden_default_copies = (
        'source_type="file"',
        'source_type="url"',
        'source_type="archive"',
        '"source_type": "url"',
        '"source_type": "archive"',
        'source_type: str = "file"',
        'source_type: str = "url"',
        'req.source_type == "url"',
        'or "file"',
        '{"file", "archive"}',
        '("file", "url", "archive")',
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/ingest_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source





def test_ingest_max_concurrency_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/cli_ingest_parser.py",
            "src/rag_core/cli_source_flags.py",
            "src/rag_core/_engine/core_archive_ingest.py",
            "src/rag_core/_engine/core_batch_ingest.py",
            "src/rag_core/facade/ingest.py",
            "src/rag_core/facade/ingest_batches.py",
            "src/rag_core/local_ingest_models.py",
            "src/rag_core/remote_ingest_models.py",
        )
    }

    assert DEFAULT_INGEST_MAX_CONCURRENCY == 1
    assert (
        sources["src/rag_core/config/ingest_config.py"].count(
            "DEFAULT_INGEST_MAX_CONCURRENCY = 1"
        )
        == 1
    )
    assert (
        "Default: {DEFAULT_INGEST_MAX_CONCURRENCY}."
        in sources["src/rag_core/cli_source_flags.py"]
    )
    forbidden_default_copies = (
        "max_concurrency: int = 1",
        "default=1",
        "Default: 1.",
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/ingest_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source





def test_cli_manifest_directory_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/config/__init__.py",
            "src/rag_core/cli_parser.py",
            "src/rag_core/cli_source_flags.py",
        )
    }

    assert DEFAULT_CLI_MANIFEST_DIRECTORY == ".rag-core/manifest"
    assert (
        sources["src/rag_core/config/ingest_config.py"].count(
            'DEFAULT_CLI_MANIFEST_DIRECTORY = ".rag-core/manifest"'
        )
        == 1
    )
    assert (
        "manifest_directory: Path | None = None"
        in sources["src/rag_core/config/ingest_config.py"]
    )
    assert "DEFAULT_CLI_MANIFEST_DIRECTORY" in sources["src/rag_core/cli_parser.py"]
    assert "DEFAULT_MANIFEST_DIRECTORY" not in sources["src/rag_core/cli_parser.py"]
    forbidden_default_copies = (
        '".rag-core/manifest"',
        "'.rag-core/manifest'",
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/ingest_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source
