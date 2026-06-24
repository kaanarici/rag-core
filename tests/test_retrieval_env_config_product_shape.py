from __future__ import annotations

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

from tests.support.source_graph import (
    defining_modules,
    modules_assigning_value,
)

CONFIG_ROOT = "src/rag_core"


def test_embedding_defaults_have_single_config_owner() -> None:
    assert DEFAULT_EMBEDDING_PROVIDER == "openai"
    assert DEMO_EMBEDDING_PROVIDER == "demo"
    assert DEFAULT_EMBEDDING_MODEL == "text-embedding-3-large"
    assert DEMO_EMBEDDING_MODEL == "demo-dense-v1"

    # Each default is bound in exactly one config module. Name-ownership (not a
    # per-file literal scrape) is the durable invariant: the constant lives in
    # one place and every consumer imports it. ``"openai"`` deliberately uses
    # name-ownership rather than value-ownership because the same string is the
    # legitimately separate pip package name (``OPENAI_PACKAGE``).
    for name in (
        "DEFAULT_EMBEDDING_PROVIDER",
        "DEMO_EMBEDDING_PROVIDER",
        "DEFAULT_EMBEDDING_MODEL",
        "DEMO_EMBEDDING_MODEL",
    ):
        assert defining_modules(CONFIG_ROOT, name=name) == {
            "rag_core.config.embedding_config"
        }

    # The demo/default model literals are unique values, so additionally pin that
    # no other module re-types them as a fresh default.
    assert modules_assigning_value(CONFIG_ROOT, value="demo") == {
        "rag_core.config.embedding_config": ["DEMO_EMBEDDING_PROVIDER"]
    }
    assert modules_assigning_value(CONFIG_ROOT, value="text-embedding-3-large") == {
        "rag_core.config.embedding_config": ["DEFAULT_EMBEDDING_MODEL"]
    }
    assert modules_assigning_value(CONFIG_ROOT, value="demo-dense-v1") == {
        "rag_core.config.embedding_config": ["DEMO_EMBEDDING_MODEL"]
    }

    # Retired CLI-specific aliases must not reappear under any module.
    assert defining_modules(CONFIG_ROOT, name="DEFAULT_CLI_EMBEDDING_MODEL") == set()
    assert defining_modules(CONFIG_ROOT, name="DEMO_CLI_EMBEDDING_MODEL") == set()


def test_processing_version_default_has_single_config_owner() -> None:
    assert DEFAULT_PROCESSING_VERSION == "rag_core_processing_v3"
    assert defining_modules(CONFIG_ROOT, name="DEFAULT_PROCESSING_VERSION") == {
        "rag_core.config.ingest_config"
    }
    # The version string is a unique literal; no other module may hardcode it as
    # a fallback default instead of importing the owner constant.
    assert modules_assigning_value(CONFIG_ROOT, value="rag_core_processing_v3") == {
        "rag_core.config.ingest_config": ["DEFAULT_PROCESSING_VERSION"]
    }


def test_ingest_source_type_defaults_have_single_config_owner() -> None:
    assert INGEST_SOURCE_TYPE_FILE == "file"
    assert INGEST_SOURCE_TYPE_URL == "url"
    assert INGEST_SOURCE_TYPE_ARCHIVE == "archive"
    assert DEFAULT_INGEST_SOURCE_TYPE == "file"
    assert STANDARD_INGEST_SOURCE_TYPES == ("file", "url", "archive")

    for name in (
        "INGEST_SOURCE_TYPE_FILE",
        "INGEST_SOURCE_TYPE_URL",
        "INGEST_SOURCE_TYPE_ARCHIVE",
        "DEFAULT_INGEST_SOURCE_TYPE",
        "STANDARD_INGEST_SOURCE_TYPES",
    ):
        assert defining_modules(CONFIG_ROOT, name=name) == {
            "rag_core.config.ingest_config"
        }

    # The source-type string literals are unique to their owning constants; no
    # other module may re-type ``"file"``/``"url"``/``"archive"`` as a default
    # (the old test froze this via a hand-picked file list + forbidden snippets).
    assert modules_assigning_value(CONFIG_ROOT, value="file") == {
        "rag_core.config.ingest_config": ["INGEST_SOURCE_TYPE_FILE"]
    }
    assert modules_assigning_value(CONFIG_ROOT, value="url") == {
        "rag_core.config.ingest_config": ["INGEST_SOURCE_TYPE_URL"]
    }
    assert modules_assigning_value(CONFIG_ROOT, value="archive") == {
        "rag_core.config.ingest_config": ["INGEST_SOURCE_TYPE_ARCHIVE"]
    }


def test_ingest_max_concurrency_default_has_single_config_owner() -> None:
    assert DEFAULT_INGEST_MAX_CONCURRENCY == 1
    # ``1`` is a ubiquitous literal, so name-ownership is the durable invariant:
    # the default lives in one config module and consumers import it rather than
    # re-typing ``max_concurrency: int = 1`` / ``default=1``.
    assert defining_modules(CONFIG_ROOT, name="DEFAULT_INGEST_MAX_CONCURRENCY") == {
        "rag_core.config.ingest_config"
    }


def test_cli_manifest_directory_default_has_single_config_owner() -> None:
    assert DEFAULT_CLI_MANIFEST_DIRECTORY == ".rag-core/manifest"
    assert defining_modules(CONFIG_ROOT, name="DEFAULT_CLI_MANIFEST_DIRECTORY") == {
        "rag_core.config.ingest_config"
    }
    # Unique path literal: no other module may hardcode the manifest directory.
    assert modules_assigning_value(CONFIG_ROOT, value=".rag-core/manifest") == {
        "rag_core.config.ingest_config": ["DEFAULT_CLI_MANIFEST_DIRECTORY"]
    }
    # The retired alias name must not reappear anywhere.
    assert defining_modules(CONFIG_ROOT, name="DEFAULT_MANIFEST_DIRECTORY") == set()
