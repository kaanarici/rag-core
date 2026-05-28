from __future__ import annotations

from pathlib import Path

from rag_core.remote_discovery_models import (
    DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    DEFAULT_REMOTE_SITEMAP_MAX_URLS,
    REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT,
    REMOTE_DISCOVERY_CLI_KIND_SITEMAP,
    REMOTE_DISCOVERY_KIND_LLMS_TXT,
    REMOTE_DISCOVERY_KIND_SITEMAP,
    REMOTE_DISCOVERY_KIND_SITEMAP_INDEX,
)

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


def test_remote_discovery_budget_defaults_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/remote_discovery_models.py",
            "src/rag_core/remote_discovery.py",
            "src/rag_core/remote_discovery_documents.py",
            "src/rag_core/cli_remote.py",
            "src/rag_core/cli_remote_fetch.py",
        )
    }

    assert DEFAULT_REMOTE_SITEMAP_MAX_URLS == 50_000
    assert DEFAULT_REMOTE_LLMS_TXT_MAX_URLS == 1_000
    assert DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES == 128
    owner = sources["src/rag_core/remote_discovery_models.py"]
    assert owner.count("DEFAULT_REMOTE_SITEMAP_MAX_URLS: Final[int] = 50_000") == 1
    assert owner.count("DEFAULT_REMOTE_LLMS_TXT_MAX_URLS: Final[int] = 1_000") == 1
    assert (
        owner.count("DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES: Final[int] = 128") == 1
    )
    for path, source in sources.items():
        if path == "src/rag_core/remote_discovery_models.py":
            continue
        assert "50_000" not in source
        assert "1_000" not in source
        assert "resolved = 128" not in source
        assert "max_sitemap_fetches: int = 128" not in source




def test_remote_discovery_kind_labels_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/remote_discovery_models.py",
            "src/rag_core/remote_discovery.py",
            "src/rag_core/remote_discovery_documents.py",
            "src/rag_core/remote_discovery_sitemaps.py",
            "src/rag_core/cli_remote.py",
            "src/rag_core/cli_remote_parser.py",
            "src/rag_core/cli_remote_output.py",
        )
    }

    assert REMOTE_DISCOVERY_KIND_SITEMAP == "sitemap"
    assert REMOTE_DISCOVERY_KIND_SITEMAP_INDEX == "sitemap_index"
    assert REMOTE_DISCOVERY_KIND_LLMS_TXT == "llms_txt"
    assert REMOTE_DISCOVERY_CLI_KIND_SITEMAP == "sitemap"
    assert REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT == "llms-txt"

    owner = sources["src/rag_core/remote_discovery_models.py"]
    for definition in (
        'REMOTE_DISCOVERY_KIND_SITEMAP: Final[RemoteDiscoveryKind] = "sitemap"',
        (
            "REMOTE_DISCOVERY_KIND_SITEMAP_INDEX: "
            'Final[RemoteDiscoveryKind] = "sitemap_index"'
        ),
        'REMOTE_DISCOVERY_KIND_LLMS_TXT: Final[RemoteDiscoveryKind] = "llms_txt"',
        'REMOTE_DISCOVERY_CLI_KIND_SITEMAP: Final[str] = "sitemap"',
        'REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT: Final[str] = "llms-txt"',
    ):
        assert owner.count(definition) == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/remote_discovery_models.py"
    )
    for symbol in (
        "REMOTE_DISCOVERY_KIND_SITEMAP",
        "REMOTE_DISCOVERY_KIND_SITEMAP_INDEX",
        "REMOTE_DISCOVERY_KIND_LLMS_TXT",
        "REMOTE_DISCOVERY_CLI_KIND_SITEMAP",
        "REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT",
    ):
        assert symbol in consumers
    for duplicate in (
        'source_kind="sitemap"',
        'source_kind="sitemap_index"',
        'source_kind="llms_txt"',
        'source_kind != "sitemap_index"',
        'source_kind == "sitemap_index"',
        'kind == "llms_txt"',
        'kind == "sitemap_index"',
        'args.kind == "sitemap"',
        'args.kind == "llms-txt"',
        'choices=("sitemap", "llms-txt")',
    ):
        assert duplicate not in consumers
