from __future__ import annotations

from pathlib import Path

from rag_core.search.providers.diagnostic_support import (
    PROVIDER_DIAGNOSTIC_READINESS_SCOPES,
    READINESS_PACKAGE_AND_ENV,
)
from rag_core.search.providers.sparse import (
    DEFAULT_SPARSE_EMBEDDER_PROVIDER,
    SPARSE_EMBEDDER_PROVIDER_ORDER,
    SPARSE_LOAD_DISABLED,
    SPARSE_LOAD_FAILED,
    SPARSE_LOAD_LOADED,
    SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR,
    SPARSE_LOAD_NOT_LOADED,
    SPLADE_LOAD_UNKNOWN_UNTIL_RUN,
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


def test_sparse_provider_default_has_single_provider_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/cli_doctor.py",
            "src/rag_core/cli_doctor_output.py",
            "src/rag_core/search/providers/diagnostic_support.py",
            "src/rag_core/search/providers/sparse.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/search/providers/provider_category_helpers.py",
            "tests/test_cli.py",
            "tests/test_model_provider_diagnostics.py",
        )
    }

    assert DEFAULT_SPARSE_EMBEDDER_PROVIDER == "fastembed"
    assert SPARSE_EMBEDDER_PROVIDER_ORDER == (DEFAULT_SPARSE_EMBEDDER_PROVIDER,)
    assert READINESS_PACKAGE_AND_ENV == "package_and_env"
    assert PROVIDER_DIAGNOSTIC_READINESS_SCOPES == (READINESS_PACKAGE_AND_ENV,)
    support_owner = sources["src/rag_core/search/providers/diagnostic_support.py"]
    assert (
        'READINESS_PACKAGE_AND_ENV: ProviderDiagnosticReadinessScope = "package_and_env"'
        in support_owner
    )
    assert (
        sources["src/rag_core/search/providers/sparse.py"].count(
            'DEFAULT_SPARSE_EMBEDDER_PROVIDER = "fastembed"'
        )
        == 1
    )
    sparse_source = sources["src/rag_core/search/providers/sparse.py"]
    assert "SPARSE_EMBEDDER_PROVIDER_ORDER = (" in sparse_source
    assert "_FASTEMBED_PROVIDER" not in sparse_source
    assert 'SPARSE_EMBEDDERS.register("fastembed"' not in sparse_source
    assert 'provider_name = "fastembed"' not in sparse_source
    assert "provider_name = DEFAULT_SPARSE_EMBEDDER_PROVIDER" in sparse_source
    diagnostics = sources[
        "src/rag_core/search/providers/provider_category_diagnostics.py"
    ]
    assert "DEFAULT_SPARSE_EMBEDDER_PROVIDER" in diagnostics
    assert "READINESS_PACKAGE_AND_ENV" in diagnostics
    assert '"readiness_scope": "package_and_env"' not in diagnostics
    assert "_SPARSE_PROVIDER_ALIASES" not in diagnostics
    assert 'default="fastembed"' not in diagnostics
    assert 'known=("fastembed",)' not in diagnostics
    assert "SPARSE_EMBEDDER_PROVIDER_ORDER" in diagnostics
    assert (
        "aliases:"
        not in sources["src/rag_core/search/providers/provider_category_helpers.py"]
    )
    doctor = sources["src/rag_core/cli_doctor.py"]
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]
    assert '"provider": "fastembed"' not in doctor
    assert '"provider": DEFAULT_SPARSE_EMBEDDER_PROVIDER' in doctor
    assert '("sparse", ("fastembed",))' not in doctor_output
    assert "(SPARSE_PROVIDER_CATEGORY, SPARSE_EMBEDDER_PROVIDER_ORDER)" in doctor_output
    test_consumers = (
        sources["tests/test_cli.py"]
        + sources["tests/test_model_provider_diagnostics.py"]
    )
    assert "READINESS_PACKAGE_AND_ENV" in test_consumers
    assert 'readiness_scope"] == "package_and_env"' not in test_consumers





def test_sparse_provider_load_status_labels_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/sparse.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "tests/test_cli.py",
            "tests/test_model_provider_diagnostics.py",
            "tests/test_sparse_provider_log_sanitization.py",
        )
    }

    assert SPARSE_LOAD_NOT_LOADED == "not_loaded"
    assert SPARSE_LOAD_DISABLED == "disabled"
    assert SPARSE_LOAD_LOADED == "loaded"
    assert SPARSE_LOAD_FAILED == "load_failed"
    assert SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR == "not_checked_by_doctor"
    assert SPLADE_LOAD_UNKNOWN_UNTIL_RUN == "unknown_until_sparse_embedding_runs"
    owner = sources["src/rag_core/search/providers/sparse.py"]
    for definition in (
        'SPARSE_LOAD_NOT_LOADED: Final[SparseLoadStatus] = "not_loaded"',
        'SPARSE_LOAD_DISABLED: Final[SparseLoadStatus] = "disabled"',
        'SPARSE_LOAD_LOADED: Final[SparseLoadStatus] = "loaded"',
        'SPARSE_LOAD_FAILED: Final[SparseLoadStatus] = "load_failed"',
        (
            "SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR: Final[SparseLoadStatus] = "
            '"not_checked_by_doctor"'
        ),
    ):
        assert owner.count(definition) == 1
    assert "SPLADE_LOAD_UNKNOWN_UNTIL_RUN: Final[SparseLoadStatus] = (" in owner

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/providers/sparse.py"
    )
    for symbol in (
        "SPARSE_LOAD_FAILED",
        "SPARSE_LOAD_LOADED",
        "SPARSE_LOAD_NOT_CHECKED_BY_DOCTOR",
        "SPLADE_LOAD_UNKNOWN_UNTIL_RUN",
    ):
        assert symbol in consumers
    for duplicate in (
        '"load_status": "not_checked_by_doctor"',
        '"load_status": "unknown_until_sparse_embedding_runs"',
        '== "unknown_until_sparse_embedding_runs"',
        '== "load_failed"',
        '== "loaded"',
    ):
        assert duplicate not in consumers





def test_turbopuffer_delete_continuation_default_has_single_source() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/vector_store_config.py",
            "src/rag_core/search/providers/turbopuffer_config.py",
            "src/rag_core/search/providers/turbopuffer_store.py",
            "src/rag_core/search/providers/turbopuffer_write.py",
        )
    }

    assert (
        sources["src/rag_core/config/vector_store_config.py"].count(
            "DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1_000"
        )
        == 1
    )
    assert (
        "DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1_000"
        not in sources["src/rag_core/search/providers/turbopuffer_config.py"]
    )
    assert (
        "DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1_000"
        not in sources["src/rag_core/search/providers/turbopuffer_store.py"]
    )
    assert (
        "DEFAULT_TURBOPUFFER_DELETE_CONTINUATION_LIMIT = 1000"
        not in sources["src/rag_core/search/providers/turbopuffer_write.py"]
    )
    assert (
        "validate_turbopuffer_delete_continuation_limit"
        in sources["src/rag_core/search/providers/turbopuffer_write.py"]
    )
    assert (
        "def _validate_delete_continuation_limit"
        not in sources["src/rag_core/search/providers/turbopuffer_write.py"]
    )
