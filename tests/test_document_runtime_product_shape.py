from __future__ import annotations

from pathlib import Path

from rag_core.documents.exception_names import exception_type, root_exception_type
from rag_core.documents.http_errors import safe_http_status
from rag_core.documents.page_indices import normalize_page_indices
from rag_core.documents.subprocess_env import (
    COMMON_SUBPROCESS_ENV_KEYS,
    NODE_SUBPROCESS_ENV_KEYS,
    PYTHON_SUBPROCESS_ENV_KEYS,
    TRANSPORT_SUBPROCESS_ENV_KEYS,
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


def test_document_subprocess_env_allowlists_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/documents/subprocess_env.py",
            "src/rag_core/documents/pdf_inspector_runtime.py",
            "src/rag_core/documents/ocr_command_runtime.py",
        )
    }

    assert COMMON_SUBPROCESS_ENV_KEYS == (
        "PATH",
        "HOME",
        "SYSTEMROOT",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    )
    assert PYTHON_SUBPROCESS_ENV_KEYS == (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUTF8",
        "VIRTUAL_ENV",
    )
    assert NODE_SUBPROCESS_ENV_KEYS == ("NODE_OPTIONS",)
    assert TRANSPORT_SUBPROCESS_ENV_KEYS == (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    )

    owner = sources["src/rag_core/documents/subprocess_env.py"]
    assert owner.count("COMMON_SUBPROCESS_ENV_KEYS = (") == 1
    assert owner.count("PYTHON_SUBPROCESS_ENV_KEYS = (") == 1
    assert owner.count('NODE_SUBPROCESS_ENV_KEYS = ("NODE_OPTIONS",)') == 1
    assert owner.count("TRANSPORT_SUBPROCESS_ENV_KEYS = (") == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/documents/subprocess_env.py"
    )
    assert "allowlisted_subprocess_env" in consumers
    assert (
        "NODE_SUBPROCESS_ENV_KEYS"
        in sources["src/rag_core/documents/pdf_inspector_runtime.py"]
    )
    assert (
        "PYTHON_SUBPROCESS_ENV_KEYS"
        in sources["src/rag_core/documents/ocr_command_runtime.py"]
    )
    assert "_RUNTIME_ENV_KEYS" not in consumers
    assert "_TRANSPORT_ENV_KEYS" not in consumers
    for env_name in (
        "SYSTEMROOT",
        "LC_CTYPE",
        "HTTP_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PYTHONUTF8",
        "NODE_OPTIONS",
    ):
        assert f'"{env_name}"' not in consumers





def test_ocr_page_index_normalization_has_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/documents/page_indices.py",
            "src/rag_core/documents/local_parse.py",
            "src/rag_core/documents/converters/pdf_converter_inspector.py",
            "src/rag_core/documents/ocr_command_runtime.py",
            "src/rag_core/documents/ocr_commands/gemini.py",
            "src/rag_core/documents/ocr_commands/mistral.py",
        )
    }

    assert normalize_page_indices([2, True, 0, False, 2, -1, "3", 1.0]) == [2, 0]
    assert normalize_page_indices(
        [2, True, 0, False, 2, -1, "3", 1.0],
        sort=True,
    ) == [0, 2]
    assert normalize_page_indices([0, 3, 1], page_count=2) == [0, 1]
    assert normalize_page_indices([], page_count=3, default_all_pages=True) == [0, 1, 2]

    owner = sources["src/rag_core/documents/page_indices.py"]
    assert owner.count("def normalize_page_indices(") == 1
    assert "isinstance(raw_index, bool)" in owner
    assert "raw_index < 0" in owner
    assert "raw_index >= page_count" in owner

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/documents/page_indices.py"
    )
    assert consumers.count("normalize_page_indices(") >= 5
    assert "for raw_index in raw_indices:" not in consumers
    assert "isinstance(raw_index, bool)" not in consumers
    assert "raw_index < 0" not in consumers
    assert "raw_index >= page_count" not in consumers





def test_ocr_http_status_sanitization_has_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/documents/http_errors.py",
            "src/rag_core/documents/ocr_commands/gemini.py",
            "src/rag_core/documents/ocr_commands/mistral_runtime.py",
        )
    }

    class WithStatus:
        code = 429

    class WithBoolStatus:
        code = True

    class WithoutStatus:
        pass

    assert safe_http_status(WithStatus()) == 429
    assert safe_http_status(WithBoolStatus()) == "unknown"
    assert safe_http_status(WithoutStatus()) == "unknown"

    owner = sources["src/rag_core/documents/http_errors.py"]
    assert owner.count("def safe_http_status(") == 1
    assert 'getattr(exc, "code", None)' in owner
    assert "isinstance(code, bool)" in owner

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/documents/http_errors.py"
    )
    assert consumers.count("safe_http_status(exc)") == 2
    assert "def _safe_http_status" not in consumers
    assert 'getattr(exc, "code", None)' not in consumers
    assert "isinstance(code, bool)" not in consumers





def test_document_exception_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/documents/exception_names.py",
            "src/rag_core/documents/pdf_inspector.py",
            "src/rag_core/documents/pdf_inspector_runtime.py",
            "src/rag_core/documents/local_parse.py",
            "src/rag_core/documents/ocr_command_runtime.py",
            "src/rag_core/documents/converters/base.py",
            "src/rag_core/documents/converters/pdf_converter_inspector_calls.py",
            "src/rag_core/documents/converters/registry_loader.py",
        )
    }

    try:
        try:
            raise OSError("inner")
        except OSError as inner:
            raise ValueError("outer") from inner
    except ValueError as exc:
        assert exception_type(exc) == "ValueError"
        assert root_exception_type(exc) == "OSError"

    owner = sources["src/rag_core/documents/exception_names.py"]
    assert owner.count("def exception_type(") == 1
    assert owner.count("def root_exception_type(") == 1
    assert "exc.__cause__" in owner
    assert "type(exc).__name__" in owner

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/documents/exception_names.py"
    )
    assert "exception_type(exc)" in consumers
    assert "root_exception_type(exc)" in consumers
    assert "def _exception_type" not in consumers
    assert "type(exc).__name__" not in consumers
    assert (
        "exc.__cause__ if isinstance(exc.__cause__, Exception) else exc"
        not in consumers
    )
