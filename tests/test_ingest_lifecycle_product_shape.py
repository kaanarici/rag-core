from __future__ import annotations

from pathlib import Path

from rag_core.ingest_states import (
    INGEST_STATE_CREATED,
    INGEST_STATE_PREVIEW,
    INGEST_STATE_REINDEXED,
    INGEST_STATE_REPLACED,
    INGEST_STATE_UNCHANGED,
)
from rag_core.ingest_progress_statuses import (
    INGEST_PROGRESS_FAILED,
    INGEST_PROGRESS_SUCCEEDED,
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


def test_ingest_states_have_single_lifecycle_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/ingest_states.py",
            "src/rag_core/_engine/core_lifecycle.py",
            "src/rag_core/_engine/core_builders.py",
            "src/rag_core/ingest_result_payloads.py",
            "src/rag_core/remote_ingest_results.py",
        )
    }

    assert INGEST_STATE_CREATED == "created"
    assert INGEST_STATE_PREVIEW == "preview"
    assert INGEST_STATE_REINDEXED == "reindexed"
    assert INGEST_STATE_REPLACED == "replaced"
    assert INGEST_STATE_UNCHANGED == "unchanged"

    owner = sources["src/rag_core/ingest_states.py"]
    for definition in (
        'INGEST_STATE_CREATED: Final[IngestState] = "created"',
        'INGEST_STATE_PREVIEW: Final[IngestState] = "preview"',
        'INGEST_STATE_REINDEXED: Final[IngestState] = "reindexed"',
        'INGEST_STATE_REPLACED: Final[IngestState] = "replaced"',
        'INGEST_STATE_UNCHANGED: Final[IngestState] = "unchanged"',
    ):
        assert owner.count(definition) == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/ingest_states.py"
    )
    for symbol in (
        "INGEST_STATE_CREATED",
        "INGEST_STATE_PREVIEW",
        "INGEST_STATE_REINDEXED",
        "INGEST_STATE_REPLACED",
        "INGEST_STATE_UNCHANGED",
    ):
        assert symbol in consumers
    for duplicate in (
        'return "created"',
        'return "reindexed"',
        'return "replaced"',
        'return "unchanged"',
        'ingest_state="preview"',
        'record.ingest_state != "unchanged"',
        'record.ingest_state == "unchanged"',
    ):
        assert duplicate not in consumers




def test_ingest_result_payload_contract_has_single_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        path: (root / path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/ingest_result_payloads.py",
            "src/rag_core/local_ingest_models.py",
            "src/rag_core/remote_ingest_results.py",
        )
    }

    owner = sources["src/rag_core/ingest_result_payloads.py"]
    for symbol in (
        "success_records",
        "failure_records",
        "written_records",
        "skipped_records",
        "ingest_result_payload",
    ):
        assert f"def {symbol}" in owner
        assert symbol in sources["src/rag_core/local_ingest_models.py"]
        assert symbol in sources["src/rag_core/remote_ingest_results.py"]

    for path in (
        "src/rag_core/local_ingest_models.py",
        "src/rag_core/remote_ingest_results.py",
    ):
        source = sources[path]
        assert "from rag_core.ingest_result_payloads import (" in source
        assert "local_ingest_result_payloads" not in source
        for duplicate in (
            '"succeeded_count": len(',
            '"written_count": len(',
            '"skipped_count": len(',
            '"failed_count": len(',
            '"records": [',
            '"succeeded": [',
            '"written": [',
            '"skipped": [',
            '"failed": [',
            "isinstance(record, LocalIngestSuccess)",
            "isinstance(record, RemoteUrlIngestSuccess)",
            "isinstance(record, LocalIngestFailure)",
            "isinstance(record, RemoteUrlIngestFailure)",
        ):
            assert duplicate not in source




def test_ingest_progress_statuses_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/ingest_progress_statuses.py",
            "src/rag_core/local_ingest_records.py",
            "src/rag_core/local_ingest_runner.py",
            "src/rag_core/remote_ingest_runner.py",
            "src/rag_core/_engine/core_archive_runner.py",
            "src/rag_core/events/ingest_events.py",
        )
    }

    assert INGEST_PROGRESS_SUCCEEDED == "succeeded"
    assert INGEST_PROGRESS_FAILED == "failed"

    owner = sources["src/rag_core/ingest_progress_statuses.py"]
    for definition in (
        'INGEST_PROGRESS_SUCCEEDED: Final[IngestProgressStatus] = "succeeded"',
        'INGEST_PROGRESS_FAILED: Final[IngestProgressStatus] = "failed"',
    ):
        assert owner.count(definition) == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/ingest_progress_statuses.py"
    )
    for symbol in (
        "INGEST_PROGRESS_SUCCEEDED",
        "INGEST_PROGRESS_FAILED",
        "IngestProgressStatus",
    ):
        assert symbol in consumers
    for duplicate in (
        'Literal["succeeded", "failed"]',
        "status: LocalIngestProgressStatus",
        "status: RemoteIngestProgressStatus",
        'status: IngestProgressStatus = "failed"',
        'status: IngestProgressStatus = "succeeded"',
        'status = "failed"',
        'status = "succeeded"',
        'status: Literal["succeeded", "failed"] = "failed"',
        'status: Literal["succeeded", "failed"] = "succeeded"',
    ):
        assert duplicate not in consumers
