"""Single-owner invariants for ingest lifecycle states, progress, and payloads.

These check the contracts that matter -- the canonical literal values, that each
lifecycle constant/helper has exactly one owning module, and that the consuming
layers import them from that owner -- via runtime values, the package import
graph, and module-level ownership. That survives file merges, renames, and
reformatting. (Previously these were asserted by reading a hand-pinned list of
source files and scanning for forbidden literal substrings, which froze the file
layout and rewarded re-deriving the literals inline.)
"""

from __future__ import annotations

from rag_core.ingest import payloads
from rag_core.ingest.progress import (
    INGEST_PROGRESS_FAILED,
    INGEST_PROGRESS_SUCCEEDED,
)
from rag_core.ingest.states import (
    INGEST_STATE_CREATED,
    INGEST_STATE_PREVIEW,
    INGEST_STATE_REINDEXED,
    INGEST_STATE_REPLACED,
    INGEST_STATE_UNCHANGED,
)

from tests.support.source_graph import (
    defining_modules,
    modules_importing,
    symbol_module,
    under_module,
)

_INGEST_ROOTS = ("src/rag_core/_engine", "src/rag_core/ingest")


def test_ingest_states_have_single_lifecycle_owner() -> None:
    assert INGEST_STATE_CREATED == "created"
    assert INGEST_STATE_PREVIEW == "preview"
    assert INGEST_STATE_REINDEXED == "reindexed"
    assert INGEST_STATE_REPLACED == "replaced"
    assert INGEST_STATE_UNCHANGED == "unchanged"

    # Exactly one module binds each state constant, so no consumer can re-derive
    # the literal under its own name.
    for name in (
        "IngestState",
        "INGEST_STATE_CREATED",
        "INGEST_STATE_PREVIEW",
        "INGEST_STATE_REINDEXED",
        "INGEST_STATE_REPLACED",
        "INGEST_STATE_UNCHANGED",
    ):
        assert defining_modules(*_INGEST_ROOTS, name=name) == {
            "rag_core.ingest.states"
        }


def test_ingest_result_payload_contract_has_single_owner() -> None:
    # The bucket/payload helpers live in one owner module; the local and url
    # result builders import them instead of re-implementing the bucketing.
    for helper in (
        payloads.success_records,
        payloads.failure_records,
        payloads.written_records,
        payloads.skipped_records,
        payloads.ingest_result_payload,
    ):
        assert symbol_module(helper) == "rag_core.ingest.payloads"
        assert defining_modules(*_INGEST_ROOTS, name=helper.__name__) == {
            "rag_core.ingest.payloads"
        }

    importers = modules_importing(
        "src/rag_core/ingest/local",
        "src/rag_core/ingest/urls",
        predicate=under_module("rag_core.ingest.payloads"),
    )
    assert {
        "rag_core.ingest.local.models",
        "rag_core.ingest.urls.results",
    } <= set(importers)


def test_ingest_progress_statuses_have_single_owner() -> None:
    assert INGEST_PROGRESS_SUCCEEDED == "succeeded"
    assert INGEST_PROGRESS_FAILED == "failed"

    for name in (
        "IngestProgressStatus",
        "INGEST_PROGRESS_SUCCEEDED",
        "INGEST_PROGRESS_FAILED",
    ):
        assert defining_modules("src/rag_core", name=name) == {
            "rag_core.ingest.progress"
        }
