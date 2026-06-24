from __future__ import annotations

import pytest

from rag_core.events.trace_payload_fields import (
    CONTEXT_PACK_SEARCH_STAGE,
    FUSE_SEARCH_STAGE,
    POSTPROCESS_SEARCH_STAGE,
    QUERY_TRANSFORM_SEARCH_STAGE,
    RERANK_SEARCH_STAGE,
    RETRIEVE_SEARCH_STAGE,
    SEARCH_ERROR_STAGE,
    TRACE_ABSENT_LABEL,
    TRACE_EMPTY_LABEL,
    TRACE_UNKNOWN_LABEL,
    safe_trace_label,
)
from rag_core.search.query_plan_trace import QUERY_PLAN_TRACE_STORE_DEFAULT_LABEL

from tests.support.source_graph import (
    defining_modules,
    modules_importing,
    under_module,
)

_TRACE_ROOTS = (
    "src/rag_core/events",
    "src/rag_core/search",
    "src/rag_core/_engine",
)
_TRACE_FIELDS_OWNER = "rag_core.events.trace_payload_fields"


@pytest.mark.parametrize(
    "secret",
    [
        # Stripe secret/restricted keys use the underscore form that the
        # hyphenated sk- pattern did not catch.
        ("sk_live_" + "a" * 24),
        ("sk_test_" + "a" * 24),
        ("rk_live_" + "a" * 24),
        ("provider:sk_live_" + "a" * 24),
        # Stripe webhook signing secret.
        ("whsec_" + "a" * 32),
        # Google API key: AIza prefix + 35 url-safe chars.
        ("AIzaSy" + "a" * 35),
    ],
)
def test_safe_trace_label_redacts_stripe_and_google_keys(secret: str) -> None:
    assert safe_trace_label(secret, stage=False) == TRACE_UNKNOWN_LABEL


@pytest.mark.parametrize(
    "benign",
    [
        "openai",
        "cohere",
        "rerank-v3.5",
        "text-embedding-3-small",
        "stage_retrieve",
        # Must not over-redact a benign label that merely ends in ...sk_live_.
        ("di" + "sk_live_" + "a" * 10),
    ],
)
def test_safe_trace_label_keeps_benign_provider_labels(benign: str) -> None:
    assert safe_trace_label(benign, stage=False) == benign


def test_safe_trace_label_rejects_oversized_label_without_scanning() -> None:
    # A label longer than the 80-char cap is unknown regardless of content, and
    # must short-circuit before the secret regexes so a hostile oversized label
    # cannot pin CPU. A 200k label that ends in a secret-shaped suffix still
    # resolves instantly to unknown.
    oversized = "a" * 200_000 + "sk_live_" + "a" * 16
    assert safe_trace_label(oversized, stage=False) == TRACE_UNKNOWN_LABEL


def test_trace_absent_labels_have_single_trace_owner() -> None:
    assert TRACE_EMPTY_LABEL == ""
    assert TRACE_ABSENT_LABEL == "none"
    assert TRACE_UNKNOWN_LABEL == "unknown"
    assert QUERY_PLAN_TRACE_STORE_DEFAULT_LABEL == "store_default"

    # Each trace label has one owner, so no consumer can re-derive the literal
    # under its own name.
    for name in ("TRACE_EMPTY_LABEL", "TRACE_ABSENT_LABEL", "TRACE_UNKNOWN_LABEL"):
        assert defining_modules(*_TRACE_ROOTS, name=name) == {_TRACE_FIELDS_OWNER}
    assert defining_modules(*_TRACE_ROOTS, name="QUERY_PLAN_TRACE_STORE_DEFAULT_LABEL") == {
        "rag_core.search.query_plan_trace"
    }

    # The trace consumers reuse the shared absent label rather than the bare
    # "none" literal.
    absent_label_consumers = modules_importing(
        *_TRACE_ROOTS,
        predicate=lambda module: module
        == f"{_TRACE_FIELDS_OWNER}.TRACE_ABSENT_LABEL",
    )
    assert {
        "rag_core.events.search_events",
        "rag_core.events.sink_payloads",
        "rag_core.search.query_plan_trace",
    } <= set(absent_label_consumers)


def test_search_stage_labels_have_single_trace_owner() -> None:
    expected = {
        "QUERY_TRANSFORM_SEARCH_STAGE": QUERY_TRANSFORM_SEARCH_STAGE,
        "RETRIEVE_SEARCH_STAGE": RETRIEVE_SEARCH_STAGE,
        "FUSE_SEARCH_STAGE": FUSE_SEARCH_STAGE,
        "RERANK_SEARCH_STAGE": RERANK_SEARCH_STAGE,
        "POSTPROCESS_SEARCH_STAGE": POSTPROCESS_SEARCH_STAGE,
        "CONTEXT_PACK_SEARCH_STAGE": CONTEXT_PACK_SEARCH_STAGE,
        "SEARCH_ERROR_STAGE": SEARCH_ERROR_STAGE,
    }
    assert expected == {
        "QUERY_TRANSFORM_SEARCH_STAGE": "query_transform",
        "RETRIEVE_SEARCH_STAGE": "retrieve",
        "FUSE_SEARCH_STAGE": "fuse",
        "RERANK_SEARCH_STAGE": "rerank",
        "POSTPROCESS_SEARCH_STAGE": "postprocess",
        "CONTEXT_PACK_SEARCH_STAGE": "context_pack",
        "SEARCH_ERROR_STAGE": "search",
    }

    # Each stage label has one owner, so the pipeline stages emit the shared
    # constant instead of re-deriving the literal as a stage= default.
    for symbol in expected:
        assert defining_modules(*_TRACE_ROOTS, name=symbol) == {_TRACE_FIELDS_OWNER}

    stage_consumers = modules_importing(
        *_TRACE_ROOTS,
        predicate=under_module(_TRACE_FIELDS_OWNER),
    )
    assert {
        "rag_core._engine.core_retrieval",
        "rag_core.search.pipeline.runner",
        "rag_core.search.pipeline_runner",
    } <= set(stage_consumers)
