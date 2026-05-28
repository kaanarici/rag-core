"""Test doubles and helpers; import from ``tests.support`` in tests."""

from tests.support.fakes import (
    BASELINE_VOCABULARY,
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSearchSidecar,
    FakeSparseEmbedder,
    FakeSparseEmbedderNoMulti,
    FixtureEmbeddingProvider,
    KeywordEmbeddingProvider,
    KeywordSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)
from tests.support.log_sanitization import (
    TEST_API_SECRET,
    assert_caplog_omits_private,
    assert_log_sanitized,
    assert_log_record_contains,
    assert_no_log_exceptions,
)

__all__ = [
    "BASELINE_VOCABULARY",
    "FakeEmbeddingProvider",
    "FakeReranker",
    "FakeSearchSidecar",
    "FakeSparseEmbedder",
    "FakeSparseEmbedderNoMulti",
    "FixtureEmbeddingProvider",
    "KeywordEmbeddingProvider",
    "KeywordSparseEmbedder",
    "RecordingVectorStore",
    "TEST_API_SECRET",
    "assert_caplog_omits_private",
    "assert_log_record_contains",
    "assert_log_sanitized",
    "assert_no_log_exceptions",
    "make_search_result",
    "make_test_config",
]
