"""Shared helpers for log sanitization tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

TEST_API_SECRET = "sk-test-secret"


def assert_log_sanitized(
    caplog: pytest.LogCaptureFixture,
    *,
    forbidden_substrings: Sequence[str],
    logger_name: str | None = None,
) -> None:
    records = caplog.records
    messages = [
        record.getMessage()
        for record in records
        if logger_name is None or record.name == logger_name
    ]
    joined = "\n".join(messages)
    for substring in forbidden_substrings:
        assert substring not in joined


def assert_no_log_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    assert all(record.exc_info is None for record in caplog.records)
