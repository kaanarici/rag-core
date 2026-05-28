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
    joined = "\n".join(_caplog_messages(caplog, logger_name=logger_name))
    for substring in forbidden_substrings:
        assert substring not in joined


def assert_log_record_contains(
    caplog: pytest.LogCaptureFixture,
    *required_substrings: str,
    logger_name: str | None = None,
) -> None:
    assert required_substrings
    messages = _caplog_messages(caplog, logger_name=logger_name)
    assert any(
        all(substring in message for substring in required_substrings)
        for message in messages
    ), (
        f"expected one log record from {logger_name or 'any logger'} to contain "
        f"{required_substrings!r}; saw {messages!r}"
    )


def assert_no_log_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    assert all(record.exc_info is None for record in caplog.records)


def assert_caplog_omits_private(
    caplog: pytest.LogCaptureFixture,
    *forbidden: str,
) -> None:
    assert_log_sanitized(
        caplog,
        forbidden_substrings=(*forbidden, TEST_API_SECRET, "Traceback"),
    )
    assert_no_log_exceptions(caplog)


def _caplog_messages(
    caplog: pytest.LogCaptureFixture,
    *,
    logger_name: str | None = None,
) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if logger_name is None or record.name == logger_name
    ]
