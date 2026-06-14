"""Retry helpers for first-party provider API calls."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

TRANSIENT_HTTP_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")
ProviderCall = Callable[[], Awaitable[T]]
ProviderErrorClassifier = Callable[[Exception], bool]
ProviderSleep = Callable[[float], Awaitable[None]]
ProviderRandom = Callable[[], float]

logger = logging.getLogger(__name__)


async def retry_provider_call(
    fn: ProviderCall[T],
    *,
    classify: ProviderErrorClassifier,
    provider_name: str,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: ProviderSleep | None = None,
    rand: ProviderRandom | None = None,
) -> T:
    if attempts < 1:
        raise ValueError("provider retry attempts must be at least 1")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("provider retry delays must be non-negative")

    sleep_fn = sleep or asyncio.sleep
    rand_fn = rand or random.random
    first_transient_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if not classify(exc):
                raise
            if first_transient_error is None:
                first_transient_error = exc
            if attempt >= attempts:
                raise first_transient_error from None
            delay = _full_jitter_delay(
                attempt=attempt,
                base_delay=base_delay,
                max_delay=max_delay,
                rand=rand_fn,
            )
            logger.warning(
                "Retrying provider API call: provider=%s attempt=%d max_attempts=%d "
                "delay_seconds=%.3f error_type=%s",
                provider_name,
                attempt,
                attempts,
                delay,
                type(exc).__name__,
            )
            await sleep_fn(delay)

    raise AssertionError("provider retry loop exited unexpectedly")


def is_transient_http_status(status_code: object) -> bool:
    return (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and status_code in TRANSIENT_HTTP_STATUS_CODES
    )


def matches_exception_type(exc: Exception, error_type: object) -> bool:
    return (
        isinstance(error_type, type)
        and issubclass(error_type, BaseException)
        and isinstance(exc, error_type)
    )


def _full_jitter_delay(
    *,
    attempt: int,
    base_delay: float,
    max_delay: float,
    rand: ProviderRandom,
) -> float:
    cap = float(min(max_delay, base_delay * (2 ** (attempt - 1))))
    raw_jitter = float(rand())
    jitter = min(max(raw_jitter, 0.0), 1.0)
    return float(cap * jitter)
