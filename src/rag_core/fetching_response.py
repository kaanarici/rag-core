from __future__ import annotations

import hashlib
import http.client
import socket
import time
from dataclasses import dataclass
from typing import Protocol

from rag_core.fetch_security import (
    FetchLimits,
    ValidatedFetchUrl,
    is_allowed_fetch_content_type,
)


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FetchDeadline:
    expires_at: float

    @classmethod
    def from_timeout(cls, timeout_seconds: float) -> _FetchDeadline:
        return cls(expires_at=time.monotonic() + timeout_seconds)

    def remaining_seconds(self, *, redacted_url: str) -> float:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0:
            raise FetchError(
                "fetch exceeded timeout_seconds for "
                f"{redacted_url}"
            )
        return remaining


@dataclass(frozen=True)
class FetchResponse:
    url: ValidatedFetchUrl
    status_code: int
    content_type: str
    content_length: int | None
    content_sha256: str
    body: bytes
    redirect_chain: tuple[ValidatedFetchUrl, ...]


class FetchClient(Protocol):
    def fetch(self, url: str) -> FetchResponse: ...


def build_fetch_response(
    *,
    validated: ValidatedFetchUrl,
    status: int,
    headers: dict[str, str],
    body: bytes,
    chain: tuple[ValidatedFetchUrl, ...],
    limits: FetchLimits,
) -> FetchResponse:
    if not 200 <= status < 300:
        raise FetchError(f"fetch failed for {validated.redacted_url}: HTTP {status}")
    content_type = headers.get("content-type", "")
    validate_fetch_response_headers(
        headers,
        limits=limits,
        redacted_url=validated.redacted_url,
    )
    return FetchResponse(
        url=validated,
        status_code=status,
        content_type=_media_type(content_type),
        content_length=_parse_content_length(headers.get("content-length")),
        content_sha256=hashlib.sha256(body).hexdigest(),
        body=body,
        redirect_chain=chain,
    )


def validate_fetch_response(
    response: FetchResponse,
    *,
    limits: FetchLimits,
) -> None:
    redirect_count = max(len(response.redirect_chain) - 1, 0)
    if redirect_count > limits.max_redirects:
        raise FetchError(
            "fetch redirect chain exceeds max_redirects "
            f"({redirect_count} > {limits.max_redirects})"
        )
    if not 200 <= response.status_code < 300:
        raise FetchError(
            f"fetch failed for {response.url.redacted_url}: HTTP {response.status_code}"
        )
    validate_fetch_response_headers(
        {
            "content-type": response.content_type,
            "content-length": str(response.content_length)
            if response.content_length is not None
            else "",
        },
        limits=limits,
        redacted_url=response.url.redacted_url,
    )
    if len(response.body) > limits.max_bytes:
        raise FetchError(f"fetch body exceeds max_bytes for {response.url.redacted_url}")
    expected_sha256 = hashlib.sha256(response.body).hexdigest()
    if response.content_sha256 != expected_sha256:
        raise FetchError(f"fetch body checksum mismatch for {response.url.redacted_url}")


def validate_fetch_response_headers(
    headers: dict[str, str],
    *,
    limits: FetchLimits,
    redacted_url: str,
) -> None:
    content_type = headers.get("content-type")
    if not is_allowed_fetch_content_type(content_type, limits=limits):
        raise FetchError(
            "fetch content type is not allowed for "
            f"{redacted_url}"
        )

    parsed_length = _parse_content_length(headers.get("content-length"))
    if parsed_length is not None and parsed_length > limits.max_bytes:
        raise FetchError(f"fetch body exceeds max_bytes for {redacted_url}")


def fetch_response_headers(response: http.client.HTTPResponse) -> dict[str, str]:
    return {key.lower(): value for key, value in response.getheaders()}


def read_success_fetch_body(
    response: http.client.HTTPResponse,
    *,
    headers: dict[str, str],
    limits: FetchLimits,
    deadline: _FetchDeadline,
    redacted_url: str,
) -> bytes:
    validate_fetch_response_headers(
        headers,
        limits=limits,
        redacted_url=redacted_url,
    )
    max_allowed = limits.max_bytes + 1
    chunk_size = min(64 * 1024, max_allowed)
    chunks: list[bytes] = []
    total_bytes = 0
    while total_bytes < max_allowed:
        remaining_seconds = deadline.remaining_seconds(redacted_url=redacted_url)
        _set_fetch_response_timeout(response, timeout_seconds=remaining_seconds)
        try:
            chunk = response.read(min(chunk_size, max_allowed - total_bytes))
        except TimeoutError as exc:
            raise FetchError(
                "fetch body read exceeded timeout_seconds for "
                f"{redacted_url}"
            ) from exc
        except OSError as exc:
            if isinstance(exc, socket.timeout):
                raise FetchError(
                    "fetch body read exceeded timeout_seconds for "
                    f"{redacted_url}"
                ) from exc
            raise
        if not chunk:
            break
        chunks.append(chunk)
        total_bytes += len(chunk)
        if total_bytes > limits.max_bytes:
            raise FetchError(f"fetch body exceeds max_bytes for {redacted_url}")
    body = b"".join(chunks)
    if len(body) > limits.max_bytes:
        raise FetchError(f"fetch body exceeds max_bytes for {redacted_url}")
    return body


def is_fetch_redirect(status: int) -> bool:
    return 300 <= status < 400


def _parse_content_length(content_length: str | None) -> int | None:
    if content_length is None:
        return None
    try:
        value = int(content_length)
    except ValueError:
        return None
    return value if value >= 0 else None


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _set_fetch_response_timeout(
    response: http.client.HTTPResponse,
    *,
    timeout_seconds: float,
) -> None:
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None) if fp is not None else None
    sock = getattr(raw, "_sock", None)
    if sock is not None and hasattr(sock, "settimeout"):
        sock.settimeout(timeout_seconds)
