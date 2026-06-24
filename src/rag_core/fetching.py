"""HTTP fetch client: connection/transport, response reading, and the security-policed HttpFetchClient."""

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
import ssl
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import cast
from urllib.parse import SplitResult
from rag_core.fetch_security import IPAddress
from urllib.parse import urljoin, urlsplit
from rag_core.fetch_security import (
    FetchSecurityPolicy,
    validate_fetch_url,
    validate_resolved_fetch_addresses,
)
from rag_core.runtime_metadata import DISTRIBUTION_NAME, package_version


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


AddressResolver = Callable[[str, int], Sequence[str | IPAddress]]

_RESOLVER_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rag-core-dns")

def default_fetch_resolver(host: str, port: int) -> tuple[str, ...]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(cast(str, info[4][0]) for info in infos))

def resolve_fetch_addresses(
    resolver: AddressResolver,
    *,
    host: str,
    port: int,
    deadline: _FetchDeadline,
    redacted_url: str,
) -> Sequence[str | IPAddress]:
    future = _RESOLVER_EXECUTOR.submit(resolver, host, port)
    try:
        return future.result(
            timeout=deadline.remaining_seconds(redacted_url=redacted_url)
        )
    except FutureTimeoutError as exc:
        future.cancel()
        raise FetchError(
            "fetch DNS resolution exceeded timeout_seconds for "
            f"{redacted_url}"
        ) from exc

def open_fetch_connection(
    *,
    scheme: str,
    host: str,
    port: int,
    address: IPAddress,
    deadline: _FetchDeadline,
    redacted_url: str,
) -> http.client.HTTPConnection:
    timeout = deadline.remaining_seconds(redacted_url=redacted_url)
    raw_socket = socket.create_connection((str(address), port), timeout=timeout)
    if scheme == "https":
        try:
            raw_socket.settimeout(deadline.remaining_seconds(redacted_url=redacted_url))
            raw_socket = ssl.create_default_context().wrap_socket(
                raw_socket,
                server_hostname=host,
            )
        except OSError:
            raw_socket.close()
            raise
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            host,
            port,
            timeout=timeout,
        )
    else:
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
    connection.sock = raw_socket
    return connection

def fetch_request_target(parsed: SplitResult, *, validated: ValidatedFetchUrl) -> str:
    path = validated.path
    return f"{path}?{parsed.query}" if parsed.query else path

def effective_fetch_port(validated: ValidatedFetchUrl) -> int:
    if validated.port is not None:
        return validated.port
    return 443 if validated.scheme == "https" else 80

def fetch_host_header(
    *,
    host: str,
    port: int,
    scheme: str,
    explicit_port: bool,
) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    if explicit_port or port != default_port:
        return f"{rendered_host}:{port}"
    return rendered_host


def _default_fetch_user_agent() -> str:
    version = package_version()
    if version is None:
        return DISTRIBUTION_NAME
    return f"{DISTRIBUTION_NAME}/{version}"

DEFAULT_FETCH_USER_AGENT = _default_fetch_user_agent()

class HttpFetchClient:
    def __init__(
        self,
        *,
        policy: FetchSecurityPolicy | None = None,
        limits: FetchLimits | None = None,
        resolver: AddressResolver | None = None,
        user_agent: str = DEFAULT_FETCH_USER_AGENT,
    ) -> None:
        self._policy = policy or FetchSecurityPolicy()
        self._limits = limits or FetchLimits()
        self._resolver = resolver or default_fetch_resolver
        self._user_agent = user_agent

    def fetch(self, url: str) -> FetchResponse:
        current_url = url.strip()
        chain: list[ValidatedFetchUrl] = []
        deadline = _FetchDeadline.from_timeout(self._limits.timeout_seconds)
        for _ in range(self._limits.max_redirects + 1):
            validated = validate_fetch_url(current_url, policy=self._policy)
            chain.append(validated)
            status, headers, body = self._request_once(
                current_url,
                validated,
                deadline=deadline,
            )
            location = headers.get("location")
            if is_fetch_redirect(status) and location:
                if len(chain) > self._limits.max_redirects:
                    raise FetchError(
                        "fetch redirect chain exceeds max_redirects "
                        f"({self._limits.max_redirects})"
                    )
                current_url = urljoin(current_url, location.strip())
                continue
            return self._response_from_parts(
                validated=validated,
                status=status,
                headers=headers,
                body=body,
                chain=tuple(chain),
            )
        raise FetchError(
            "fetch redirect chain exceeds max_redirects "
            f"({self._limits.max_redirects})"
        )

    def _request_once(
        self,
        url: str,
        validated: ValidatedFetchUrl,
        *,
        deadline: _FetchDeadline,
    ) -> tuple[int, dict[str, str], bytes]:
        parsed = urlsplit(url)
        port = effective_fetch_port(validated)
        addresses = validate_resolved_fetch_addresses(
            validated.host,
            resolve_fetch_addresses(
                self._resolver,
                host=validated.host,
                port=port,
                deadline=deadline,
                redacted_url=validated.redacted_url,
            ),
            policy=self._policy,
        )
        last_error: Exception | None = None
        for address in addresses:
            try:
                return self._request_address(
                    parsed=parsed,
                    validated=validated,
                    port=port,
                    address=address,
                    deadline=deadline,
                )
            except OSError as exc:
                if _is_fetch_timeout(exc):
                    raise FetchError(
                        "fetch connection exceeded timeout_seconds for "
                        f"{validated.redacted_url}"
                    ) from exc
                last_error = exc
            except http.client.HTTPException as exc:
                raise FetchError(
                    f"fetch request failed for {validated.redacted_url}: {exc}"
                ) from exc
        raise FetchError(
            f"fetch connection failed for {validated.redacted_url}: {last_error}"
        )

    def _request_address(
        self,
        *,
        parsed: SplitResult,
        validated: ValidatedFetchUrl,
        port: int,
        address: IPAddress,
        deadline: _FetchDeadline,
    ) -> tuple[int, dict[str, str], bytes]:
        target = fetch_request_target(parsed, validated=validated)
        headers = {
            "Accept": ", ".join(self._limits.allowed_content_types),
            "Host": fetch_host_header(
                host=validated.host,
                port=port,
                scheme=validated.scheme,
                explicit_port=validated.port is not None,
            ),
            "User-Agent": self._user_agent,
        }
        connection = open_fetch_connection(
            scheme=validated.scheme,
            host=validated.host,
            port=port,
            address=address,
            deadline=deadline,
            redacted_url=validated.redacted_url,
        )
        try:
            _set_connection_timeout(
                connection,
                timeout_seconds=deadline.remaining_seconds(
                    redacted_url=validated.redacted_url
                ),
            )
            try:
                connection.request("GET", target, headers=headers)
            except (OSError, TimeoutError) as exc:
                if _is_fetch_timeout(exc):
                    raise FetchError(
                        "fetch request exceeded timeout_seconds for "
                        f"{validated.redacted_url}"
                    ) from exc
                raise
            _set_connection_timeout(
                connection,
                timeout_seconds=deadline.remaining_seconds(
                    redacted_url=validated.redacted_url
                ),
            )
            try:
                response = connection.getresponse()
            except (OSError, TimeoutError) as exc:
                if _is_fetch_timeout(exc):
                    raise FetchError(
                        "fetch response exceeded timeout_seconds for "
                        f"{validated.redacted_url}"
                    ) from exc
                raise
            response_headers = fetch_response_headers(response)
            status = response.status
            body = b""
            if 200 <= status < 300:
                body = read_success_fetch_body(
                    response,
                    headers=response_headers,
                    limits=self._limits,
                    deadline=deadline,
                    redacted_url=validated.redacted_url,
                )
            return status, response_headers, body
        finally:
            connection.close()

    def _response_from_parts(
        self,
        *,
        validated: ValidatedFetchUrl,
        status: int,
        headers: dict[str, str],
        body: bytes,
        chain: tuple[ValidatedFetchUrl, ...],
    ) -> FetchResponse:
        return build_fetch_response(
            validated=validated,
            status=status,
            headers=headers,
            body=body,
            chain=chain,
            limits=self._limits,
        )

def _set_connection_timeout(
    connection: http.client.HTTPConnection,
    *,
    timeout_seconds: float,
) -> None:
    connection.timeout = timeout_seconds
    if connection.sock is not None:
        connection.sock.settimeout(timeout_seconds)

def _is_fetch_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, socket.timeout))


__all__ = [
    "AddressResolver",
    "DEFAULT_FETCH_USER_AGENT",
    "FetchError",
    "FetchClient",
    "FetchResponse",
    "HttpFetchClient",
    "default_fetch_resolver",
]
