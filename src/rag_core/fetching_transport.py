from __future__ import annotations

import http.client
import socket
import ssl
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import cast
from urllib.parse import SplitResult

from rag_core.fetch_security import IPAddress, ValidatedFetchUrl
from rag_core.fetching_response import FetchError, _FetchDeadline

AddressResolver = Callable[[str, int], Sequence[str | IPAddress]]


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
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(resolver, host, port)
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
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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
