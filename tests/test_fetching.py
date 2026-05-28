from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from typing import cast

import pytest

from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
from rag_core.fetching import (
    DEFAULT_FETCH_USER_AGENT,
    AddressResolver,
    FetchError,
    HttpFetchClient,
)
from rag_core.runtime_metadata import DISTRIBUTION_NAME, package_version


Route = tuple[int, dict[str, str], bytes]


@dataclass
class LocalHttpOrigin:
    base_url: str
    routes: dict[str, Route]


@pytest.fixture
def http_origin() -> Iterator[LocalHttpOrigin]:
    routes: dict[str, Route] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, route_headers, body = routes.get(
                self.path,
                (404, {"Content-Type": "text/plain"}, b"missing"),
            )
            headers = dict(route_headers)
            chunk_delay = float(headers.pop("X-Chunk-Delay", "0"))
            response_delay = float(headers.pop("X-Response-Delay", "0"))
            if response_delay > 0:
                time.sleep(response_delay)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if chunk_delay > 0:
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(chunk_delay)
                return
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = cast(tuple[str, int], server.server_address)
        yield LocalHttpOrigin(base_url=f"http://{host}:{port}", routes=routes)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_fetch_client_downloads_body_with_safe_url_metadata(
    http_origin: LocalHttpOrigin,
) -> None:
    body = b"# Guide\n\nHello"
    http_origin.routes["/docs?token=secret"] = (
        200,
        {"Content-Type": "text/markdown; charset=utf-8"},
        body,
    )

    response = _local_fetch_client().fetch(f"{http_origin.base_url}/docs?token=secret")

    assert response.status_code == 200
    assert response.content_type == "text/markdown"
    assert response.content_length is None
    assert response.content_sha256 == hashlib.sha256(body).hexdigest()
    assert response.body == body
    assert response.url.redacted_url.endswith("/docs?redacted")
    rendered = repr(response) + repr(asdict(response))
    assert "token=secret" not in rendered


def test_default_fetch_user_agent_uses_runtime_package_identity() -> None:
    version = package_version()

    assert DEFAULT_FETCH_USER_AGENT == (
        f"{DISTRIBUTION_NAME}/{version}" if version is not None else DISTRIBUTION_NAME
    )
    assert DEFAULT_FETCH_USER_AGENT != "rag-core/0.1"


def test_http_fetch_client_strips_url_whitespace_before_request(
    http_origin: LocalHttpOrigin,
) -> None:
    http_origin.routes["/docs"] = (
        200,
        {"Content-Type": "text/plain"},
        b"ok",
    )

    response = _local_fetch_client().fetch(f" {http_origin.base_url}/docs ")

    assert response.body == b"ok"


def test_http_fetch_client_follows_redirects_with_limit(
    http_origin: LocalHttpOrigin,
) -> None:
    http_origin.routes["/start"] = (
        302,
        {"Location": "/final?token=secret", "Content-Type": "text/plain"},
        b"",
    )
    http_origin.routes["/final?token=secret"] = (
        200,
        {"Content-Type": "text/plain"},
        b"done",
    )

    response = _local_fetch_client().fetch(f"{http_origin.base_url}/start")

    assert response.body == b"done"
    assert [item.path for item in response.redirect_chain] == ["/start", "/final"]
    assert response.redirect_chain[-1].redacted_url.endswith("/final?redacted")

    with pytest.raises(FetchError, match="max_redirects"):
        _local_fetch_client(limits=FetchLimits(max_redirects=0)).fetch(
            f"{http_origin.base_url}/start"
        )


def test_http_fetch_client_revalidates_redirect_targets(
    http_origin: LocalHttpOrigin,
) -> None:
    http_origin.routes["/bad-redirect"] = (
        302,
        {"Location": "file:///tmp/secret", "Content-Type": "text/plain"},
        b"",
    )

    with pytest.raises(ValueError, match="unsupported fetch URL scheme"):
        _local_fetch_client().fetch(f"{http_origin.base_url}/bad-redirect")


def test_http_fetch_client_blocks_private_dns_results_by_default() -> None:
    client = HttpFetchClient(resolver=lambda _host, _port: ("127.0.0.1",))

    with pytest.raises(ValueError, match="not public"):
        client.fetch("https://example.com/docs")


def test_http_fetch_client_rejects_disallowed_content_type(
    http_origin: LocalHttpOrigin,
) -> None:
    raw_content_type = "image/png; token=secret"
    http_origin.routes["/image"] = (
        200,
        {"Content-Type": raw_content_type},
        b"fake-png",
    )

    with pytest.raises(FetchError, match="content type") as exc_info:
        _local_fetch_client().fetch(f"{http_origin.base_url}/image")
    assert raw_content_type not in str(exc_info.value)
    assert "token=secret" not in str(exc_info.value)


def test_http_fetch_client_rejects_oversized_body(
    http_origin: LocalHttpOrigin,
) -> None:
    http_origin.routes["/large"] = (
        200,
        {"Content-Type": "text/plain", "Content-Length": "6"},
        b"larger",
    )

    with pytest.raises(FetchError, match="max_bytes"):
        _local_fetch_client(limits=FetchLimits(max_bytes=5)).fetch(
            f"{http_origin.base_url}/large"
        )


def test_http_fetch_client_enforces_end_to_end_body_timeout(
    http_origin: LocalHttpOrigin,
) -> None:
    http_origin.routes["/slow"] = (
        200,
        {
            "Content-Type": "text/plain",
            "Content-Length": "3",
            "X-Chunk-Delay": "0.04",
        },
        b"abc",
    )

    with pytest.raises(FetchError, match="timeout_seconds"):
        _local_fetch_client(limits=FetchLimits(timeout_seconds=0.05)).fetch(
            f"{http_origin.base_url}/slow"
        )


def test_http_fetch_client_enforces_one_timeout_across_redirects(
    http_origin: LocalHttpOrigin,
) -> None:
    http_origin.routes["/start"] = (
        302,
        {
            "Location": "/middle",
            "Content-Type": "text/plain",
            "X-Response-Delay": "0.04",
        },
        b"",
    )
    http_origin.routes["/middle"] = (
        302,
        {
            "Location": "/final",
            "Content-Type": "text/plain",
            "X-Response-Delay": "0.04",
        },
        b"",
    )
    http_origin.routes["/final"] = (
        200,
        {"Content-Type": "text/plain"},
        b"done",
    )

    with pytest.raises(FetchError, match="timeout_seconds"):
        _local_fetch_client(limits=FetchLimits(timeout_seconds=0.07)).fetch(
            f"{http_origin.base_url}/start"
        )


def test_http_fetch_client_enforces_timeout_during_resolution() -> None:
    def slow_resolver(_host: str, _port: int) -> tuple[str, ...]:
        time.sleep(0.2)
        return ("127.0.0.1",)

    started_at = time.monotonic()
    with pytest.raises(FetchError, match="DNS resolution exceeded timeout_seconds"):
        _local_fetch_client(
            limits=FetchLimits(timeout_seconds=0.02),
            resolver=slow_resolver,
        ).fetch("http://example.test/docs")

    assert time.monotonic() - started_at < 0.15


def _local_fetch_client(
    *,
    limits: FetchLimits | None = None,
    resolver: AddressResolver | None = None,
) -> HttpFetchClient:
    return HttpFetchClient(
        policy=FetchSecurityPolicy(
            allowed_schemes=("https", "http"),
            allow_private_addresses=True,
        ),
        limits=limits,
        resolver=resolver,
    )
