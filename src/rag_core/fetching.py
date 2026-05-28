from __future__ import annotations

import http.client
import socket
from urllib.parse import SplitResult, urljoin, urlsplit

from rag_core.fetch_security import (
    FetchLimits,
    FetchSecurityPolicy,
    IPAddress,
    ValidatedFetchUrl,
    validate_fetch_url,
    validate_resolved_fetch_addresses,
)
from rag_core.fetching_transport import (
    AddressResolver,
    default_fetch_resolver,
    effective_fetch_port,
    fetch_host_header,
    fetch_request_target,
    open_fetch_connection,
    resolve_fetch_addresses,
)
from rag_core.fetching_response import (
    FetchClient,
    FetchError,
    FetchResponse,
    _FetchDeadline,
    build_fetch_response,
    fetch_response_headers,
    is_fetch_redirect,
    read_success_fetch_body,
)
from rag_core.runtime_metadata import DISTRIBUTION_NAME, package_version


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


__all__ = [
    "AddressResolver",
    "DEFAULT_FETCH_USER_AGENT",
    "FetchError",
    "FetchClient",
    "FetchResponse",
    "HttpFetchClient",
    "default_fetch_resolver",
]


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
