from __future__ import annotations

from urllib.parse import urlunsplit

from rag_core.fetching import FetchResponse
from rag_core.fetch_security import (
    FetchSecurityPolicy,
    ValidatedFetchUrl,
    validate_fetch_url,
)


def response_base_url(response: FetchResponse) -> str:
    url = response.url
    path = url.path or "/"
    host = f"[{url.host}]" if ":" in url.host else url.host
    netloc = f"{host}:{url.port}" if url.port is not None else host
    return urlunsplit((url.scheme, netloc, path, "", ""))


def validate_fetch_response_policy(
    response: FetchResponse,
    *,
    policy: FetchSecurityPolicy | None,
) -> None:
    seen: set[tuple[str, str, int | None, str, bool]] = set()
    for url in (*response.redirect_chain, response.url):
        key = (url.scheme, url.host, url.port, url.path, url.has_query)
        if key in seen:
            continue
        seen.add(key)
        validate_fetch_url(_policy_check_url(url), policy=policy)


def _policy_check_url(url: ValidatedFetchUrl) -> str:
    path = url.path or "/"
    host = f"[{url.host}]" if ":" in url.host else url.host
    netloc = f"{host}:{url.port}" if url.port is not None else host
    query = "redacted" if url.has_query else ""
    return urlunsplit((url.scheme, netloc, path, query, ""))
