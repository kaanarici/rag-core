from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from rag_core.cli_inputs import cli_error_message
from rag_core.fetch_security import (
    FetchSecurityPolicy,
    ValidatedFetchUrl,
    validate_fetch_url,
)
from rag_core.fetch_security_url import (
    safe_remote_source_url,
)
from rag_core.local_sources import is_multi_link_regular_file, path_has_symlink_segment
from rag_core.remote_ingest_models import RemoteUrlIngestRequest, RemoteUrlSourceItem

REMOTE_URL_FILE_MAX_BYTES = 1_048_576
REMOTE_URL_LIST_MAX_ITEMS = 10_000


def remote_url_source_items(
    request: RemoteUrlIngestRequest,
) -> tuple[tuple[RemoteUrlSourceItem, ...], str, str | None]:
    inline_urls = tuple(request.urls)
    if request.url_file is not None and inline_urls:
        raise ValueError("url_file cannot be combined with inline urls")
    if request.url_file is None:
        return (
            tuple(
                _read_url_values(
                    inline_urls,
                    policy=request.fetch_policy,
                    error_label="item",
                )
            ),
            "inline",
            None,
        )
    url_file = Path(request.url_file)
    return (
        tuple(_read_url_file(url_file, policy=request.fetch_policy)),
        "file",
        str(url_file),
    )


def validate_unique_url_keys(urls: Sequence[RemoteUrlSourceItem]) -> None:
    seen: dict[str, RemoteUrlSourceItem] = {}
    for item in urls:
        previous = seen.get(item.document_key)
        if previous is not None:
            raise ValueError(
                "URL list entries resolve to the same document key: "
                f"lines {previous.source_line} and {item.source_line}"
            )
        seen[item.document_key] = item


def _read_url_file(
    path: Path,
    *,
    policy: FetchSecurityPolicy | None = None,
) -> list[RemoteUrlSourceItem]:
    _reject_unsafe_url_file_path(path)
    if path.stat().st_size > REMOTE_URL_FILE_MAX_BYTES:
        raise ValueError(
            f"URL file is too large: {path} exceeds {REMOTE_URL_FILE_MAX_BYTES} bytes"
        )
    with path.open(encoding="utf-8") as handle:
        return _read_url_values(handle, policy=policy, error_label="line")


def _reject_unsafe_url_file_path(path: Path) -> None:
    if path_has_symlink_segment(path):
        raise ValueError("URL file path must not include symlink segments")
    if is_multi_link_regular_file(path):
        raise ValueError("URL file path must not be a multi-link file")


def _read_url_values(
    values: Iterable[str],
    *,
    policy: FetchSecurityPolicy | None = None,
    error_label: str,
) -> list[RemoteUrlSourceItem]:
    items: list[RemoteUrlSourceItem] = []
    for line_number, line in enumerate(values, 1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if len(items) >= REMOTE_URL_LIST_MAX_ITEMS:
            raise ValueError(
                f"URL list has more than {REMOTE_URL_LIST_MAX_ITEMS} entries"
            )
        try:
            items.append(_source_item(raw, line_number=line_number, policy=policy))
        except ValueError as exc:
            raise ValueError(
                f"URL list {error_label} {line_number}: {cli_error_message(exc)}"
            ) from exc
    return items


def _source_item(
    url: str,
    *,
    line_number: int,
    policy: FetchSecurityPolicy | None = None,
) -> RemoteUrlSourceItem:
    validated = validate_fetch_url(url, policy=policy)
    raw = url.strip()
    return RemoteUrlSourceItem(
        url=raw,
        redacted_url=safe_remote_source_url(validated),
        document_key=_document_key(validated),
        query_sha256=validated.query_sha256,
        source_line=line_number,
        raw_query=urlsplit(raw).query,
    )


def _document_key(url: ValidatedFetchUrl) -> str:
    key = f"url:{safe_remote_source_url(url)}"
    if isinstance(url.query_sha256, str) and url.query_sha256:
        return f"{key}|query_sha256:{url.query_sha256}"
    return key
