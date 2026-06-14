"""URL-list ingest binds URLs to a corpus via LicensedSourceRegistry.

A URL whose host is not licensed for the target corpus_id must be rejected
with ``LicensedSourceMismatch`` before any fetch happens. The rejection must
*surface*. Silent skipping would let licensed-tier ingest quietly drop a
procurement-bound URL and then fail open on the lower tier later.
"""

from __future__ import annotations

import pytest

from rag_core.licensed_source_registry import LicensedSourceRegistry
from rag_core.remote_ingest_models import RemoteUrlIngestRequest
from rag_core.remote_ingest_sources import remote_url_source_items


def _registry() -> LicensedSourceRegistry:
    registry = LicensedSourceRegistry.from_mapping(
        {
            "public": ("docs.example.com", "*.public.example"),
            "licensed": ("vendor.licensed.example",),
            "restricted": (),
        }
    )
    assert registry is not None
    return registry


def test_registry_accepts_exact_host_for_corpus() -> None:
    request = RemoteUrlIngestRequest(
        namespace="ws",
        corpus_id="licensed",
        urls=("https://vendor.licensed.example/doc.pdf",),
        licensed_source_registry=_registry(),
    )

    items, source_kind, source_path = remote_url_source_items(request)

    assert source_kind == "inline"
    assert source_path is None
    assert len(items) == 1
    assert items[0].redacted_url == "https://vendor.licensed.example/doc.pdf"


def test_registry_rejects_unlicensed_host_for_corpus() -> None:
    request = RemoteUrlIngestRequest(
        namespace="ws",
        corpus_id="licensed",
        urls=("https://docs.example.com/leak.pdf",),
        licensed_source_registry=_registry(),
    )

    with pytest.raises(ValueError, match="not licensed for"):
        remote_url_source_items(request)


def test_registry_rejects_when_corpus_id_has_no_entry() -> None:
    request = RemoteUrlIngestRequest(
        namespace="ws",
        corpus_id="undeclared_corpus",
        urls=("https://docs.example.com/x.pdf",),
        licensed_source_registry=_registry(),
    )

    with pytest.raises(ValueError, match="has no entry for corpus_id"):
        remote_url_source_items(request)


def test_registry_empty_corpus_denies_everything() -> None:
    request = RemoteUrlIngestRequest(
        namespace="ws",
        corpus_id="restricted",
        urls=("https://vendor.licensed.example/doc.pdf",),
        licensed_source_registry=_registry(),
    )

    with pytest.raises(ValueError, match="denies every host"):
        remote_url_source_items(request)


def test_registry_wildcard_matches_subdomains() -> None:
    request = RemoteUrlIngestRequest(
        namespace="ws",
        corpus_id="public",
        urls=("https://docs.public.example/page",),
        licensed_source_registry=_registry(),
    )

    items, _, _ = remote_url_source_items(request)

    assert len(items) == 1


def test_registry_from_mapping_returns_none_for_none() -> None:
    assert LicensedSourceRegistry.from_mapping(None) is None


def test_registry_rejects_blank_corpus_id_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LicensedSourceRegistry.from_mapping({"": ("docs.example.com",)})


def test_url_list_first_mismatch_surfaces_with_line_number() -> None:
    # Surface mismatches with their list-line number so callers can correlate
    # which URL in the upload broke the binding.
    request = RemoteUrlIngestRequest(
        namespace="ws",
        corpus_id="licensed",
        urls=(
            "https://vendor.licensed.example/ok.pdf",
            "https://docs.example.com/forbidden.pdf",
        ),
        licensed_source_registry=_registry(),
    )

    with pytest.raises(ValueError, match=r"URL list item 2"):
        remote_url_source_items(request)
