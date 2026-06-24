from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import rag_core.ingest.urls.sources as remote_ingest_sources_module
from rag_core.events import (
    EventBuffer,
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
)
from rag_core.core_models import CollectionManifestEntry, IngestedDocument
from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
from rag_core.manifest.persistence import write_entry
from rag_core.ingest.urls import (
    REMOTE_INGEST_MAX_CONCURRENCY_CAP,
    RemoteUrlIngestRequest,
    build_remote_url_ingest_plan,
    reconcile_remote_url_ingest_plan,
    run_remote_url_ingest,
)
from rag_core.ingest.urls.models import RemoteUrlIngestResult, RemoteUrlIngestSuccess

ALLOW_HTTP_POLICY = FetchSecurityPolicy(allowed_schemes=("https", "http"))
ALLOW_HTTP_PRIVATE_POLICY = FetchSecurityPolicy(
    allowed_schemes=("https", "http"),
    allow_private_addresses=True,
)


class _FakeRemoteUrlCore:
    def __init__(self, *, fail_on: str = "") -> None:
        self.fail_on = fail_on
        self.ensure_ready_called = False
        self.closed = False
        self.ingest_calls: list[dict[str, Any]] = []

    async def ensure_ready(self) -> None:
        self.ensure_ready_called = True

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        self.ingest_calls.append({"url": url, **kwargs})
        if self.fail_on and self.fail_on in url:
            raise RuntimeError(f"fetch exploded for {url}")
        slug = url.split("/", 3)[-1].split("?", 1)[0].replace("/", "-") or "index"
        redacted_url = url.split("?", 1)[0] + ("?redacted" if "?" in url else "")
        return IngestedDocument(
            document_id=f"doc-{len(self.ingest_calls)}",
            namespace=kwargs["namespace"],
            collection=kwargs["collection"],
            chunk_count=2,
            filename=f"{slug}.txt",
            mime_type="text/plain",
            document_key=f"url:{redacted_url}",
            content_sha256=f"hash-{len(self.ingest_calls)}",
            ingest_state="created",
            replaced_existing=False,
            metadata={"source_type": "url", "source_url": redacted_url},
        )

    async def close(self) -> None:
        self.closed = True


class _FakeOpenAIError(Exception):
    __module__ = "openai"


class _BootstrapErrorRemoteUrlCore(_FakeRemoteUrlCore):
    def __init__(self, *, fail_url: str) -> None:
        super().__init__()
        self.fail_url = fail_url

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        if url == self.fail_url:
            self.ingest_calls.append({"url": url, **kwargs})
            raise _FakeOpenAIError("raw api_key client option OPENAI_API_KEY secret")
        return await super().add_url(url, **kwargs)


class _SiblingAbortRemoteUrlCore(_FakeRemoteUrlCore):
    def __init__(self, *, fail_url: str) -> None:
        super().__init__()
        self.fail_url = fail_url
        self.active = 0
        self.closed_while_active = False
        self.sibling_started = asyncio.Event()
        self.release_sibling = asyncio.Event()

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        self.ingest_calls.append({"url": url, **kwargs})
        self.active += 1
        try:
            if url == self.fail_url:
                await self.sibling_started.wait()
                raise _FakeOpenAIError("raw api_key client option OPENAI_API_KEY secret")
            self.sibling_started.set()
            await self.release_sibling.wait()
            slug = url.rsplit("/", 1)[-1]
            return IngestedDocument(
                document_id=f"doc-{slug}",
                namespace=kwargs["namespace"],
                collection=kwargs["collection"],
                chunk_count=1,
                filename=f"{slug}.txt",
                mime_type="text/plain",
                document_key=f"url:{url}",
                content_sha256=f"hash-{slug}",
                ingest_state="created",
                metadata={"source_url": url},
            )
        finally:
            self.active -= 1

    async def close(self) -> None:
        self.closed_while_active = self.active > 0
        self.release_sibling.set()
        await super().close()


class _CanonicalizingRemoteUrlCore(_FakeRemoteUrlCore):
    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        self.ingest_calls.append({"url": url, **kwargs})
        return IngestedDocument(
            document_id="doc-final",
            namespace=kwargs["namespace"],
            collection=kwargs["collection"],
            chunk_count=2,
            filename="final.txt",
            mime_type="text/plain",
            document_key="url:https://example.com/docs/final",
            content_sha256="hash-final",
            ingest_state="created",
            replaced_existing=False,
            metadata={
                "source_type": "url",
                "source_url": "https://example.com/docs/final",
            },
        )


class _MixedStateRemoteUrlCore(_FakeRemoteUrlCore):
    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        document = await super().add_url(url, **kwargs)
        ingest_state = "unchanged" if "/reference" in url else "created"
        return replace(document, ingest_state=ingest_state)


class _UnsafeMetadataRemoteUrlCore(_FakeRemoteUrlCore):
    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        self.ingest_calls.append({"url": url, **kwargs})
        return IngestedDocument(
            document_id="doc-final",
            namespace=kwargs["namespace"],
            collection=kwargs["collection"],
            chunk_count=2,
            filename="final.txt",
            mime_type="text/plain",
            document_key="url:https://example.com/docs/final?redacted",
            content_sha256="hash-final",
            ingest_state="created",
            replaced_existing=False,
            metadata={
                "source_type": "url",
                "source_url": " https://example.com/docs/final?token=secret ",
            },
        )


def _manifest_entry(
    *,
    document_key: str,
    document_id: str = "doc-existing",
    content_sha256: str = "hash-1",
) -> CollectionManifestEntry:
    return CollectionManifestEntry(
        document_id=document_id,
        namespace="team-space",
        collection="help",
        document_key=document_key,
        content_sha256=content_sha256,
        filename="remote.txt",
        mime_type="text/plain",
        chunk_count=2,
        parser="remote:text",
        needs_ocr=False,
        metadata={"source_type": "url"},
    )


def test_build_remote_url_ingest_plan_validates_and_redacts_url_file(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "# docs",
                "https://example.com/docs/guide?private=alpha",
                "",
                "https://example.com/reference",
            ]
        ),
        encoding="utf-8",
    )

    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
        )
    )

    assert plan.url_count == 2
    assert plan.to_payload()["source_type"] == "url"
    assert plan.urls[0].redacted_url == "https://example.com/docs/guide?redacted"
    assert plan.urls[0].source_line == 2
    assert plan.urls[0].query_sha256 is not None
    assert plan.urls[1].redacted_url == "https://example.com/reference"
    assert "private=alpha" not in repr(plan)
    assert "private=alpha" not in repr(plan.to_payload())


def test_build_remote_url_ingest_plan_accepts_inline_urls() -> None:
    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            namespace="team-space",
            collection="help",
            urls=(
                "# docs",
                "https://example.com/docs/guide?private=alpha",
                "",
                "https://example.com/reference",
            ),
        )
    )

    payload = plan.to_payload()
    assert plan.url_count == 2
    assert payload["url_source"] == "inline"
    assert "url_file" not in payload
    assert plan.urls[0].redacted_url == "https://example.com/docs/guide?redacted"
    assert plan.urls[0].source_line == 2
    assert plan.urls[1].redacted_url == "https://example.com/reference"
    assert "private=alpha" not in repr(plan)


def test_build_remote_url_ingest_plan_distinguishes_redacted_query_identities() -> None:
    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            namespace="team-space",
            collection="help",
            urls=(
                "https://example.com/export?id=1",
                "https://example.com/export?id=2",
            ),
        )
    )

    assert [item.redacted_url for item in plan.urls] == [
        "https://example.com/export?redacted",
        "https://example.com/export?redacted",
    ]
    assert plan.urls[0].document_key != plan.urls[1].document_key
    payload = plan.to_payload()
    urls = cast(list[dict[str, object]], payload["urls"])
    assert [item["document_key"] for item in urls] == [
        "url:https://example.com/export?redacted",
        "url:https://example.com/export?redacted",
    ]
    assert all(item["has_private_query_identity"] is True for item in urls)
    assert "query_sha256" not in repr(payload)
    private_payload = plan.to_payload(include_private=True)
    private_urls = cast(list[dict[str, object]], private_payload["urls"])
    assert private_urls[0]["document_key"] == plan.urls[0].document_key


def test_remote_url_ingest_result_payload_hides_query_identity_hash() -> None:
    record = RemoteUrlIngestSuccess(
        requested_url="https://example.com/export?redacted",
        source_url="https://example.com/export?redacted",
        document_key="url:https://example.com/export?redacted|query_sha256:abc123",
        content_sha256="sha",
        document_id="doc-1",
        filename="export.json",
        chunk_count=1,
        ingest_state="created",
        replaced_existing=False,
    )
    result = RemoteUrlIngestResult(
        namespace="team-space",
        collection="help",
        records=(record,),
    )

    public_payload = result.to_payload()
    private_payload = result.to_payload(include_private=True)

    public_record = cast(list[dict[str, object]], public_payload["records"])[0]
    private_record = cast(list[dict[str, object]], private_payload["records"])[0]
    assert public_record["document_key"] == "url:https://example.com/export?redacted"
    assert public_record["has_private_query_identity"] is True
    assert "query_sha256" not in repr(public_payload)
    assert private_record["document_key"] == record.document_key


def test_build_remote_url_ingest_plan_rejects_dot_segment_alias_document_keys() -> None:
    with pytest.raises(ValueError, match="same document key"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                namespace="team-space",
                collection="help",
                urls=(
                    "https://example.com/docs/guide",
                    "https://example.com/docs/./archive/%2e%2e/guide",
                ),
            )
        )


def test_build_remote_url_ingest_plan_requires_one_url_source(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/docs\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no URLs found in inline URL list"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(namespace="team-space", collection="help")
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                namespace="team-space",
                collection="help",
                url_file=url_file,
                urls=("https://example.com/docs",),
            )
        )


def test_build_remote_url_ingest_plan_rejects_invalid_url_before_runtime(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("file:///tmp/private.md\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="URL list line 1: unsupported fetch URL scheme"
    ):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            )
        )


def test_build_remote_url_ingest_plan_rejects_oversized_url_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_ingest_sources_module, "REMOTE_URL_FILE_MAX_BYTES", 16)
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/docs\n", encoding="utf-8")

    with pytest.raises(ValueError, match="URL file is too large"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            )
        )


def test_build_remote_url_ingest_plan_rejects_symlink_url_file(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable on this platform")
    target = tmp_path / "urls.txt"
    target.write_text("https://example.com/docs\n", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=alias,
                namespace="team-space",
                collection="help",
            )
        )


def test_build_remote_url_ingest_plan_rejects_hardlinked_url_file(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    source = tmp_path / "urls.txt"
    source.write_text("https://example.com/docs\n", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink support unavailable: {exc}")

    with pytest.raises(ValueError, match="multi-link"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=alias,
                namespace="team-space",
                collection="help",
            )
        )


def test_build_remote_url_ingest_plan_rejects_too_many_url_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_ingest_sources_module, "REMOTE_URL_LIST_MAX_ITEMS", 1)
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "https://example.com/docs\nhttps://example.com/reference\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="URL list has more than 1 entries"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            )
        )


def test_build_remote_url_ingest_plan_rejects_duplicate_document_keys(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com/docs",
                "https://example.com/docs",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lines 1 and 2"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            )
        )


@pytest.mark.parametrize(
    ("first_url", "second_url"),
    [
        ("https://example.com/docs", "https://example.com:443/docs"),
        ("http://example.org/docs", "http://example.org:80/docs"),
    ],
)
def test_build_remote_url_ingest_plan_rejects_default_port_duplicates(
    tmp_path: Path,
    first_url: str,
    second_url: str,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                first_url,
                second_url,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lines 1 and 2"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
                fetch_policy=ALLOW_HTTP_POLICY
                if first_url.startswith("http://")
                else None,
            )
        )


def test_build_remote_url_ingest_plan_rejects_empty_root_path_duplicates(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com",
                "https://example.com/",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lines 1 and 2"):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            )
        )


def test_build_remote_url_ingest_plan_private_addresses_are_explicit_opt_in(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("http://localhost:8000/docs?private=alpha\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="URL list line 1: .*HTTP requires explicit opt-in"
    ):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            )
        )

    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
            fetch_policy=ALLOW_HTTP_PRIVATE_POLICY,
        )
    )

    assert plan.urls[0].redacted_url == "http://localhost:8000/docs?redacted"
    assert "private=alpha" not in repr(plan)


def test_reconcile_remote_url_ingest_plan_reports_manifest_state(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com/docs/guide?private=alpha",
                "https://example.com/reference",
            ]
        ),
        encoding="utf-8",
    )
    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
        )
    )
    manifest_dir = tmp_path / "manifest"
    write_entry(
        manifest_dir,
        _manifest_entry(document_key=plan.urls[0].document_key),
    )

    reconciliation = reconcile_remote_url_ingest_plan(plan, manifest_dir=manifest_dir)
    payload = plan.to_payload(reconciliation=reconciliation)
    reconciliation_payload = cast(dict[str, object], payload["reconciliation"])
    summary = cast(dict[str, object], reconciliation_payload["summary"])
    urls = cast(list[dict[str, object]], payload["urls"])

    assert summary["unchanged_count"] == 1
    assert summary["missing_count"] == 1
    assert urls[0]["manifest_status"] == "unchanged"
    assert urls[0]["manifest_reason"] == "present_without_hash_check"
    assert urls[1]["manifest_status"] == "unknown_until_fetch"
    assert urls[1]["manifest_reason"] == "canonical_url_unknown_until_fetch"
    assert "private=alpha" not in repr(payload)


def test_reconcile_remote_url_ingest_plan_keeps_redirect_targets_unknown_until_fetch(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/docs/redirect\n", encoding="utf-8")
    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
        )
    )
    manifest_dir = tmp_path / "manifest"
    write_entry(
        manifest_dir,
        _manifest_entry(
            document_key="url:https://example.com/docs/final",
            document_id="doc-final",
            content_sha256="hash-final",
        ),
    )

    reconciliation = reconcile_remote_url_ingest_plan(plan, manifest_dir=manifest_dir)
    payload = plan.to_payload(reconciliation=reconciliation)
    urls = cast(list[dict[str, object]], payload["urls"])

    assert urls[0]["manifest_status"] == "unknown_until_fetch"
    assert urls[0]["manifest_reason"] == "canonical_url_unknown_until_fetch"


def test_run_remote_url_ingest_preserves_order_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "\n".join(
                [
                    "https://example.com/docs/guide?private=alpha",
                    "https://example.com/docs/fail?private=beta",
                    "https://example.com/docs/reference",
                ]
            ),
            encoding="utf-8",
        )
        events = EventBuffer()
        core = _FakeRemoteUrlCore(fail_on="/fail")

        result = await run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
                metadata={"team": "docs"},
                force_reindex=True,
                max_concurrency=2,
            ),
            core_factory=lambda: core,
            event_sink=events,
        )

        assert core.ensure_ready_called is True
        assert core.closed is True
        assert [call["url"] for call in core.ingest_calls] == [
            "https://example.com/docs/guide?private=alpha",
            "https://example.com/docs/fail?private=beta",
            "https://example.com/docs/reference",
        ]
        assert all(call["metadata"] == {"team": "docs"} for call in core.ingest_calls)
        assert all(call["force_reindex"] is True for call in core.ingest_calls)
        assert result.written_count == 2
        assert result.failed_count == 1
        assert [record.to_payload()["ok"] for record in result.records] == [
            True,
            False,
            True,
        ]
        failed = result.failed[0]
        assert failed.requested_url == "https://example.com/docs/fail?redacted"
        assert (
            failed.error
            == "RuntimeError while ingesting https://example.com/docs/fail?redacted"
        )
        assert "fetch exploded" not in failed.error
        assert "private=beta" not in failed.error
        assert "private=beta" not in repr(result)

        completed = events.by_type("ingest.batch.completed")
        assert len(completed) == 1
        assert isinstance(completed[0], IngestBatchCompleted)
        assert completed[0].planned_count == 3
        assert completed[0].succeeded_count == 2
        assert completed[0].failed_count == 1

        progress = events.by_type("ingest.batch.progress")
        assert len(progress) == 3
        assert all(isinstance(event, IngestBatchProgress) for event in progress)
        assert "private=alpha" not in repr(events.events)
        assert "private=beta" not in repr(events.events)

    asyncio.run(_run())


def test_run_remote_url_ingest_batch_failed_preserves_partial_provider_abort_counts(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "\n".join(
                [
                    "https://example.com/docs/guide",
                    "https://example.com/docs/broken",
                    "https://example.com/docs/reference",
                ]
            ),
            encoding="utf-8",
        )
        core = _BootstrapErrorRemoteUrlCore(
            fail_url="https://example.com/docs/broken"
        )
        events = EventBuffer()

        with pytest.raises(ValueError) as exc_info:
            await run_remote_url_ingest(
                RemoteUrlIngestRequest(
                    url_file=url_file,
                    namespace="team-space",
                    collection="help",
                    max_concurrency=1,
                ),
                core_factory=lambda: core,
                event_sink=events,
            )

        assert "provider failed during ingest" in str(exc_info.value)
        assert "OPENAI_API_KEY" not in str(exc_info.value)
        assert [call["url"] for call in core.ingest_calls] == [
            "https://example.com/docs/guide",
            "https://example.com/docs/broken",
        ]
        assert core.closed is True
        progress = [
            event for event in events.events if isinstance(event, IngestBatchProgress)
        ]
        assert len(progress) == 1
        assert progress[0].status == "succeeded"
        assert progress[0].filename == "https://example.com/docs/guide"
        failed = [
            event for event in events.events if isinstance(event, IngestBatchFailed)
        ]
        assert len(failed) == 1
        assert failed[0].planned_count == 3
        assert failed[0].completed_count == 1
        assert failed[0].succeeded_count == 1
        assert failed[0].failed_count == 0
        assert failed[0].error == "ProviderCliError"
        assert "OPENAI_API_KEY" not in str(failed[0])

    asyncio.run(_run())


def test_run_remote_url_ingest_drains_sibling_workers_before_close(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "\n".join(
                [
                    "https://example.com/docs/a",
                    "https://example.com/docs/b",
                ]
            ),
            encoding="utf-8",
        )
        core = _SiblingAbortRemoteUrlCore(fail_url="https://example.com/docs/a")

        with pytest.raises(ValueError) as exc_info:
            await run_remote_url_ingest(
                RemoteUrlIngestRequest(
                    url_file=url_file,
                    namespace="team-space",
                    collection="help",
                    max_concurrency=2,
                ),
                core_factory=lambda: core,
            )

        assert "provider failed during ingest" in str(exc_info.value)
        assert core.closed is True
        assert core.closed_while_active is False

    asyncio.run(_run())


def test_run_remote_url_ingest_distinguishes_succeeded_written_and_skipped(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "\n".join(
                [
                    "https://example.com/docs/guide",
                    "https://example.com/docs/reference",
                ]
            ),
            encoding="utf-8",
        )

        result = await run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            ),
            core_factory=_MixedStateRemoteUrlCore,
        )

        assert result.succeeded_count == 2
        assert result.written_count == 1
        assert result.skipped_count == 1
        assert result.failed_count == 0
        assert [record.ingest_state for record in result.succeeded] == [
            "created",
            "unchanged",
        ]
        assert [record.source_url for record in result.written] == [
            "https://example.com/docs/guide",
        ]
        assert [record.source_url for record in result.skipped] == [
            "https://example.com/docs/reference",
        ]

        payload = result.to_payload()
        assert payload["succeeded_count"] == 2
        assert payload["written_count"] == 1
        assert payload["skipped_count"] == 1
        succeeded = cast(list[dict[str, object]], payload["succeeded"])
        written = cast(list[dict[str, object]], payload["written"])
        skipped = cast(list[dict[str, object]], payload["skipped"])
        assert [record["ingest_state"] for record in succeeded] == [
            "created",
            "unchanged",
        ]
        assert [record["ingest_state"] for record in written] == ["created"]
        assert [record["ingest_state"] for record in skipped] == ["unchanged"]

    asyncio.run(_run())


def test_run_remote_url_ingest_reports_manifest_statuses(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "\n".join(
                [
                    "https://example.com/docs/guide",
                    "https://example.com/docs/reference",
                ]
            ),
            encoding="utf-8",
        )
        request = RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
        )
        plan = build_remote_url_ingest_plan(request)
        manifest_dir = tmp_path / "manifest"
        write_entry(
            manifest_dir,
            _manifest_entry(
                document_key=plan.urls[0].document_key,
                document_id="doc-guide",
                content_sha256="hash-1",
            ),
        )
        write_entry(
            manifest_dir,
            _manifest_entry(
                document_key=plan.urls[1].document_key,
                document_id="doc-reference",
                content_sha256="old-hash",
            ),
        )
        events = EventBuffer()

        result = await run_remote_url_ingest(
            request,
            core_factory=lambda: _FakeRemoteUrlCore(),
            event_sink=events,
            manifest_dir=manifest_dir,
        )

        assert [
            (record.manifest_status, record.manifest_reason)
            for record in result.records
        ] == [
            ("unchanged", "content_sha256_match"),
            ("changed", "content_sha256_changed"),
        ]
        progress = events.by_type("ingest.batch.progress")
        assert len(progress) == 2
        assert all(isinstance(event, IngestBatchProgress) for event in progress)
        typed_progress = cast(list[IngestBatchProgress], progress)
        assert [
            (event.manifest_status, event.manifest_reason) for event in typed_progress
        ] == [
            ("unchanged", "content_sha256_match"),
            ("changed", "content_sha256_changed"),
        ]

    asyncio.run(_run())


def test_run_remote_url_ingest_reconciles_canonicalized_document_key(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com/docs/redirect\n",
            encoding="utf-8",
        )
        request = RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
        )
        manifest_dir = tmp_path / "manifest"
        write_entry(
            manifest_dir,
            _manifest_entry(
                document_key="url:https://example.com/docs/final",
                document_id="doc-final",
                content_sha256="hash-final",
            ),
        )

        result = await run_remote_url_ingest(
            request,
            core_factory=lambda: _CanonicalizingRemoteUrlCore(),
            manifest_dir=manifest_dir,
        )

        [record] = result.records
        assert record.document_key == "url:https://example.com/docs/final"
        assert record.manifest_status == "unchanged"
        assert record.manifest_reason == "content_sha256_match"

    asyncio.run(_run())


def test_run_remote_url_ingest_sanitizes_app_owned_source_urls(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com/docs/redirect\n",
            encoding="utf-8",
        )

        result = await run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            ),
            core_factory=lambda: _UnsafeMetadataRemoteUrlCore(),
        )

        [record] = result.records
        assert isinstance(record, RemoteUrlIngestSuccess)
        assert record.source_url == "https://example.com/docs/final?redacted"
        assert "token=secret" not in repr(result)

    asyncio.run(_run())


def test_run_remote_url_ingest_preserves_duplicate_manifest_status(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com/docs/guide\n",
            encoding="utf-8",
        )
        request = RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="team-space",
            collection="help",
        )
        plan = build_remote_url_ingest_plan(request)
        manifest_dir = tmp_path / "manifest"
        write_entry(
            manifest_dir,
            _manifest_entry(
                document_key=plan.urls[0].document_key,
                document_id="doc-a",
                content_sha256="old-hash-a",
            ),
        )
        write_entry(
            manifest_dir,
            _manifest_entry(
                document_key=plan.urls[0].document_key,
                document_id="doc-b",
                content_sha256="old-hash-b",
            ),
        )

        result = await run_remote_url_ingest(
            request,
            core_factory=lambda: _FakeRemoteUrlCore(),
            manifest_dir=manifest_dir,
        )

        [record] = result.records
        assert record.manifest_status == "duplicate"
        assert record.manifest_reason == "duplicate_manifest_document_key"

    asyncio.run(_run())


def test_run_remote_url_ingest_passes_request_fetch_policy_and_limits(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "http://localhost:8000/docs?private=alpha\n", encoding="utf-8"
        )
        core = _FakeRemoteUrlCore()

        result = await run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
                fetch_policy=ALLOW_HTTP_PRIVATE_POLICY,
                fetch_limits=FetchLimits(max_bytes=1024, timeout_seconds=2.5),
            ),
            core_factory=lambda: core,
        )

        assert result.written_count == 1
        [call] = core.ingest_calls
        assert call.get("fetch_client") is None
        assert call["fetch_policy"] == ALLOW_HTTP_PRIVATE_POLICY
        assert call["fetch_limits"] == FetchLimits(max_bytes=1024, timeout_seconds=2.5)
        assert "private=alpha" not in repr(result)

    asyncio.run(_run())


def test_run_remote_url_ingest_passes_custom_fetch_client_without_request_controls(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/docs\n", encoding="utf-8")
        core = _FakeRemoteUrlCore()
        fetch_client = object()

        result = await run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="team-space",
                collection="help",
            ),
            core_factory=lambda: core,
            fetch_client=fetch_client,  # type: ignore[arg-type]
        )

        assert result.written_count == 1
        [call] = core.ingest_calls
        assert call["fetch_client"] is fetch_client
        assert "fetch_policy" not in call
        assert "fetch_limits" not in call

    asyncio.run(_run())


def test_run_remote_url_ingest_rejects_request_fetch_policy_with_custom_fetch_client(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text("http://127.0.0.1/docs\n", encoding="utf-8")
        core = _FakeRemoteUrlCore()
        fetch_client = object()
        policy = ALLOW_HTTP_PRIVATE_POLICY

        with pytest.raises(
            ValueError,
            match="fetch_client cannot be combined with request fetch_policy",
        ):
            await run_remote_url_ingest(
                RemoteUrlIngestRequest(
                    url_file=url_file,
                    namespace="team-space",
                    collection="help",
                    fetch_policy=policy,
                ),
                core_factory=lambda: core,
                fetch_client=fetch_client,  # type: ignore[arg-type]
            )

        assert core.ingest_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("fetch_policy", "fetch_limits"),
    [
        (None, FetchLimits(max_bytes=1024)),
        (FetchSecurityPolicy(), FetchLimits(max_bytes=1024)),
    ],
)
def test_run_remote_url_ingest_rejects_ambiguous_fetch_configuration(
    tmp_path: Path,
    fetch_policy: FetchSecurityPolicy | None,
    fetch_limits: FetchLimits | None,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/docs\n", encoding="utf-8")
        core = _FakeRemoteUrlCore()

        with pytest.raises(
            ValueError,
            match="fetch_client cannot be combined with request fetch_(policy|limits)",
        ):
            await run_remote_url_ingest(
                RemoteUrlIngestRequest(
                    url_file=url_file,
                    namespace="team-space",
                    collection="help",
                    fetch_policy=fetch_policy,
                    fetch_limits=fetch_limits,
                ),
                core_factory=lambda: core,
                fetch_client=object(),  # type: ignore[arg-type]
            )

        assert core.ensure_ready_called is False
        assert core.ingest_calls == []

    asyncio.run(_run())


def test_run_remote_url_ingest_rejects_bad_concurrency_before_core_setup(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/docs\n", encoding="utf-8")

        def fail_core() -> _FakeRemoteUrlCore:
            raise AssertionError("core should not be constructed")

        with pytest.raises(ValueError, match="max_concurrency must be between 1 and"):
            await run_remote_url_ingest(
                RemoteUrlIngestRequest(
                    url_file=url_file,
                    namespace="team-space",
                    collection="help",
                    max_concurrency=0,
                ),
                core_factory=fail_core,
            )

    asyncio.run(_run())


def test_max_concurrency_above_cap_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=rf"max_concurrency must be between 1 and {REMOTE_INGEST_MAX_CONCURRENCY_CAP}",
    ):
        build_remote_url_ingest_plan(
            RemoteUrlIngestRequest(
                namespace="team-space",
                collection="help",
                urls=["https://example.com/docs"],
                max_concurrency=REMOTE_INGEST_MAX_CONCURRENCY_CAP + 1,
            )
        )

    # Exactly at the cap succeeds
    plan = build_remote_url_ingest_plan(
        RemoteUrlIngestRequest(
            namespace="team-space",
            collection="help",
            urls=["https://example.com/docs"],
            max_concurrency=REMOTE_INGEST_MAX_CONCURRENCY_CAP,
        )
    )
    assert plan.url_count == 1
