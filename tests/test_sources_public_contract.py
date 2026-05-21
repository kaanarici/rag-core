from rag_core import sources
from rag_core.sources import (
    ArchiveLimits,
    ArchiveSourceItem,
    ArchiveSourcePlan,
    LocalFileSourceReader,
    LocalSourceItem,
    LocalSourcePlan,
    RemoteDiscoveredUrl,
    RemoteDiscovery,
    RemoteDiscoveryReader,
    RemoteSourceDocument,
    RemoteUrlSourceReader,
    ZipArchiveSourceReader,
    parse_llms_txt_urls,
    parse_sitemap_urls,
)


def test_source_primitives_are_sources_namespace_public_imports() -> None:
    assert ArchiveLimits.__name__ == "ArchiveLimits"
    assert ArchiveSourceItem.__name__ == "ArchiveSourceItem"
    assert ArchiveSourcePlan.__name__ == "ArchiveSourcePlan"
    assert LocalFileSourceReader.__name__ == "LocalFileSourceReader"
    assert LocalSourceItem.__name__ == "LocalSourceItem"
    assert LocalSourcePlan.__name__ == "LocalSourcePlan"
    assert RemoteDiscoveredUrl.__name__ == "RemoteDiscoveredUrl"
    assert RemoteDiscovery.__name__ == "RemoteDiscovery"
    assert RemoteDiscoveryReader.__name__ == "RemoteDiscoveryReader"
    assert RemoteSourceDocument.__name__ == "RemoteSourceDocument"
    assert RemoteUrlSourceReader.__name__ == "RemoteUrlSourceReader"
    assert ZipArchiveSourceReader.__name__ == "ZipArchiveSourceReader"
    assert callable(parse_llms_txt_urls)
    assert callable(parse_sitemap_urls)


def test_remote_discovery_exposes_fetchable_and_redacted_url_sequences() -> None:
    discovery = parse_llms_txt_urls(
        "- [Guide](/guide?private=alpha)\n",
        base_url="https://example.com/llms.txt",
    )

    assert discovery.urls == ("https://example.com/guide?private=alpha",)
    assert discovery.redacted_urls == ("https://example.com/guide?redacted",)
    assert "private=alpha" not in repr(discovery)


def test_sources_namespace_is_curated() -> None:
    assert sources.__all__ == [
        "ArchiveLimits",
        "ArchiveSourceItem",
        "ArchiveSourcePlan",
        "LocalFileSourceReader",
        "LocalSourceItem",
        "LocalSourcePlan",
        "RemoteDiscoveredUrl",
        "RemoteDiscovery",
        "RemoteDiscoveryKind",
        "RemoteDiscoveryReader",
        "RemoteSourceDocument",
        "RemoteUrlSourceReader",
        "ZipArchiveSourceReader",
        "archive_document_key",
        "document_key",
        "expand_supported_local_files",
        "file_content_sha256",
        "is_supported_archive_member_path",
        "is_ignored_local_file",
        "is_supported_local_candidate",
        "is_supported_local_file",
        "local_file_source_item",
        "local_source_key_root",
        "parse_llms_txt_urls",
        "parse_sitemap_urls",
        "read_zip_member_bytes",
        "remote_source_document",
        "safe_archive_member_path",
        "source_error_message",
        "write_discovered_url_file",
        "write_raw_discovered_url_file",
    ]
