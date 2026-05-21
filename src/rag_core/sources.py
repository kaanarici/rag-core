from __future__ import annotations

from pathlib import Path

import rag_core.local_sources as _local_sources
from rag_core.archive_sources import (
    ArchiveLimits,
    ArchiveSourceItem,
    ArchiveSourcePlan,
    ZipArchiveSourceReader,
    archive_document_key,
    is_supported_archive_member_path,
    read_zip_member_bytes,
    safe_archive_member_path,
)
from rag_core.local_sources import (
    LocalSourceItem,
    LocalSourcePlan,
    document_key,
    expand_supported_local_files,
    is_ignored_local_file,
    is_supported_local_candidate,
    is_supported_local_file,
    local_source_key_root,
    source_error_message,
)
from rag_core.remote_discovery import (
    RemoteDiscoveredUrl,
    RemoteDiscovery,
    RemoteDiscoveryKind,
    RemoteDiscoveryReader,
    parse_llms_txt_urls,
    parse_sitemap_urls,
    write_discovered_url_file,
    write_raw_discovered_url_file,
)
from rag_core.remote_sources import (
    RemoteSourceDocument,
    RemoteUrlSourceReader,
    remote_source_document,
)


class LocalFileSourceReader:
    def read(self, source: str | Path) -> LocalSourcePlan:
        return _local_sources.read_local_source(source, hash_file=file_content_sha256)


def local_file_source_item(path: Path, *, root: Path) -> LocalSourceItem:
    return _local_sources.local_file_source_item(
        path,
        root=root,
        hash_file=file_content_sha256,
    )


file_content_sha256 = _local_sources.file_content_sha256


__all__ = [
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
