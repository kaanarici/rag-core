from __future__ import annotations

from typing import Final

MANIFEST_REASON_CANONICAL_URL_UNKNOWN_UNTIL_FETCH: Final[str] = (
    "canonical_url_unknown_until_fetch"
)
MANIFEST_REASON_CONTENT_SHA256_CHANGED: Final[str] = "content_sha256_changed"
MANIFEST_REASON_CONTENT_SHA256_MATCH: Final[str] = "content_sha256_match"
MANIFEST_REASON_DUPLICATE_DOCUMENT_KEY: Final[str] = "duplicate_manifest_document_key"
MANIFEST_REASON_ENTRY_WITHOUT_SOURCE: Final[str] = "manifest_entry_without_source"
MANIFEST_REASON_NOT_CHECKED: Final[str] = "manifest_not_checked"
MANIFEST_REASON_PRESENT_WITHOUT_HASH_CHECK: Final[str] = "present_without_hash_check"
MANIFEST_REASON_SOURCE_NOT_IN_MANIFEST: Final[str] = "source_not_in_manifest"
MANIFEST_REASON_SOURCE_READ_FAILED: Final[str] = "source_read_failed"
