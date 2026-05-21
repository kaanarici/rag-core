from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import traceback
from pathlib import Path
from typing import cast

import pytest

from rag_core.core_models import CorpusManifestEntry
from rag_core.core_ingest_recovery import refreshed_manifest_entry
from rag_core.manifest_persistence import (
    ManifestCompactionResult,
    ManifestReadError,
    ManifestSource,
    compact_manifest,
    manifest_reconciliation_payload,
    manifest_path,
    read_entries,
    reconcile_entries,
    summarize_entries,
    write_entry,
    write_entry_if_stale,
)
from rag_core.private_files import reject_symlink_ancestors
from rag_core.search.types import StoredDocumentRecord


def _entry(
    *,
    document_id: str = "doc-1",
    namespace: str = "acme",
    corpus_id: str = "help",
    chunk_count: int = 3,
    parser: str | None = "local:converter",
    needs_ocr: bool = False,
    document_key: str | None = None,
    content_sha256: str = "sha256-stub",
    metadata: dict[str, object] | None = None,
) -> CorpusManifestEntry:
    return CorpusManifestEntry(
        document_id=document_id,
        namespace=namespace,
        corpus_id=corpus_id,
        document_key=document_key or f"{document_id}.md",
        content_sha256=content_sha256,
        filename=f"{document_id}.md",
        mime_type="text/markdown",
        chunk_count=chunk_count,
        parser=parser,
        needs_ocr=needs_ocr,
        metadata=metadata or {},
    )


def test_private_file_symlink_guard_allows_macos_system_aliases() -> None:
    reject_symlink_ancestors(Path("/var/folders/rag-core/manifest.jsonl"))
    reject_symlink_ancestors(Path("/tmp/rag-core/manifest.jsonl"))


def _manifest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_id": "doc-1",
        "namespace": "acme",
        "corpus_id": "help",
        "document_key": "doc-1.md",
        "content_sha256": "sha",
        "filename": "doc-1.md",
        "mime_type": "text/markdown",
        "chunk_count": 1,
        "parser": "local:converter",
        "needs_ocr": False,
        "metadata": {},
    }
    payload.update(overrides)
    return payload


def _write_manifest_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_manifest_path_layout(tmp_path: Path) -> None:
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    assert path == tmp_path / "acme" / "help.jsonl"


@pytest.mark.parametrize(
    ("namespace", "corpus_id"),
    [
        ("../escape", "help"),
        ("acme", "../escape"),
        ("", "help"),
        ("acme", ""),
        ("acme/slash", "help"),
        ("acme", "nested/slash"),
    ],
)
def test_manifest_path_rejects_non_segment_scope_values(
    tmp_path: Path, namespace: str, corpus_id: str
) -> None:
    with pytest.raises(ValueError, match="single non-empty path segment"):
        manifest_path(tmp_path, namespace=namespace, corpus_id=corpus_id)


@pytest.mark.parametrize(("namespace", "corpus_id"), [("Acme", "help"), ("acme", "Help")])
def test_manifest_path_rejects_mixed_case_scope_values(
    tmp_path: Path, namespace: str, corpus_id: str
) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        manifest_path(tmp_path, namespace=namespace, corpus_id=corpus_id)


@pytest.mark.parametrize("namespace", ["../escape", "", "acme/slash", " acme"])
def test_read_entries_rejects_non_segment_namespace(
    tmp_path: Path, namespace: str
) -> None:
    with pytest.raises(ValueError, match="single non-empty path segment"):
        read_entries(tmp_path, namespace=namespace)


@pytest.mark.parametrize("corpus_id", ["../escape", "", "nested/slash", "help "])
def test_read_entries_rejects_non_segment_corpus_even_when_namespace_missing(
    tmp_path: Path, corpus_id: str
) -> None:
    with pytest.raises(ValueError, match="single non-empty path segment"):
        read_entries(tmp_path, namespace="acme", corpus_id=corpus_id)


def test_write_entry_creates_directory_and_file(tmp_path: Path) -> None:
    write_entry(tmp_path, _entry())
    assert (tmp_path / "acme" / "help.jsonl").is_file()


def test_write_entry_rejects_symlinked_manifest_dir_ancestor(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="path parent must not be a symlink"):
        write_entry(alias_root, _entry())

    assert not (real_root / "acme").exists()


def test_write_entry_rejects_relative_path_from_symlinked_logical_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.chdir(real_root)
    monkeypatch.setenv("PWD", str(alias_root))

    with pytest.raises(ValueError, match="symlinked PWD"):
        write_entry(Path("manifest"), _entry())

    assert not (real_root / "manifest").exists()


def test_read_entries_rejects_symlinked_manifest_dir_ancestor(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="path parent must not be a symlink"):
        read_entries(alias_root, namespace="acme")


def test_write_entry_rejects_entries_that_cannot_be_read_back(tmp_path: Path) -> None:
    invalid = CorpusManifestEntry(
        document_id="doc-1",
        namespace="acme",
        corpus_id="help",
        document_key="doc-1.md",
        content_sha256="sha",
        filename="",
        mime_type="text/markdown",
        chunk_count=1,
    )

    with pytest.raises(ValueError, match="manifest entry"):
        write_entry(tmp_path, invalid)

    assert read_entries(tmp_path, namespace="acme", corpus_id="help") == []


def test_read_entries_returns_latest_per_document(tmp_path: Path) -> None:
    write_entry(tmp_path, _entry(chunk_count=2))
    write_entry(tmp_path, _entry(chunk_count=5))

    entries = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert len(entries) == 1
    assert entries[0].chunk_count == 5


def test_read_entries_per_corpus_isolation(tmp_path: Path) -> None:
    write_entry(tmp_path, _entry(corpus_id="help", document_id="d1"))
    write_entry(tmp_path, _entry(corpus_id="docs", document_id="d2"))

    help_only = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert [e.document_id for e in help_only] == ["d1"]

    docs_only = read_entries(tmp_path, namespace="acme", corpus_id="docs")
    assert [e.document_id for e in docs_only] == ["d2"]


def test_read_entries_namespace_wide_merges_corpora(tmp_path: Path) -> None:
    write_entry(tmp_path, _entry(corpus_id="help", document_id="d1"))
    write_entry(tmp_path, _entry(corpus_id="docs", document_id="d2"))

    merged = read_entries(tmp_path, namespace="acme")
    assert sorted(e.document_id for e in merged) == ["d1", "d2"]


def test_read_entries_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert read_entries(tmp_path, namespace="missing") == []


def test_read_entries_reports_malformed_json_line_without_payload(
    tmp_path: Path,
) -> None:
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    path.parent.mkdir(parents=True)
    path.write_text('{"document_id": "private-doc"\n', encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    error = exc_info.value
    assert error.path == path
    assert error.line_number == 1
    assert error.reason == "invalid_json"
    assert str(path) in str(error)
    assert "line 1" in str(error)
    assert "private-doc" not in str(error)


def test_read_entries_reports_invalid_entry_shape_without_payload(
    tmp_path: Path,
) -> None:
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    path.parent.mkdir(parents=True)
    path.write_text('{"document_id": "private-doc"}\n', encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    error = exc_info.value
    assert error.path == path
    assert error.line_number == 1
    assert error.reason == "invalid_entry"
    assert "private-doc" not in str(error)


def test_read_entries_reports_non_object_entry(tmp_path: Path) -> None:
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    path.parent.mkdir(parents=True)
    path.write_text('"private-doc"\n', encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    error = exc_info.value
    assert error.path == path
    assert error.line_number == 1
    assert error.reason == "entry_must_be_object"
    assert "private-doc" not in str(error)


def test_read_entries_suppresses_traceback_payload_for_bad_values(
    tmp_path: Path,
) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(chunk_count="secret-token"),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.reason == "invalid_entry"
    assert "secret-token" not in rendered
    assert "invalid literal" not in rendered


@pytest.mark.parametrize("field", ["parser", "metadata"])
def test_read_entries_rejects_missing_canonical_fields(
    tmp_path: Path, field: str
) -> None:
    payload = _manifest_payload()
    del payload[field]
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


def test_read_entries_rejects_extra_manifest_fields(tmp_path: Path) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(extra_field="extra"),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", []),
        ("namespace", "nested/slash"),
        ("corpus_id", " help "),
        ("filename", ""),
        ("mime_type", None),
    ],
)
def test_read_entries_rejects_malformed_required_string_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(**{field: value}),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_key", []),
        ("content_sha256", ""),
        ("parser", False),
    ],
)
def test_read_entries_rejects_malformed_optional_string_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(**{field: value}),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


def test_compact_manifest_sanitizes_bad_rehydrated_identifiers(
    tmp_path: Path,
) -> None:
    path = _write_manifest_payload(
        tmp_path,
        _manifest_payload(document_id=[]),
    )
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("chunk_count", [True, -1, "7"])
def test_read_entries_rejects_non_canonical_chunk_count(
    tmp_path: Path, chunk_count: object
) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(chunk_count=chunk_count),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


def test_read_entries_rejects_missing_chunk_count(tmp_path: Path) -> None:
    payload = _manifest_payload()
    del payload["chunk_count"]
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


@pytest.mark.parametrize("needs_ocr", ["false", 0, None])
def test_read_entries_rejects_non_boolean_needs_ocr(
    tmp_path: Path, needs_ocr: object
) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(needs_ocr=needs_ocr),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


def test_read_entries_rejects_missing_needs_ocr(tmp_path: Path) -> None:
    payload = _manifest_payload()
    del payload["needs_ocr"]
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


def test_read_entries_rejects_missing_metadata(tmp_path: Path) -> None:
    payload = _manifest_payload()
    del payload["metadata"]
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


@pytest.mark.parametrize("metadata", [None, [], "metadata"])
def test_read_entries_rejects_non_object_metadata(
    tmp_path: Path, metadata: object
) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(metadata=metadata),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "invalid_entry"


def test_write_entry_drops_unsafe_manifest_metadata(tmp_path: Path) -> None:
    write_entry(
        tmp_path,
        _entry(
            metadata={
                "team": "search",
                "error": "raw parser failure sk-test-secret",
                "parse_error": "raw parse failure sk-test-secret",
                "quality": {
                    "verdict": "poor",
                    "details": "public parse quality explanation",
                },
            },
        ),
    )

    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    raw = path.read_text(encoding="utf-8")
    assert "raw parser failure" not in raw
    assert "raw parse failure" not in raw
    assert "sk-test-secret" not in raw

    [entry] = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert entry.metadata == {
        "quality": {
            "details": "public parse quality explanation",
            "verdict": "poor",
        },
        "team": "search",
    }


def test_write_entry_caps_manifest_page_index_metadata(tmp_path: Path) -> None:
    write_entry(
        tmp_path,
        _entry(
            metadata={
                "image_only_page_indices": list(range(450)),
                "ocr_page_indices": list(range(450)),
            },
        ),
    )

    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert len(metadata["image_only_page_indices"]) == 400
    assert len(metadata["ocr_page_indices"]) == 400
    assert metadata["image_only_page_indices"][-1] == 399
    assert metadata["ocr_page_indices"][-1] == 399


def test_write_entry_sanitizes_manifest_page_index_metadata_values(
    tmp_path: Path,
) -> None:
    write_entry(
        tmp_path,
        _entry(
            metadata={
                "ocr_page_indices": [True, 2, False, 0, 2, "3", -1],
            },
        ),
    )

    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["ocr_page_indices"] == [2, 0]


def test_read_entries_rejects_unsafe_manifest_metadata(tmp_path: Path) -> None:
    _write_manifest_payload(
        tmp_path,
        _manifest_payload(metadata={"error": "raw parser failure sk-test-secret"}),
    )

    with pytest.raises(ManifestReadError) as exc_info:
        read_entries(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "unsafe_metadata"
    assert "sk-test-secret" not in str(exc_info.value)


def test_compact_manifest_rewrites_latest_entry_per_document_id(
    tmp_path: Path,
) -> None:
    write_entry(tmp_path, _entry(document_id="doc-1", chunk_count=2))
    write_entry(tmp_path, _entry(document_id="doc-2", chunk_count=3))
    write_entry(
        tmp_path,
        _entry(document_id="doc-1", chunk_count=5, metadata={"revision": "latest"}),
    )

    result = compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    assert result.before_entry_count == 3
    assert result.after_entry_count == 2
    assert result.removed_entry_count == 1
    assert result.changed is True
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    entries = {
        entry.document_id: entry
        for entry in read_entries(tmp_path, namespace="acme", corpus_id="help")
    }
    assert entries["doc-1"].chunk_count == 5
    assert entries["doc-1"].metadata == {"revision": "latest"}
    assert entries["doc-2"].chunk_count == 3

    unchanged = compact_manifest(tmp_path, namespace="acme", corpus_id="help")
    assert unchanged.before_entry_count == 2
    assert unchanged.after_entry_count == 2
    assert unchanged.removed_entry_count == 0
    assert unchanged.changed is False


def test_compact_manifest_rewrites_latest_entry_per_document_key(
    tmp_path: Path,
) -> None:
    write_entry(
        tmp_path,
        _entry(document_id="doc-old", document_key="same.md", chunk_count=2),
    )
    write_entry(
        tmp_path,
        _entry(document_id="doc-new", document_key="same.md", chunk_count=5),
    )

    result = compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    assert result.before_entry_count == 2
    assert result.after_entry_count == 1
    assert result.changed is True
    entries = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert [entry.document_id for entry in entries] == ["doc-new"]
    assert entries[0].document_key == "same.md"


def test_compact_manifest_rewrites_healed_null_document_key(
    tmp_path: Path,
) -> None:
    write_entry(
        tmp_path,
        CorpusManifestEntry(
            document_id="doc-1",
            namespace="acme",
            corpus_id="help",
            document_key=None,
            content_sha256="old-sha",
            filename="doc.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )
    write_entry(
        tmp_path,
        _entry(
            document_id="doc-1",
            document_key="doc.md",
            content_sha256="new-sha",
            chunk_count=2,
        ),
    )

    result = compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    assert result.before_entry_count == 2
    assert result.after_entry_count == 1
    entries = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert [(entry.document_id, entry.document_key, entry.content_sha256) for entry in entries] == [
        ("doc-1", "doc.md", "new-sha")
    ]


def test_compact_manifest_skips_byte_identical_unchanged_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_entry(tmp_path, _entry(document_id="doc-1"))
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    original = path.read_text(encoding="utf-8")

    def fail_write(path: Path, content: str) -> None:
        raise AssertionError("unchanged manifest should not be rewritten")

    monkeypatch.setattr("rag_core.manifest_jsonl.atomic_write_text", fail_write)

    result = compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    assert result.before_entry_count == 1
    assert result.after_entry_count == 1
    assert result.changed is False
    assert path.read_text(encoding="utf-8") == original


def test_compact_manifest_missing_path_returns_zero_result(tmp_path: Path) -> None:
    result = compact_manifest(tmp_path, namespace="missing", corpus_id="help")

    assert result == ManifestCompactionResult(
        before_entry_count=0,
        after_entry_count=0,
    )
    assert result.removed_entry_count == 0
    assert result.changed is False
    assert not (tmp_path / "missing").exists()


def test_compact_manifest_reports_bad_line_number_without_rewriting(
    tmp_path: Path,
) -> None:
    write_entry(tmp_path, _entry(document_id="doc-1"))
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    original = path.read_text(encoding="utf-8") + '{"document_id": "private-doc"\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    error = exc_info.value
    assert error.path == path
    assert error.line_number == 2
    assert error.reason == "invalid_json"
    assert "private-doc" not in str(error)
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("namespace", "corpus_id"),
    [
        ("../escape", "help"),
        ("acme", "../escape"),
        ("", "help"),
        ("acme", ""),
        ("acme/slash", "help"),
        ("acme", "nested/slash"),
    ],
)
def test_compact_manifest_rejects_non_segment_scope_values(
    tmp_path: Path, namespace: str, corpus_id: str
) -> None:
    with pytest.raises(ValueError, match="single non-empty path segment"):
        compact_manifest(tmp_path, namespace=namespace, corpus_id=corpus_id)


def test_write_entry_atomic_write_no_partial_file(tmp_path: Path) -> None:
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    write_entry(tmp_path, _entry())
    # Locked appends use a sidecar lock file; no payload temp files should remain.
    leftovers = [
        p for p in path.parent.iterdir() if p != path and p.name != f"{path.name}.lock"
    ]
    assert leftovers == []


def test_round_trip_preserves_metadata(tmp_path: Path) -> None:
    write_entry(tmp_path, _entry(metadata={"team": "search", "env": "test"}))
    [entry] = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert entry.metadata == {"team": "search", "env": "test"}
    assert entry.parser == "local:converter"
    assert entry.needs_ocr is False


def test_write_entry_if_stale_deduplicates_concurrent_unchanged_appends(
    tmp_path: Path,
) -> None:
    entry = _entry(document_id="doc-1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(write_entry_if_stale, tmp_path, entry) for _ in range(40)
        ]
        for future in futures:
            future.result()

    entries = read_entries(tmp_path, namespace="acme", corpus_id="help")
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    assert [entry.document_id for entry in entries] == ["doc-1"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_write_entry_if_stale_updates_manifest_visible_changes(
    tmp_path: Path,
) -> None:
    write_entry(tmp_path, _entry(document_id="doc-1", metadata={"quality": "good"}))

    changed = write_entry_if_stale(
        tmp_path,
        _entry(document_id="doc-1", metadata={"quality": "poor"}),
    )

    assert changed is True
    [entry] = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert entry.metadata == {"quality": "poor"}
    path = manifest_path(tmp_path, namespace="acme", corpus_id="help")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_refreshed_manifest_entry_heals_missing_existing_content_hash(
    tmp_path: Path,
) -> None:
    entry = refreshed_manifest_entry(
        previous=None,
        existing=StoredDocumentRecord(
            document_id="doc-1",
            namespace="acme",
            corpus_id="help",
            document_key="doc-1.md",
            content_sha256=None,
            chunk_count=2,
        ),
        document_id="doc-1",
        namespace="acme",
        corpus_id="help",
        document_key="doc-1.md",
        content_sha256="fresh-sha",
        filename="doc-1.md",
        mime_type="text/markdown",
        metadata=None,
    )

    assert write_entry_if_stale(tmp_path, entry) is True
    [stored] = read_entries(tmp_path, namespace="acme", corpus_id="help")
    assert stored.content_sha256 == "fresh-sha"
    assert stored.chunk_count == 2


def test_write_entry_if_stale_rejects_unsafe_manifest_metadata_without_rewrite(
    tmp_path: Path,
) -> None:
    path = _write_manifest_payload(
        tmp_path,
        _manifest_payload(metadata={"error": "raw parser failure sk-test-secret"}),
    )
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        write_entry_if_stale(tmp_path, _entry(content_sha256="sha"))

    assert exc_info.value.reason == "unsafe_metadata"
    assert path.read_text(encoding="utf-8") == original


def test_compact_manifest_rejects_unsafe_manifest_metadata_without_rewrite(
    tmp_path: Path,
) -> None:
    path = _write_manifest_payload(
        tmp_path,
        _manifest_payload(metadata={"details": "public", "error": "sk-test-secret"}),
    )
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ManifestReadError) as exc_info:
        compact_manifest(tmp_path, namespace="acme", corpus_id="help")

    assert exc_info.value.reason == "unsafe_metadata"
    assert path.read_text(encoding="utf-8") == original


def test_summarize_entries_counts_documents_chunks_parsers_and_ocr() -> None:
    summary = summarize_entries(
        [
            _entry(
                document_id="doc-1", chunk_count=2, parser="local:pdf", needs_ocr=True
            ),
            _entry(document_id="doc-2", chunk_count=3, parser="local:pdf"),
            _entry(document_id="doc-3", chunk_count=5, parser=None),
        ]
    )

    assert summary.document_count == 3
    assert summary.chunk_count == 10
    assert summary.needs_ocr_count == 1
    assert summary.parser_counts == {"local:pdf": 2, "unknown": 1}


def test_reconcile_entries_explains_unchanged_changed_missing_and_orphaned() -> None:
    entries = [
        _entry(document_id="doc-1", metadata={}, chunk_count=1),
        _entry(document_id="doc-2", chunk_count=1),
        _entry(document_id="doc-3", chunk_count=1),
    ]
    sources = [
        ManifestSource(document_key="doc-1.md", content_sha256="sha256-stub"),
        ManifestSource(document_key="doc-2.md", content_sha256="new-sha"),
        ManifestSource(document_key="doc-4.md", content_sha256="sha4"),
    ]

    reconciliation = reconcile_entries(entries, sources)

    assert [(item.document_key, item.reason) for item in reconciliation.unchanged] == [
        ("doc-1.md", "content_sha256_match")
    ]
    assert [(item.document_key, item.reason) for item in reconciliation.changed] == [
        ("doc-2.md", "content_sha256_changed")
    ]
    assert [(item.document_key, item.reason) for item in reconciliation.missing] == [
        ("doc-4.md", "source_not_in_manifest")
    ]
    assert [(item.document_key, item.reason) for item in reconciliation.orphaned] == [
        ("doc-3.md", "manifest_entry_without_source")
    ]
    assert [item.document_key for item in reconciliation.needs_reindex] == [
        "doc-2.md",
        "doc-4.md",
    ]


def test_manifest_reconciliation_payload_reports_summary_and_items() -> None:
    reconciliation = reconcile_entries(
        [_entry(document_id="doc-1", metadata={}, chunk_count=1)],
        [
            ManifestSource(document_key="doc-1.md", content_sha256="sha256-stub"),
            ManifestSource(document_key="doc-2.md", content_sha256="sha2"),
        ],
    )

    payload = manifest_reconciliation_payload(reconciliation)

    assert payload["summary"] == {
        "changed_count": 0,
        "duplicate_count": 0,
        "missing_count": 1,
        "needs_reindex_count": 1,
        "orphaned_count": 0,
        "unchanged_count": 1,
    }
    items = cast(list[dict[str, object]], payload["items"])
    assert items == [
        {
            "item_index": 0,
            "status": "unchanged",
            "reason": "content_sha256_match",
            "has_document_id": True,
            "has_manifest_content_sha256": True,
            "has_source_content_sha256": True,
        },
        {
            "item_index": 1,
            "status": "missing",
            "reason": "source_not_in_manifest",
            "has_document_id": False,
            "has_manifest_content_sha256": False,
            "has_source_content_sha256": True,
        },
    ]
    private_payload = manifest_reconciliation_payload(
        reconciliation,
        include_private=True,
    )
    private_items = cast(list[dict[str, object]], private_payload["items"])
    assert private_items[0]["document_key"] == "doc-1.md"
    assert private_items[1]["document_key"] == "doc-2.md"


def test_reconcile_entries_marks_duplicate_manifest_document_keys() -> None:
    entries = [
        _entry(document_id="doc-a", document_key="same.md", chunk_count=1),
        _entry(document_id="doc-b", document_key="same.md", chunk_count=1),
    ]

    reconciliation = reconcile_entries(
        entries,
        [ManifestSource(document_key="same.md", content_sha256="source-sha")],
    )

    assert [item.document_id for item in reconciliation.duplicate] == ["doc-a", "doc-b"]
    assert all(
        item.reason == "duplicate_manifest_document_key"
        for item in reconciliation.duplicate
    )
    assert all(
        item.source_content_sha256 == "source-sha" for item in reconciliation.duplicate
    )
    assert reconciliation.missing == ()


def test_reconcile_entries_rejects_duplicate_sources() -> None:
    with pytest.raises(ValueError, match="source document_key must be unique"):
        reconcile_entries(
            [],
            [
                ManifestSource(document_key="same.md"),
                ManifestSource(document_key="same.md"),
            ],
        )
