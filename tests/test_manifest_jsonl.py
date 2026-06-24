from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import rag_core.manifest.jsonl as manifest_jsonl
from rag_core.core_models import CollectionManifestEntry
from rag_core.manifest.entries import ManifestReadError
from rag_core.manifest.jsonl import (
    append_manifest_jsonl_entry,
    append_manifest_jsonl_entry_if_stale,
    atomic_write_text,
    read_manifest_jsonl_entries,
    update_manifest_jsonl_entries,
)


def _entry(document_id: str) -> CollectionManifestEntry:
    return CollectionManifestEntry(
        document_id=document_id,
        namespace="acme",
        collection="help",
        document_key=f"{document_id}.md",
        content_sha256=f"sha-{document_id}",
        filename=f"{document_id}.md",
        mime_type="text/markdown",
        chunk_count=1,
        parser="local:markdown",
        needs_ocr=False,
        metadata={},
    )


def test_append_manifest_jsonl_entry_preserves_entries_with_concurrent_writers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    writer_count = 8
    entries_per_writer = 30

    def write_batch(worker_index: int) -> None:
        for item_index in range(entries_per_writer):
            document_id = f"w{worker_index}-d{item_index}"
            append_manifest_jsonl_entry(path, _entry(document_id))

    with ThreadPoolExecutor(max_workers=writer_count) as pool:
        futures = [pool.submit(write_batch, index) for index in range(writer_count)]
        for future in futures:
            future.result()

    entries = read_manifest_jsonl_entries(path)
    expected_ids = {
        f"w{worker_index}-d{item_index}"
        for worker_index in range(writer_count)
        for item_index in range(entries_per_writer)
    }

    assert len(entries) == len(expected_ids)
    assert {entry.document_id for entry in entries} == expected_ids


def test_manifest_jsonl_files_are_private_to_current_user(tmp_path: Path) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    append_manifest_jsonl_entry(path, _entry("doc-1"))

    if os.name != "nt":
        assert (tmp_path / "manifest").stat().st_mode & 0o777 == 0o700
        assert (tmp_path / "manifest" / "acme").stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.with_name("help.jsonl.lock").stat().st_mode & 0o777 == 0o600


def test_manifest_jsonl_rejects_symlink_data_file(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    target = tmp_path / "target.jsonl"
    path.parent.mkdir(parents=True)
    path.symlink_to(target)

    with pytest.raises(ValueError, match="path must not be a symlink"):
        append_manifest_jsonl_entry(path, _entry("doc-1"))

    assert not target.exists()


def test_manifest_jsonl_rejects_symlink_lock_file(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    lock_path = path.with_name("help.jsonl.lock")
    target = tmp_path / "target.lock"
    path.parent.mkdir(parents=True)
    lock_path.symlink_to(target)

    with pytest.raises(ValueError, match="path must not be a symlink"):
        append_manifest_jsonl_entry(path, _entry("doc-1"))

    assert not target.exists()


def test_manifest_jsonl_rejects_hardlinked_data_file(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "link"):
        pytest.skip("hard links unavailable")
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    target = tmp_path / "target.jsonl"
    path.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")
    os.link(target, path)

    with pytest.raises(ValueError, match="hard-linked"):
        append_manifest_jsonl_entry(path, _entry("doc-1"))

    assert target.read_text(encoding="utf-8") == ""


def test_manifest_jsonl_rejects_hardlinked_lock_file(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "link"):
        pytest.skip("hard links unavailable")
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    lock_path = path.with_name("help.jsonl.lock")
    target = tmp_path / "target.lock"
    path.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")
    os.link(target, lock_path)

    with pytest.raises(ValueError, match="hard-linked"):
        append_manifest_jsonl_entry(path, _entry("doc-1"))

    assert target.read_text(encoding="utf-8") == ""


def test_manifest_jsonl_rejects_symlink_manifest_parent_ancestor(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    path = alias_root / "manifest" / "acme" / "help.jsonl"

    with pytest.raises(ValueError, match="path parent must not be a symlink"):
        append_manifest_jsonl_entry(path, _entry("doc-1"))

    assert not (real_root / "manifest").exists()


def test_manifest_jsonl_read_rejects_symlink_manifest_parent_ancestor(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    path = alias_root / "manifest" / "acme" / "help.jsonl"

    with pytest.raises(ValueError, match="path parent must not be a symlink"):
        read_manifest_jsonl_entries(path)

    assert not (real_root / "manifest").exists()


def test_append_manifest_jsonl_entry_if_stale_deduplicates_concurrent_writers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    entry = _entry("doc-1")

    def append_if_needed() -> None:
        append_manifest_jsonl_entry_if_stale(
            path,
            entry,
            lambda current, candidate: (
                current is None
                or current.content_sha256 != candidate.content_sha256
                or current.chunk_count != candidate.chunk_count
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(append_if_needed) for _ in range(40)]
        for future in futures:
            future.result()

    entries = read_manifest_jsonl_entries(path)
    assert len(entries) == 1
    assert entries[0].document_id == "doc-1"


def test_append_manifest_jsonl_entry_if_stale_reuses_latest_entry_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    existing_count = 20
    appended_count = 5
    for index in range(existing_count):
        append_manifest_jsonl_entry(path, _entry(f"existing-{index}"))
    parse_count = 0
    original_parser = manifest_jsonl.entry_from_json_line

    def count_parse(path_arg: Path, line_number: int, line: str) -> CollectionManifestEntry:
        nonlocal parse_count
        parse_count += 1
        return original_parser(path_arg, line_number, line)

    monkeypatch.setattr(manifest_jsonl, "entry_from_json_line", count_parse)

    for index in range(appended_count):
        append_manifest_jsonl_entry_if_stale(
            path,
            _entry(f"appended-{index}"),
            lambda current, _candidate: current is None,
        )

    assert parse_count == existing_count + appended_count
    assert len(path.read_text(encoding="utf-8").splitlines()) == (
        existing_count + appended_count
    )


def test_update_manifest_jsonl_entries_skips_byte_identical_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    append_manifest_jsonl_entry(path, _entry("doc-1"))
    original = path.read_text(encoding="utf-8")
    writes: list[str] = []

    def fail_write(path: Path, content: str) -> None:
        writes.append(content)
        raise AssertionError("unchanged manifest should not be rewritten")

    monkeypatch.setattr("rag_core.manifest.jsonl.atomic_write_text", fail_write)

    before, after, changed = update_manifest_jsonl_entries(
        path, lambda entries: entries
    )

    assert (before, after) == (1, 1)
    assert changed is False
    assert writes == []
    assert path.read_text(encoding="utf-8") == original


def test_append_manifest_jsonl_entry_failure_leaves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    append_manifest_jsonl_entry(path, _entry("doc-1"))
    original = path.read_text(encoding="utf-8")

    def fail_append(path: Path, content: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("rag_core.manifest.jsonl._append_text_durable", fail_append)

    with pytest.raises(OSError, match="disk full"):
        append_manifest_jsonl_entry(path, _entry("doc-2"))

    assert path.read_text(encoding="utf-8") == original
    assert [entry.document_id for entry in read_manifest_jsonl_entries(path)] == [
        "doc-1"
    ]


def test_read_manifest_jsonl_entries_rejects_trailing_partial_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest" / "acme" / "help.jsonl"
    append_manifest_jsonl_entry(path, _entry("doc-1"))
    original = path.read_bytes()

    def partial_append(path: Path, content: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, content.encode("utf-8")[:12])
            raise OSError("disk full")
        finally:
            os.close(fd)

    monkeypatch.setattr("rag_core.manifest.jsonl._append_text_durable", partial_append)

    with pytest.raises(OSError, match="disk full"):
        append_manifest_jsonl_entry(path, _entry("doc-2"))

    assert path.read_bytes().startswith(original)
    with pytest.raises(ManifestReadError):
        read_manifest_jsonl_entries(path)


def test_atomic_write_text_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def record_fsync(fd: int) -> None:
        calls.append(fd)

    monkeypatch.setattr("rag_core.manifest.jsonl.os.fsync", record_fsync)

    path = tmp_path / "manifest.jsonl"
    atomic_write_text(path, "payload\n")

    assert path.read_text(encoding="utf-8") == "payload\n"
    expected_call_count = 1 if os.name == "nt" else 2
    assert len(calls) == expected_call_count


def test_manifest_jsonl_has_no_platform_specific_top_level_lock_imports() -> None:
    module = ast.parse(Path("src/rag_core/manifest/jsonl.py").read_text())
    top_level_imports = [
        alias.name
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ]

    assert "fcntl" not in top_level_imports
    assert "msvcrt" not in top_level_imports
