from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import pytest

from rag_core.archive_sources import (
    ArchiveLimits,
    ZipArchiveSourceReader,
    archive_document_key,
    is_supported_archive_member_path,
    read_zip_member_bytes,
    safe_archive_member_path,
)


def test_zip_archive_source_reader_plans_supported_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "docs.zip"
    _write_zip(
        archive_path,
        {
            "docs/guide.md": b"# Guide\n\nHello",
            "docs/reference.py": b"print('hi')\n",
            "assets/logo.png": b"\x89PNG\r\n\x1a\n",
            "__pycache__/skip.pyc": b"generated",
        },
    )

    plan = ZipArchiveSourceReader().read(archive_path)

    assert plan.item_count == 2
    assert [item.member_path for item in plan.items] == [
        "docs/guide.md",
        "docs/reference.py",
    ]
    guide = plan.items[0]
    assert guide.path == f"{archive_path}!/docs/guide.md"
    assert guide.document_key == archive_document_key(archive_path, "docs/guide.md")
    assert guide.filename == "guide.md"
    assert guide.mime_type == "text/markdown"
    assert guide.content_sha256 == hashlib.sha256(b"# Guide\n\nHello").hexdigest()
    guide_payload = guide.to_payload()
    assert guide_payload["archive_name"] == "docs.zip"
    assert guide_payload["path"] == "docs.zip!/docs/guide.md"
    assert guide_payload["byte_count"] == len(b"# Guide\n\nHello")
    plan_payload = plan.to_payload()
    assert plan_payload["archive_name"] == "docs.zip"
    assert plan_payload["item_count"] == 2
    assert str(tmp_path) not in repr(plan_payload)


@pytest.mark.parametrize(
    "member_path",
    [
        "../secret.md",
        "/absolute.md",
        "docs/../secret.md",
        "docs\\secret.md",
        "docs//guide.md",
        " docs/guide.md",
        "docs/guide.md ",
    ],
)
def test_zip_archive_source_reader_rejects_unsafe_member_paths(
    tmp_path: Path,
    member_path: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    _write_zip(archive_path, {member_path: b"# Secret"})

    with pytest.raises(ValueError, match="unsafe"):
        ZipArchiveSourceReader().read(archive_path)


def test_zip_archive_source_reader_enforces_limits(tmp_path: Path) -> None:
    archive_path = tmp_path / "docs.zip"
    _write_zip(
        archive_path,
        {
            "a.md": b"a",
            "b.md": b"bb",
        },
    )

    with pytest.raises(ValueError, match="max_entries"):
        ZipArchiveSourceReader().read(archive_path, limits=ArchiveLimits(max_entries=1))
    with pytest.raises(ValueError, match="max_entry_bytes"):
        ZipArchiveSourceReader().read(
            archive_path,
            limits=ArchiveLimits(max_entry_bytes=1),
        )
    with pytest.raises(ValueError, match="max_total_bytes"):
        ZipArchiveSourceReader().read(
            archive_path,
            limits=ArchiveLimits(max_total_bytes=2),
        )


def test_zip_archive_source_reader_rejects_symlink_archive_path(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    archive_path = tmp_path / "docs.zip"
    alias_path = tmp_path / "alias.zip"
    _write_zip(archive_path, {"docs/guide.md": b"# Guide"})
    alias_path.symlink_to(archive_path)

    with pytest.raises(ValueError, match="does not allow symlink paths"):
        ZipArchiveSourceReader().read(alias_path)

    with pytest.raises(ValueError, match="does not allow symlink paths"):
        read_zip_member_bytes(alias_path, "docs/guide.md")


def test_archive_entrypoints_reject_hardlinked_archive_path(tmp_path: Path) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlink support unavailable")
    archive_path = tmp_path / "docs.zip"
    alias_path = tmp_path / "alias.zip"
    _write_zip(archive_path, {"docs/guide.md": b"# Guide"})
    try:
        os.link(archive_path, alias_path)
    except OSError as exc:
        pytest.skip(f"hardlink support unavailable: {exc}")

    with pytest.raises(ValueError, match="does not allow multi-link file paths"):
        ZipArchiveSourceReader().read(alias_path)

    with pytest.raises(ValueError, match="does not allow multi-link file paths"):
        read_zip_member_bytes(alias_path, "docs/guide.md")


def test_read_zip_member_bytes_uses_safe_limits(tmp_path: Path) -> None:
    archive_path = tmp_path / "docs.zip"
    _write_zip(archive_path, {"docs/guide.md": b"# Guide"})

    assert read_zip_member_bytes(archive_path, "docs/guide.md") == b"# Guide"
    with pytest.raises(ValueError, match="max_entry_bytes"):
        read_zip_member_bytes(
            archive_path,
            "docs/guide.md",
            limits=ArchiveLimits(max_entry_bytes=1),
        )
    with pytest.raises(ValueError, match="not found"):
        read_zip_member_bytes(archive_path, "docs/missing.md")


def test_read_zip_member_bytes_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", b"first")
        archive.writestr("docs/guide.md", b"second")

    with pytest.raises(ValueError, match="duplicate member path"):
        read_zip_member_bytes(archive_path, "docs/guide.md")


def test_archive_helpers_are_public_contracts(tmp_path: Path) -> None:
    archive_path = tmp_path / "docs.zip"
    document_key = archive_document_key(archive_path, "docs/guide.md")

    assert safe_archive_member_path("docs/guide.md") == "docs/guide.md"
    assert document_key.startswith("archive:docs/guide.md#source:")
    assert str(archive_path.resolve()) not in document_key
    assert is_supported_archive_member_path("docs/guide.md")
    assert not is_supported_archive_member_path("../secret.md")
    with pytest.raises(ValueError, match="unsafe"):
        read_zip_member_bytes(Path("missing.zip"), "../secret.md")


def test_archive_document_key_disambiguates_same_basename_archives(tmp_path: Path) -> None:
    first = tmp_path / "one" / "docs.zip"
    second = tmp_path / "two" / "docs.zip"
    first_key = archive_document_key(first, "docs/guide.md")
    second_key = archive_document_key(second, "docs/guide.md")

    assert first_key != second_key
    assert first_key.startswith("archive:docs/guide.md#source:")
    assert second_key.startswith("archive:docs/guide.md#source:")


def _write_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
