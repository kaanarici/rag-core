import os
from pathlib import Path
from typing import Any

import pytest

from rag_core.core_lifecycle import compute_content_sha256
from rag_core.local_ingest_models import LocalIngestRequest
from rag_core.local_ingest_planning import build_local_ingest_plan
from rag_core.local_search_models import LocalSearchRequest
from rag_core.local_search_planning import build_local_search_plan
from rag_core.sources import (
    LocalFileSourceReader,
    document_key as local_document_key,
    local_file_source_item,
)


def test_local_file_source_reader_expands_supported_files_with_stable_keys(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    first = docs / "a.md"
    first.write_text("a", encoding="utf-8")
    second = nested / "b.txt"
    second.write_text("b", encoding="utf-8")
    (docs / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    plan = LocalFileSourceReader().read(docs)

    assert plan.source == str(docs)
    assert plan.root == docs
    assert [
        (item.path, item.document_key, item.content_sha256, item.source_error)
        for item in plan.items
    ] == [
        (first, local_document_key(docs, first), compute_content_sha256(b"a"), ""),
        (second, local_document_key(docs, second), compute_content_sha256(b"b"), ""),
    ]
    assert plan.items[0].to_payload() == {
        "path": "<local-file>",
        "filename": "a.md",
        "content_sha256_available": True,
        "source_error": "",
    }
    assert plan.items[0].to_payload(include_private=True) == {
        "path": str(first),
        "document_key": local_document_key(docs, first),
        "content_sha256": compute_content_sha256(b"a"),
        "source_error": "",
    }


def test_local_file_source_reader_uses_literal_file_and_glob_roots(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    file_path = docs / "single.md"
    file_path.write_text("single", encoding="utf-8")
    nested_file = nested / "item.md"
    nested_file.write_text("nested", encoding="utf-8")

    file_plan = LocalFileSourceReader().read(file_path)
    glob_plan = LocalFileSourceReader().read(docs / "**" / "*.md")

    assert [item.document_key for item in file_plan.items] == [
        local_document_key(docs, file_path)
    ]
    assert [item.document_key for item in glob_plan.items] == [
        local_document_key(docs, nested_file),
        local_document_key(docs, file_path),
    ]


def test_local_file_source_reader_dedupes_overlapping_glob_matches(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "single.md"
    file_path.write_text("single", encoding="utf-8")

    plan = LocalFileSourceReader().read(docs / "**" / "**" / "*.md")

    assert [item.path for item in plan.items] == [file_path]


def test_local_file_source_reader_treats_existing_metachar_path_as_literal(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    literal_file = docs / "notes[final].md"
    literal_file.write_text("final", encoding="utf-8")

    plan = LocalFileSourceReader().read(literal_file)

    assert plan.root == docs
    assert [item.path for item in plan.items] == [literal_file]
    assert [item.document_key for item in plan.items] == [
        local_document_key(docs, literal_file)
    ]


def test_local_file_source_reader_single_file_key_is_stable_across_path_spellings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "single.md"
    file_path.write_text("single", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    relative_plan = LocalFileSourceReader().read(Path("docs/single.md"))
    dotted_plan = LocalFileSourceReader().read(Path("./docs/single.md"))
    absolute_plan = LocalFileSourceReader().read(file_path.resolve())

    expected_key = local_document_key(docs, file_path)
    assert [item.document_key for item in relative_plan.items] == [expected_key]
    assert [item.document_key for item in dotted_plan.items] == [expected_key]
    assert [item.document_key for item in absolute_plan.items] == [expected_key]


def test_local_file_source_reader_rejects_symlinked_roots(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable on this platform")
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "single.md"
    file_path.write_text("single", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(docs, target_is_directory=True)

    real_plan = LocalFileSourceReader().read(docs)
    alias_plan = LocalFileSourceReader().read(alias)

    assert [item.document_key for item in real_plan.items]
    assert alias_plan.items == ()


def test_local_search_plan_rejects_symlinked_roots(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable on this platform")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "single.md").write_text("single", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(docs, target_is_directory=True)

    with pytest.raises(ValueError, match="does not allow symlink paths"):
        build_local_search_plan(LocalSearchRequest(path=alias, query="single"))


def test_local_file_source_reader_single_file_key_does_not_collide_on_basename(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_file = left / "same.md"
    right_file = right / "same.md"
    left_file.write_text("left", encoding="utf-8")
    right_file.write_text("right", encoding="utf-8")

    left_plan = LocalFileSourceReader().read(left_file)
    right_plan = LocalFileSourceReader().read(right_file)

    assert [item.document_key for item in left_plan.items] == [
        local_document_key(left, left_file)
    ]
    assert [item.document_key for item in right_plan.items] == [
        local_document_key(right, right_file)
    ]


def test_local_file_source_reader_rejects_literal_symlink_file_path(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    plan = LocalFileSourceReader().read(alias)

    assert [item.path for item in plan.items] == []


def test_local_file_source_reader_skips_multi_link_regular_files(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    os.link(target, alias)

    plan = LocalFileSourceReader().read(docs)

    assert [item.path for item in plan.items] == []


def test_local_file_source_item_rejects_multi_link_regular_files(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    os.link(target, alias)

    with pytest.raises(ValueError, match="multi-link file paths"):
        local_file_source_item(target, root=docs)


def test_local_file_source_reader_rejects_direct_multi_link_file_path(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    os.link(target, alias)

    plan = LocalFileSourceReader().read(target)

    assert plan.items == ()


def test_local_file_source_item_preserves_read_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_core import sources as sources_module

    file_path = tmp_path / "blocked.txt"
    file_path.write_text("blocked", encoding="utf-8")

    def fail_hash(path: Path) -> str:
        raise PermissionError("blocked")

    monkeypatch.setattr(sources_module, "file_content_sha256", fail_hash)

    item = local_file_source_item(file_path, root=file_path)

    assert item.content_sha256 is None
    assert item.source_error == "blocked"


def test_local_file_source_item_rejects_symlink_aliases(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="does not allow symlink paths"):
        local_file_source_item(alias, root=docs)


def test_relative_local_sources_reject_symlinked_logical_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(docs, target_is_directory=True)
    monkeypatch.chdir(alias)
    monkeypatch.setenv("PWD", str(alias))

    plan = LocalFileSourceReader().read(Path("target.md"))

    assert plan.items == ()
    with pytest.raises(ValueError, match="does not allow symlink paths"):
        build_local_ingest_plan(
            LocalIngestRequest(path=Path("target.md"), namespace="acme", corpus_id="help")
        )


def test_local_file_source_reader_skips_directory_symlinks_outside_requested_root(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "external.md"
    target.write_text("external", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    plan = LocalFileSourceReader().read(docs)

    assert [item.path for item in plan.items] == []


def test_local_file_source_reader_skips_directory_symlink_aliases_inside_requested_root(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    plan = LocalFileSourceReader().read(docs)

    assert [item.path for item in plan.items] == [target]
    assert [item.document_key for item in plan.items] == [
        local_document_key(docs, target)
    ]


def test_local_file_source_reader_skips_glob_symlink_aliases_inside_requested_root(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    plan = LocalFileSourceReader().read(docs / "*.md")

    assert [item.path for item in plan.items] == [target]


def test_local_search_plan_skips_directory_symlinks_outside_requested_root(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    inside_file = docs / "inside.md"
    inside_file.write_text("inside", encoding="utf-8")
    target = outside / "external.md"
    target.write_text("external", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    plan = build_local_search_plan(
        LocalSearchRequest(path=docs, query="hello", max_files=20)
    )

    assert [document.path for document in plan.documents] == [inside_file]
    assert plan.skipped_unsupported_count == 1


def test_local_search_plan_skips_directory_symlink_aliases_inside_requested_root(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    plan = build_local_search_plan(
        LocalSearchRequest(path=docs, query="hello", max_files=20)
    )

    assert [document.path for document in plan.documents] == [target]
    assert [document.document_key for document in plan.documents] == [
        local_document_key(docs, target)
    ]
    assert plan.skipped_unsupported_count == 1


def test_local_search_plan_dedupes_hardlinked_aliases(tmp_path: Path) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "a.md"
    target.write_text("target", encoding="utf-8")
    alias = docs / "b.md"
    os.link(target, alias)

    plan = build_local_search_plan(
        LocalSearchRequest(path=docs, query="target", max_files=20)
    )

    assert [document.path for document in plan.documents] == [target]
    assert plan.skipped_unsupported_count == 1


def test_local_search_plan_ignores_unreadable_empty_package_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    marker = docs / "__init__.py"
    marker.write_text("", encoding="utf-8")
    note = docs / "note.md"
    note.write_text("note", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_marker_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == marker:
            raise PermissionError("blocked")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_marker_read)

    plan = build_local_search_plan(
        LocalSearchRequest(path=docs, query="note", max_files=20)
    )

    assert [document.path for document in plan.documents] == [note]
    assert plan.skipped_empty_count == 0


def test_local_search_plan_keeps_unreadable_substantive_package_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    marker = docs / "__init__.py"
    marker.write_text("VALUE = 1\n", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_marker_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == marker:
            raise PermissionError("blocked")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_marker_read)

    plan = build_local_search_plan(
        LocalSearchRequest(path=docs, query="value", max_files=20)
    )

    assert [document.path for document in plan.documents] == [marker]
