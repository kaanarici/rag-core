import asyncio
import os
from pathlib import Path
from typing import cast

import pytest

from rag_core.local_corpus import ManifestPreviewRequest, preview_manifest
from rag_core.sources import document_key as local_document_key


def test_preview_manifest_default_document_key_uses_local_relative_hash(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "guide.txt"
    file_path.write_text("billing docs stay easy to find", encoding="utf-8")

    result = asyncio.run(
        preview_manifest(
            ManifestPreviewRequest(
                path=file_path,
                namespace="acme",
                corpus_id="help",
            )
        )
    )

    payload = result.to_payload()
    document = cast(dict[str, object], payload["document"])
    manifest_entry = cast(dict[str, object], payload["manifest_entry"])
    expected_key = local_document_key(docs, file_path)

    assert document["document_key"] == expected_key
    assert manifest_entry["document_key"] == expected_key
    assert str(tmp_path) not in str(document["document_key"])
    assert expected_key.startswith("local:guide.txt#source:")


def test_preview_manifest_blank_document_key_uses_local_relative_hash(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing docs stay easy to find", encoding="utf-8")

    result = asyncio.run(
        preview_manifest(
            ManifestPreviewRequest(
                path=file_path,
                namespace="acme",
                corpus_id="help",
                document_key="  ",
            )
        )
    )

    payload = result.to_payload()
    document = cast(dict[str, object], payload["document"])

    assert document["document_key"] == local_document_key(tmp_path, file_path)
    assert str(tmp_path) not in str(document["document_key"])


def test_preview_manifest_rejects_symlink_aliases(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.txt"
    target.write_text("target", encoding="utf-8")
    alias = docs / "alias.txt"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="does not allow symlink paths"):
        asyncio.run(
            preview_manifest(
                ManifestPreviewRequest(
                    path=alias,
                    namespace="acme",
                    corpus_id="help",
                )
            )
        )


def test_preview_manifest_rejects_relative_path_from_symlinked_logical_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "target.txt"
    target.write_text("target", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(docs, target_is_directory=True)
    monkeypatch.chdir(alias)
    monkeypatch.setenv("PWD", str(alias))

    with pytest.raises(ValueError, match="does not allow symlink paths"):
        asyncio.run(
            preview_manifest(
                ManifestPreviewRequest(
                    path=Path("target.txt"),
                    namespace="acme",
                    corpus_id="help",
                )
            )
        )
