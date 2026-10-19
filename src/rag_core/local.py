"""Persistent local-folder configuration shared by the Python helper and CLI."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path

from rag_core.core_models import Config
from rag_core.ingest.sources.local import reject_local_symlink_path

DEFAULT_LOCAL_TENANT_ID = "local"


def local_collection(path: Path) -> str:
    return (path.stem if path.is_file() else path.name) or "local-collection"


def local_config(
    path: str | Path,
    *,
    persist_dir: str | Path | None = None,
    hybrid: bool = True,
) -> Config:
    source = Path(path).expanduser()
    reject_local_symlink_path(source)
    source = source.resolve(strict=True)
    base = (
        Path(persist_dir).expanduser()
        if persist_dir is not None
        else Path(
            os.environ.get("RAG_CORE_INDEX_DIR")
            or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
            / "rag-core"
        )
    ).resolve()
    if source.is_dir() and base.is_relative_to(source):
        raise ValueError("persist_dir must be outside the indexed folder")
    identity = hashlib.sha256(str(source).encode()).hexdigest()
    directory = base / identity / "v2" / ("hybrid" if hybrid else "dense")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = Config.local(directory / "store", hybrid=hybrid)
    return replace(
        config,
        ingest=replace(config.ingest, manifest_directory=directory / "manifest"),
    )
