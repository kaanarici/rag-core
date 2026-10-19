"""Persistent folder search with local embeddings and BM25, without API keys.

from rag_core.easy import index
with index("./my-docs") as docs:
    print(docs.context("how do invoices get paid?"))
"""

from __future__ import annotations

from pathlib import Path

from rag_core._sync import run_coro_blocking
from rag_core.core import Engine
from rag_core.ingest.local import LocalIngestResult
from rag_core.local import DEFAULT_LOCAL_TENANT_ID, local_collection, local_config
from rag_core.rag import RAGCore, RetrievalResult
from rag_core.scope import normalize_tenant_id


class Index:
    """A persistent folder index. Use as a context manager or call close()."""

    def __init__(
        self, core: Engine, *, path: Path, tenant_id: str, collection: str
    ) -> None:
        self._core = core
        self._path = path
        self._rag = RAGCore(core, tenant_id=tenant_id, collection=collection)

    @property
    def persist_dir(self) -> Path:
        location = self._core._config.vector_store.qdrant.location
        assert location is not None
        return Path(location).parent

    def refresh(self) -> LocalIngestResult:
        """Pick up edits and remove deleted files; unchanged files are skipped."""
        return run_coro_blocking(
            self._core.add(
                self._path,
                tenant_id=self._rag.tenant_id,
                collection=self._rag.collection,
                prune_missing=self._path.is_dir(),
            )
        )

    def search(self, question: str, *, limit: int = 5) -> RetrievalResult:
        """Return ranked evidence with source identity and locators."""
        return run_coro_blocking(self._rag.search(question, limit=limit))

    def context(self, question: str, *, limit: int = 5) -> str:
        """Return prompt-ready context plus citations."""
        pack = run_coro_blocking(self._rag.context(question, limit=limit))
        text = pack.as_prompt_text()
        summary = pack.prompt_citation_summary
        return f"{text}\n\nCitations:\n{summary}" if summary else text

    def close(self) -> None:
        run_coro_blocking(self._rag.close())

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def index(
    folder: str | Path,
    *,
    persist_dir: str | Path | None = None,
    hybrid: bool = True,
    collection: str | None = None,
    tenant_id: str | None = None,
) -> Index:
    """Open and refresh a persistent local index over folder.

    Uses meaning + keyword search by default; hybrid=False selects dense search.
    Storage is isolated by source path and search mode under persist_dir,
    RAG_CORE_INDEX_DIR, or ~/.cache/rag-core (respecting XDG_CACHE_HOME).
    The storage root must be outside the folder being indexed.
    """
    config = local_config(folder, persist_dir=persist_dir, hybrid=hybrid)
    path = Path(folder).expanduser().resolve(strict=True)
    core = Engine(config)
    try:
        idx = Index(
            core,
            path=path,
            tenant_id=normalize_tenant_id(
                DEFAULT_LOCAL_TENANT_ID if tenant_id is None else tenant_id
            ),
            collection=collection or local_collection(path),
        )
        result = idx.refresh()
        if result.failed:
            raise RuntimeError(
                f"could not index {result.failed_count} files from {folder}: "
                f"{result.failed[0].error}"
            )
        if result.succeeded_count == 0:
            raise RuntimeError(f"indexed no files from {folder}")
        return idx
    except BaseException:
        run_coro_blocking(core.close())
        raise


__all__ = ["Index", "index"]
