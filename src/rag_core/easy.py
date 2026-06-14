"""The easy path: index a folder, ask a question. No config, no async, no scope.

    import rag_core
    rag = rag_core.index("./my-docs")
    print(rag.ask("how do invoices get paid?"))

This uses real local semantic embeddings (FastEmbed, no API key) over an
embedded vector store. For production, async, multi-tenant scope, custom
providers, reranking, contextual retrieval, or HTTP serving, use ``RAGCore``
and ``RAGCoreConfig`` directly. See https://kaanarici.github.io/rag-core/docs/embed.
"""

from __future__ import annotations

from pathlib import Path

from rag_core._sync import run_coro_blocking
from rag_core.core import RAGCore
from rag_core.core_models import RAGCoreConfig
from rag_core.search.vector_models import SearchResult

_DEFAULT_NAMESPACE = "local"
_DEFAULT_CORPUS = "default"


class Rag:
    """A ready-to-query index over a folder. Returned by :func:`index`.

    Use as a context manager (``with rag_core.index(...) as rag:``) or call
    :meth:`close` when done so the embedded store releases its resources.
    """

    def __init__(self, core: RAGCore, *, namespace: str, corpus: str) -> None:
        self._core = core
        self._namespace = namespace
        self._corpus = corpus

    def ask(self, question: str, *, limit: int = 5) -> str:
        """Return prompt-ready context plus citations for ``question``."""
        pack = run_coro_blocking(
            self._core.retrieve_context(
                query=question,
                namespace=self._namespace,
                corpus_ids=[self._corpus],
                limit=limit,
            )
        )
        text = pack.as_prompt_text()
        summary = pack.prompt_citation_summary
        return f"{text}\n\nCitations:\n{summary}" if summary else text

    def search(self, question: str, *, limit: int = 5) -> list[SearchResult]:
        """Return ranked hits for ``question`` (for custom prompt assembly)."""
        return run_coro_blocking(
            self._core.search(
                query=question,
                namespace=self._namespace,
                corpus_ids=[self._corpus],
                limit=limit,
            )
        )

    def close(self) -> None:
        run_coro_blocking(self._core.close())

    def __enter__(self) -> "Rag":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def index(
    folder: str | Path,
    *,
    namespace: str = _DEFAULT_NAMESPACE,
    corpus: str = _DEFAULT_CORPUS,
) -> Rag:
    """Index ``folder`` for retrieval and return a queryable :class:`Rag`.

    Real local semantic embeddings, no API key, embedded store. ``namespace``
    and ``corpus`` default to a single local scope; pass them only for
    multi-tenant separation.
    """
    core = RAGCore(RAGCoreConfig.local())
    try:
        result = run_coro_blocking(
            core.ingest_files(folder, namespace=namespace, corpus_id=corpus)
        )
    except ValueError as exc:
        # An empty folder / no-match raises a glob-hint ValueError one layer down;
        # the day-one facade reports the plain outcome instead of leaking it.
        run_coro_blocking(core.close())
        raise RuntimeError(f"indexed no files from {folder}") from exc
    if result.succeeded_count == 0:
        run_coro_blocking(core.close())
        raise RuntimeError(f"indexed no files from {folder}")
    return Rag(core, namespace=namespace, corpus=corpus)


__all__ = ["Rag", "index"]
