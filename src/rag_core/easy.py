"""The easy path: index a folder, ask for context. No config, no async, no scope.

    from rag_core import index
    idx = index("./my-docs")
    print(idx.context("how do invoices get paid?"))

This uses real local semantic embeddings (FastEmbed, no API key) over an
embedded vector store. For production, async, multi-tenant scope, custom
providers, reranking, contextual retrieval, or HTTP serving, use ``Engine``
and ``Config`` directly. See https://kaanarici.github.io/rag-core/docs/api-python.
"""

from __future__ import annotations

from pathlib import Path

from rag_core._sync import run_coro_blocking
from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.scope import Scope
from rag_core.search.vector_models import Evidence

_DEFAULT_COLLECTION = "default"


class Index:
    """A ready-to-query index over a folder. Returned by :func:`index`.

    Use as a context manager (``with index(...) as idx:``) or call
    :meth:`close` when done so the embedded store releases its resources.
    """

    def __init__(self, core: Engine, *, namespace: str, collection: str) -> None:
        self._core = core
        self._namespace = namespace
        self._collection = collection

    def context(self, question: str, *, limit: int = 5) -> str:
        """Return prompt-ready context plus citations for ``question``."""
        pack = run_coro_blocking(
            self._core.context(
                query=question,
                namespace=self._namespace,
                collection=self._collection,
                limit=limit,
            )
        )
        text = pack.as_prompt_text()
        summary = pack.prompt_citation_summary
        return f"{text}\n\nCitations:\n{summary}" if summary else text

    def search(self, question: str, *, limit: int = 5) -> list[Evidence]:
        """Return ranked hits for ``question`` (for custom prompt assembly)."""
        result = run_coro_blocking(
            self._core.search(
                question,
                scope=Scope(
                    tenant_id=self._namespace,
                    collection=self._collection,
                ),
                limit=limit,
            )
        )
        return list(result.evidence)

    def close(self) -> None:
        run_coro_blocking(self._core.close())

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def index(
    folder: str | Path,
    *,
    collection: str = _DEFAULT_COLLECTION,
    namespace: str | None = None,
) -> Index:
    """Index ``folder`` for retrieval and return a queryable :class:`Index`.

    Real local semantic embeddings, no API key, embedded store. ``collection``
    defaults to a single local scope; pass ``namespace`` only for multi-tenant
    separation.
    """
    scope = Scope(tenant_id=namespace or "default", collection=collection)
    core = Engine(Config.local())
    try:
        result = run_coro_blocking(
            core.add(
                folder,
                collection=scope.collection,
                namespace=scope.tenant_id,
            )
        )
    except ValueError as exc:
        run_coro_blocking(core.close())
        if str(exc).startswith("no supported files matched "):
            raise RuntimeError(f"indexed no files from {folder}") from exc
        raise
    except Exception:
        run_coro_blocking(core.close())
        raise
    if result.succeeded_count == 0:
        run_coro_blocking(core.close())
        raise RuntimeError(f"indexed no files from {folder}")
    return Index(core, namespace=scope.tenant_id, collection=scope.collection)


__all__ = ["Index", "index"]
