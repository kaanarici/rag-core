"""Real Qdrant lifecycle and API contracts; deterministic embeddings, no quality claim."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import pytest

from rag_core import Config, Document, RAGCore
from rag_core.config import ChunkingConfig
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.easy import index
from rag_core.local import local_config
from rag_core.manifest.persistence import read_entries

pytestmark = pytest.mark.integration


class _DocumentScope(TypedDict):
    tenant_id: str
    collection: str
    document_id: str


class CountingEmbedding(DemoEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=64)
        self.documents = 0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.documents += len(texts)
        return await super().embed_texts(texts)


@pytest.fixture
def local_embedding(monkeypatch: pytest.MonkeyPatch) -> CountingEmbedding:
    embedding = CountingEmbedding()
    monkeypatch.setattr(
        "rag_core.search.providers.embedding.create_embedding_provider",
        lambda **kwargs: embedding,
    )
    monkeypatch.setattr(
        "rag_core.search.providers.sparse.FastEmbedSparseEmbedder",
        lambda **kwargs: DemoSparseEmbedder(),
    )
    return embedding


def test_folder_index_reopens_skips_unchanged_and_removes_deleted_files(
    tmp_path: Path, local_embedding: CountingEmbedding
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    billing = folder / "billing.md"
    support = folder / "support.md"
    billing.write_text("Invoices can be paid by ACH.")
    support.write_text("Support tickets are reviewed on weekdays.")
    storage = tmp_path / "indexes"

    with index(folder, persist_dir=storage) as docs:
        search = docs._core.describe_runtime()["search"]
        assert isinstance(search, dict)
        assert search["mode"] == "hybrid"
        directory = docs.persist_dir
        first = docs.search("How are invoices paid?")
        assert any("billing.md" in (e.document_key or "") for e in first.evidence)
        assert "Citations:" in docs.context("How are invoices paid?")
    embedded = local_embedding.documents

    with index(folder, persist_dir=storage) as docs:
        assert docs.persist_dir == directory
        assert local_embedding.documents == embedded
        assert all(r.ingest_state == "unchanged" for r in docs.refresh().succeeded)
        billing.write_text("Invoices can now be paid by card only.")
        assert {r.ingest_state for r in docs.refresh().succeeded} == {
            "replaced",
            "unchanged",
        }
        assert all(
            "paid by ACH" not in e.text for e in docs.search("invoices").evidence
        )
        billing.rename(tmp_path / "removed-billing.md")
        docs.refresh()
        assert all(
            "billing.md" not in (e.document_key or "")
            for e in docs.search("invoices").evidence
        )
        support.rename(tmp_path / "removed-support.md")
        assert docs.refresh().succeeded_count == 0
        assert docs.search("invoices").evidence == ()


@pytest.mark.parametrize("hybrid", [False, True])
def test_same_named_folders_do_not_share_results(
    tmp_path: Path, local_embedding: CountingEmbedding, hybrid: bool
) -> None:
    one, two = tmp_path / "one" / "docs", tmp_path / "two" / "docs"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    (one / "alpha.md").write_text("Alpha billing policy.")
    (two / "beta.md").write_text("Beta billing policy.")
    with index(one, persist_dir=tmp_path / "indexes", hybrid=hybrid) as a:
        first_dir = a.persist_dir
    with index(two, persist_dir=tmp_path / "indexes", hybrid=hybrid) as b:
        assert b.persist_dir != first_dir
        assert all("Beta" in e.text for e in b.search("billing").evidence)
    with index(one, persist_dir=tmp_path / "indexes", hybrid=hybrid) as a:
        assert all("Alpha" in e.text for e in a.search("billing").evidence)


def test_dense_and_hybrid_folder_indexes_can_be_reopened_independently(
    tmp_path: Path, local_embedding: CountingEmbedding
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "billing.md").write_text("Invoices can be paid by ACH.")
    directories = []
    for hybrid in [False, True, False, True]:
        with index(folder, persist_dir=tmp_path / "indexes", hybrid=hybrid) as docs:
            directories.append(docs.persist_dir)
            assert docs.search("invoices").evidence
    assert directories[0] == directories[2]
    assert directories[1] == directories[3]
    assert directories[0] != directories[1]


def test_hybrid_app_context_preserves_the_bound_document_scope(
    tmp_path: Path, local_embedding: CountingEmbedding
) -> None:
    async def scenario() -> None:
        async with RAGCore(
            Config.local(tmp_path / "app-index", hybrid=True),
            tenant_id="acme",
            collection="docs",
            document_ids=("billing",),
        ) as rag:
            await rag.ingest(
                Document(
                    id="billing",
                    key="billing.md",
                    content=b"Invoices can be paid by ACH.",
                    content_type="text/markdown",
                )
            )
            result = await rag.search("invoices")
            pack = await rag.context("invoices")
            assert [s.text for s in pack.snippets] == [e.text for e in result.evidence]
            assert pack.citations
            with pytest.raises(
                ValueError, match="outside the configured retrieval scope"
            ):
                await rag.context("invoices", document_ids=("other",))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changed",
    [
        ChunkingConfig(max_chars=500, overlap=100),
        ChunkingConfig(max_chars=2000, overlap=0),
        ChunkingConfig(strategy="markdown"),
    ],
)
def test_chunk_settings_reindex_unchanged_source_after_reopen(
    tmp_path: Path, local_embedding: CountingEmbedding, changed: ChunkingConfig
) -> None:
    content = ("Invoice payment terms and conditions. " * 120).encode()
    config = Config.local(tmp_path / "store")

    async def ingest(current: Config):
        async with RAGCore(current, tenant_id="acme", collection="docs") as rag:
            return await rag.ingest(
                Document(
                    key="billing.md", content=content, content_type="text/markdown"
                )
            )

    original = asyncio.run(ingest(config))
    replaced = asyncio.run(ingest(replace(config, chunking=changed)))
    unchanged = asyncio.run(ingest(replace(config, chunking=changed)))
    assert replaced.document_id == original.document_id
    assert replaced.status == "replaced"
    assert unchanged.status == "unchanged"
    if changed.max_chars == 500:
        assert replaced.chunk_count > original.chunk_count


def test_truncated_scan_does_not_prune_unseen_files_or_other_roots(
    tmp_path: Path, local_embedding: CountingEmbedding
) -> None:
    folder = tmp_path / "docs"
    other = tmp_path / "other"
    folder.mkdir()
    other.mkdir()
    for name in ["alpha", "beta", "gamma"]:
        (folder / f"{name}.md").write_text(f"{name} billing policy.")
    (other / "outside.md").write_text("Other folder policy.")
    config = local_config(folder, persist_dir=tmp_path / "indexes")
    manifest_dir = config.ingest.manifest_directory
    assert manifest_dir is not None

    async def scenario() -> None:
        async with Engine(config) as core:
            await core.add(
                folder, tenant_id="local", collection="docs", prune_missing=True
            )
            await core.add(other, tenant_id="local", collection="docs")
            (folder / "gamma.md").rename(tmp_path / "removed.md")
            partial = await core.add(
                folder,
                tenant_id="local",
                collection="docs",
                prune_missing=True,
                max_files=1,
            )
            assert partial.truncated is True
            assert len(read_entries(manifest_dir, "local", "docs")) == 4
            await core.add(
                folder, tenant_id="local", collection="docs", prune_missing=True
            )
            entries = read_entries(manifest_dir, "local", "docs")
            assert len(entries) == 3
            assert any(e.filename == "outside.md" for e in entries)

    asyncio.run(scenario())


def test_unreadable_subfolder_does_not_prune_stored_documents(
    tmp_path: Path, local_embedding: CountingEmbedding, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    folder = tmp_path / "docs"
    hidden = folder / "subfolder"
    hidden.mkdir(parents=True)
    (folder / "billing.md").write_text("Invoices can be paid by ACH.")
    (hidden / "support.md").write_text("Support tickets are reviewed on weekdays.")
    original_scandir = os.scandir

    def blocked_scandir(path):
        if Path(path) == hidden:
            raise PermissionError("subfolder unavailable")
        return original_scandir(path)

    with index(folder, persist_dir=tmp_path / "indexes") as docs:
        before = docs.search("support")
        assert any("support.md" in (e.document_key or "") for e in before.evidence)
        with monkeypatch.context() as patch:
            patch.setattr(os, "scandir", blocked_scandir)
            with pytest.raises(PermissionError, match="subfolder unavailable"):
                docs.refresh()
        after = docs.search("support")
        assert after == before


def test_injected_sparse_provider_changes_processing_identity(
    local_embedding: CountingEmbedding,
) -> None:
    from rag_core.core_models import ProcessingFingerprint

    async def scenario() -> list[bool]:
        states = []
        for sparse in [None, DemoSparseEmbedder()]:
            async with Engine(
                Config.local(),
                embedding_provider=local_embedding,
                sparse_embedder=sparse,
            ) as core:
                raw = core.describe_runtime()["pipeline_version"]
                assert isinstance(raw, str)
                states.append(ProcessingFingerprint.parse(raw).sparse_index)
        return states

    assert asyncio.run(scenario()) == [False, True]


@pytest.mark.parametrize("replace_existing", [False, True])
def test_cancelled_partial_upsert_is_removed_and_retry_is_complete(
    tmp_path: Path,
    replace_existing: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Sequence
    from rag_core.search.vector_models import VectorPoint

    async def scenario() -> None:
        config = replace(
            Config.local(tmp_path / "store"),
            chunking=ChunkingConfig(max_chars=300, overlap=20),
            ingest=replace(
                Config.local().ingest, manifest_directory=tmp_path / "manifest"
            ),
        )
        async with Engine(config, embedding_provider=CountingEmbedding()) as core:
            scope: _DocumentScope = dict(tenant_id="local", collection="docs", document_id="cancelled")
            if replace_existing:
                await core.add_bytes(
                    file_bytes=b"Original source before replacement.",
                    filename="doc.md",
                    mime_type="text/markdown",
                    **scope,
                )
            original_upsert = core._store.upsert
            partial = asyncio.Event()

            async def interrupted(points: Sequence[VectorPoint]) -> None:
                assert len(points) > 1
                await original_upsert(points[:1])
                partial.set()
                await asyncio.Event().wait()

            data = (
                "A lengthy invoice policy paragraph, paid by ACH and reviewed by support. "
                * 40
            ).encode()
            with monkeypatch.context() as patch:
                patch.setattr(core._store, "upsert", interrupted)
                task = asyncio.create_task(
                    core.add_bytes(
                        file_bytes=data,
                        filename="doc.md",
                        mime_type="text/markdown",
                        **scope,
                    )
                )
                await asyncio.wait_for(partial.wait(), timeout=3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            previous = await core._store.get_document_record(**scope)
            if replace_existing:
                assert previous is not None and previous.is_complete and previous.chunk_count == 1
                assert len(read_entries(tmp_path / "manifest", "local", "docs")) == 1
            else:
                assert previous is None
                assert read_entries(tmp_path / "manifest", "local", "docs") == []
            restored = await core.add_bytes(
                file_bytes=data, filename="doc.md", mime_type="text/markdown", **scope
            )
            record = await core._store.get_document_record(**scope)
            assert record is not None and record.chunk_count == restored.chunk_count > 1
            assert len(read_entries(tmp_path / "manifest", "local", "docs")) == 1
            assert core._ingest._fence.active_keys() == ()

    asyncio.run(scenario())
