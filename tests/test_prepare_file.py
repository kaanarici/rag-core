import asyncio
import os
from pathlib import Path
import pytest
from rag_core import Config
from rag_core.core import Engine
from tests.support import FakeEmbeddingProvider, RecordingVectorStore


def test_prepare_file_uses_native_parser_without_indexing(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Billing\n\nInvoices can be paid by ACH.")
    embedding, store = FakeEmbeddingProvider(), RecordingVectorStore()

    async def scenario() -> None:
        async with Engine(
            Config.local(), embedding_provider=embedding, vector_store=store
        ) as core:
            prepared = await core.prepare_file(path)
            assert prepared.metadata["parser"] == "local:text"
            assert prepared.chunks and "ACH" in prepared.markdown
            assert not store.upsert_calls
            assert not embedding.embed_texts_calls

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_prepare_file_rejects_local_aliases(tmp_path: Path, kind: str) -> None:
    path, alias = tmp_path / "source.md", tmp_path / "alias.md"
    path.write_text("Private document.")
    if kind == "symlink":
        alias.symlink_to(path)
    else:
        os.link(path, alias)

    async def scenario() -> None:
        async with Engine(
            Config.local(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=RecordingVectorStore(),
        ) as core:
            with pytest.raises(ValueError, match="symlink|multi-link"):
                await core.prepare_file(alias)

    asyncio.run(scenario())
