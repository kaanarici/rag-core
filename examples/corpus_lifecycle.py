"""Application-owned collection manifest around rag-core document lifecycle calls."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from rag_core import Engine, SearchResult
from rag_core.core_models import CollectionManifestEntry
from rag_core.demo import build_demo_core
from rag_core.retrieval_defaults import DEFAULT_RERANK, DEFAULT_SEARCH_LIMIT


def manifest_key(*, namespace: str, collection: str, document_key: str) -> str:
    return f"{namespace}:{collection}:{document_key}"


async def ingest_into_manifest(
    core: Engine,
    *,
    manifest: dict[str, CollectionManifestEntry],
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    namespace: str,
    collection: str,
    metadata: dict[str, str] | None = None,
) -> CollectionManifestEntry:
    ingested = await core.add_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        namespace=namespace,
        collection=collection,
        metadata=metadata,
    )
    document_key = ingested.document_key or ingested.filename
    key = manifest_key(
        namespace=namespace,
        collection=collection,
        document_key=document_key,
    )
    entry = core.build_manifest_entry(document=ingested)
    manifest[key] = entry
    return entry


async def search_corpus(
    core: Engine,
    *,
    entry: CollectionManifestEntry,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    rerank: bool = DEFAULT_RERANK,
) -> list[SearchResult]:
    return await core.search(
        query=query,
        namespace=entry.namespace,
        collections=[entry.collection],
        limit=limit,
        rerank=rerank,
    )


async def delete_from_manifest(
    core: Engine,
    *,
    manifest: dict[str, CollectionManifestEntry],
    key: str,
) -> CollectionManifestEntry:
    entry = manifest.pop(key)
    await core.delete_document(
        document_id=entry.document_id,
        namespace=entry.namespace,
        collection=entry.collection,
    )
    return entry


def manifest_row(entry: CollectionManifestEntry) -> dict[str, object]:
    return asdict(entry)


def preview_text(hit: SearchResult, *, max_chars: int = 96) -> str:
    text = hit.text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


async def run_demo() -> None:
    core = build_demo_core(store_collection="corpus_lifecycle")
    manifest: dict[str, CollectionManifestEntry] = {}

    async with core:
        billing_entry = await ingest_into_manifest(
            core,
            manifest=manifest,
            file_bytes=b"Billing is due monthly. Payment methods include card and ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            collection="help-center",
            metadata={"source": "quickstart"},
        )
        await ingest_into_manifest(
            core,
            manifest=manifest,
            file_bytes=b"Shipping times are 3-5 business days in the continental US.",
            filename="shipping.txt",
            mime_type="text/plain",
            namespace="acme",
            collection="help-center",
            metadata={"source": "quickstart"},
        )

        hits = await search_corpus(
            core,
            entry=billing_entry,
            query="How can I pay invoices?",
            limit=3,
        )
        print("Manifest keys:")
        for key in sorted(manifest):
            print(f"- {key}")
        print("\nTop hits:")
        for hit in hits:
            title = hit.title or hit.document_id or "unknown"
            print(f"- {hit.score:.3f} {title}: {preview_text(hit)}")

        first_key = sorted(manifest)[0]
        deleted = await delete_from_manifest(core, manifest=manifest, key=first_key)
        print(f"\nDeleted document: {deleted.document_id}")
        print(f"Remaining manifest entries: {len(manifest)}")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
