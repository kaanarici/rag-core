"""Ingest local, archive, and URL sources through one Engine instance."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from rag_core.demo import build_demo_core
from rag_core.fetch_security import validate_fetch_url
from rag_core.fetching import FetchResponse

_REMOTE_URL = "https://example.com/docs/remote-guide?token=secret"
_REMOTE_BODY = b"Remote guide pages can be fetched, parsed, indexed, and cited."


class StaticFetchClient:
    def fetch(self, url: str) -> FetchResponse:
        validated = validate_fetch_url(url)
        return FetchResponse(
            url=validated,
            status_code=200,
            content_type="text/plain",
            content_length=len(_REMOTE_BODY),
            content_sha256=hashlib.sha256(_REMOTE_BODY).hexdigest(),
            body=_REMOTE_BODY,
            redirect_chain=(validated,),
        )


def _write_sources(root: Path) -> tuple[Path, Path]:
    local_file = root / "local-guide.md"
    local_file.write_text(
        "# Local Guide\n\nLocal files can be parsed and indexed directly.",
        encoding="utf-8",
    )

    archive_path = root / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "archive-guide.md",
            "# Archive Guide\n\nZIP members are parsed without extracting to disk.",
        )
    return local_file, archive_path


async def run_demo() -> dict[str, object]:
    core = build_demo_core(store_collection="source_ingest")
    async with core:
        with tempfile.TemporaryDirectory() as tmp:
            local_file, archive_path = _write_sources(Path(tmp))
            local_doc = await core.add_file(
                local_file,
                namespace="acme",
                collection="sources",
                document_key="local-guide.md",
            )
            archive_result = await core.add_archive(
                archive_path,
                namespace="acme",
                collection="sources",
            )
            remote_doc = await core.add_url(
                _REMOTE_URL,
                namespace="acme",
                collection="sources",
                fetch_client=StaticFetchClient(),
            )

        context = await core.context(
            query="Which sources can be indexed and cited?",
            namespace="acme",
            collections=["sources"],
            limit=5,
            rerank=False,
            max_chars=1_500,
        )
        return {
            "local_document_key": local_doc.document_key,
            "archive_written_count": archive_result.written_count,
            "remote_document_key": f"url:{remote_doc.metadata['source_url']}",
            "citation_count": len(context.citations),
            "context_text": context.as_prompt_text(),
        }


def main() -> None:
    print(json.dumps(asyncio.run(run_demo()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
