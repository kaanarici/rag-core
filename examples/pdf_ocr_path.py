"""Inspect how `rag-core` routes one PDF through parse and prepare steps."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rag_core import Engine
from rag_core.core_models import PreparedDocument
from rag_core.demo import build_demo_core


async def inspect_pdf_route(
    core: Engine,
    *,
    file_bytes: bytes,
    filename: str,
    path: str | None = None,
) -> dict[str, object]:
    parsed = await core.parse_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type="application/pdf",
        path=path,
    )
    return {
        "parser": parsed.metadata.get("parser"),
        "needs_ocr": bool(parsed.metadata.get("needs_ocr")),
        "ocr_page_indices": _normalize_page_indices(parsed.metadata.get("ocr_page_indices")),
    }


async def prepare_pdf_for_ingest(
    core: Engine,
    *,
    file_bytes: bytes,
    filename: str,
    path: str | None = None,
) -> PreparedDocument:
    return await core.prepare_bytes(
        file_bytes=file_bytes,
        filename=filename,
        mime_type="application/pdf",
        path=path,
    )


def describe_pdf_runtime(core: Engine) -> dict[str, object]:
    runtime = core.describe_runtime()
    return {
        "ocr": runtime.get("ocr"),
        "pdf_inspector": runtime.get("pdf_inspector"),
    }


def _normalize_page_indices(raw_indices: object) -> list[int]:
    if not isinstance(raw_indices, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        if not isinstance(raw_index, int) or raw_index < 0 or raw_index in seen:
            continue
        seen.add(raw_index)
        normalized.append(raw_index)
    return sorted(normalized)


async def run_demo(pdf_path: Path) -> None:
    core = build_demo_core(store_collection="pdf_ocr")
    file_bytes = pdf_path.read_bytes()

    async with core:
        runtime = describe_pdf_runtime(core)
        route = await inspect_pdf_route(
            core,
            file_bytes=file_bytes,
            filename=pdf_path.name,
            path=str(pdf_path),
        )
        prepared = await prepare_pdf_for_ingest(
            core,
            file_bytes=file_bytes,
            filename=pdf_path.name,
            path=str(pdf_path),
        )

        print("Runtime:")
        print(json.dumps(runtime, indent=2, sort_keys=True))
        print("\nRoute:")
        print(json.dumps(route, indent=2, sort_keys=True))
        print("\nPrepared:")
        print(
            json.dumps(
                {
                    "parser": prepared.metadata.get("parser"),
                    "needs_ocr": prepared.ocr.needed,
                    "ocr_page_indices": prepared.ocr.page_indices,
                    "chunk_count": len(prepared.chunks),
                },
                indent=2,
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect rag-core PDF routing for one file.")
    parser.add_argument("pdf", type=Path, help="Path to a local PDF file.")
    args = parser.parse_args()
    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")
    asyncio.run(run_demo(args.pdf))


if __name__ == "__main__":
    main()
