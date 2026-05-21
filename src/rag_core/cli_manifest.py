from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.cli_inputs import parse_metadata_fields
from rag_core.cli_output import require_mapping
from rag_core.local_corpus import (
    preview_manifest,
    validate_supported_local_file,
)
from rag_core.manifest_preview_models import ManifestPreviewRequest
from rag_core.manifest_persistence import compact_manifest, validate_manifest_scope


async def run_manifest_command(args: argparse.Namespace) -> int:
    validate_manifest_scope(args.namespace, args.corpus_id)
    path = Path(args.path)
    validate_supported_local_file(path, label="manifest path")
    result = await preview_manifest(
        ManifestPreviewRequest(
            path=path,
            namespace=args.namespace,
            corpus_id=args.corpus_id,
            document_id=args.document_id,
            document_key=args.document_key,
            metadata=parse_metadata_fields(args.metadata),
        )
    )
    _emit_manifest(result.to_payload(), as_json=args.json)
    return 0


async def run_manifest_compact_command(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.manifest_dir)
    if manifest_dir.exists() and not manifest_dir.is_dir():
        raise ValueError(f"manifest directory must be a directory: {manifest_dir}")
    result = compact_manifest(
        manifest_dir,
        namespace=args.namespace,
        corpus_id=args.corpus_id,
    )
    _emit_manifest_compaction(
        {
            "manifest_dir": str(args.manifest_dir),
            "namespace": args.namespace,
            "corpus_id": args.corpus_id,
            "before_entry_count": result.before_entry_count,
            "after_entry_count": result.after_entry_count,
            "removed_entry_count": result.removed_entry_count,
            "changed": result.changed,
        },
        as_json=args.json,
    )
    return 0


def _emit_manifest(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    document = require_mapping(payload.get("document"))
    entry = require_mapping(payload.get("manifest_entry"))
    print(f"Document ID: {document.get('document_id')}")
    print(f"Namespace: {document.get('namespace')}")
    print(f"Corpus: {document.get('corpus_id')}")
    print(f"Document Key: {entry.get('document_key')}")
    print(f"Chunks: {entry.get('chunk_count')}")
    print(f"Parser: {entry.get('parser') or 'unknown'}")
    print(f"Needs OCR: {entry.get('needs_ocr')}")


def _emit_manifest_compaction(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"Manifest Directory: {payload.get('manifest_dir')}")
    print(f"Namespace: {payload.get('namespace')}")
    print(f"Corpus: {payload.get('corpus_id')}")
    print(f"Entries Before: {payload.get('before_entry_count')}")
    print(f"Entries After: {payload.get('after_entry_count')}")
    print(f"Entries Removed: {payload.get('removed_entry_count')}")
    print(f"Changed: {payload.get('changed')}")
