from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_core.cli.inputs import parse_metadata_fields
from rag_core.cli.output import require_mapping
from rag_core.ingest.local import (
    preview_manifest,
    validate_supported_local_file,
)
from rag_core.manifest.preview.models import ManifestPreviewRequest
from rag_core.manifest.persistence import compact_manifest, validate_manifest_scope


async def run_manifest_command(args: argparse.Namespace) -> int:
    if not args.collection:
        raise ValueError("--collection is required")
    if args.compact:
        return _run_manifest_compaction(args)
    if args.path is None:
        raise ValueError("manifest path is required unless --compact is passed")
    validate_manifest_scope(args.namespace, args.collection)
    path = Path(args.path)
    validate_supported_local_file(path, label="manifest path")
    result = await preview_manifest(
        ManifestPreviewRequest(
            path=path,
            namespace=args.namespace,
            collection=args.collection,
            document_id=args.document_id,
            document_key=args.document_key,
            metadata=parse_metadata_fields(args.metadata),
        )
    )
    _emit_manifest(result.to_payload(), as_json=args.json)
    return 0


def _run_manifest_compaction(args: argparse.Namespace) -> int:
    if args.path is not None:
        raise ValueError("manifest --compact does not accept a file path")
    if args.document_id is not None or args.document_key is not None:
        raise ValueError("--document-id and --document-key require a manifest path")
    if args.metadata:
        raise ValueError("--metadata requires a manifest path")
    manifest_dir = Path(args.manifest_dir)
    if manifest_dir.exists() and not manifest_dir.is_dir():
        raise ValueError(f"manifest directory must be a directory: {manifest_dir}")
    result = compact_manifest(
        manifest_dir,
        namespace=args.namespace,
        collection=args.collection,
    )
    _emit_manifest_compaction(
        {
            "manifest_dir": str(args.manifest_dir),
            "namespace": args.namespace,
            "collection": args.collection,
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
    print(f"Corpus: {document.get('collection')}")
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
    print(f"Corpus: {payload.get('collection')}")
    print(f"Entries Before: {payload.get('before_entry_count')}")
    print(f"Entries After: {payload.get('after_entry_count')}")
    print(f"Entries Removed: {payload.get('removed_entry_count')}")
    print(f"Changed: {payload.get('changed')}")
