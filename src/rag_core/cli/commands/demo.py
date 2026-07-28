from __future__ import annotations

import argparse
import json
import sys

from rag_core.demo import run_demo_app


async def run_demo_command(args: argparse.Namespace) -> int:
    print(
        "Note: demo uses deterministic demo embeddings, not semantic search.",
        file=sys.stderr,
    )
    payload = await run_demo_app()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"Indexed document: {payload['document_id']} ({payload['chunk_count']} chunks)")
    print("Top hits:")
    for raw_hit in payload["hits"]:
        score = raw_hit["score"]
        title = raw_hit["title"]
        text = raw_hit["text"]
        print(f"- {score:.3f} {title}: {text[:80]}")
    return 0
