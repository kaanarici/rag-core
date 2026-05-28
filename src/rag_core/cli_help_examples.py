"""Example blocks for core CLI subcommands."""

from __future__ import annotations

import argparse
from typing import Literal

CommandName = Literal[
    "demo",
    "doctor",
    "local-search",
    "local-eval",
    "ingest",
    "search",
    "retrieve-context",
    "manifest",
]

_EXAMPLES: dict[CommandName, str] = {
    "demo": """\
Examples:
  rag-core demo
  rag-core demo --json
""",
    "doctor": """\
Examples:
  rag-core doctor
  rag-core doctor --json
  rag-core doctor --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64 --json
  rag-core doctor --check-store --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64 --json
""",
    "local-search": """\
Examples:
  rag-core local-search /tmp/docs "billing policy" --json
  rag-core local-search examples/demo_corpus "corpus lifecycle" --events-jsonl /tmp/events.jsonl
""",
    "local-eval": """\
Examples:
  rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --json
  rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --min-recall-at-5 1 --min-mrr 1
""",
    "ingest": """\
Examples:
  rag-core ingest ./docs --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536 --json
  rag-core ingest ./docs --namespace acme --corpus-id help --manifest-dir .rag-core/manifest
""",
    "search": """\
Examples:
  rag-core search "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536 --json
  rag-core search "billing" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536 --search-profile balanced --json
""",
    "retrieve-context": """\
Examples:
  rag-core retrieve-context "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
""",
    "manifest": """\
Examples:
  rag-core manifest ./docs/guide.md --namespace acme --corpus-id help --json
  rag-core manifest-compact --manifest-dir .rag-core/manifest
""",
}


def apply_command_examples(
    parser: argparse.ArgumentParser,
    command: CommandName,
) -> None:
    parser.epilog = _EXAMPLES[command]
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
