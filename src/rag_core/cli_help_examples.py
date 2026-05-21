"""CLI-for-agents Examples blocks on core subcommands."""

from __future__ import annotations

import argparse
from typing import Literal

CommandName = Literal[
    "demo",
    "doctor",
    "local-search",
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
  rag-core doctor --json
  rag-core doctor --qdrant-location :memory: --embedding-model text-embedding-3-small --json
  rag-core doctor --check-store --qdrant-location :memory: --embedding-model text-embedding-3-small --json
""",
    "local-search": """\
Examples:
  rag-core local-search /tmp/docs "billing policy" --json
  rag-core local-search examples/demo_corpus "corpus lifecycle" --events-jsonl /tmp/events.jsonl
""",
    "ingest": """\
Examples:
  rag-core ingest ./docs --namespace acme --corpus-id help --qdrant-location :memory: --json
  rag-core ingest ./docs --namespace acme --corpus-id help --manifest-dir .rag-core/manifest
""",
    "search": """\
Examples:
  rag-core search "billing policy" --namespace acme --corpus-id help --qdrant-location :memory: --json
  rag-core search "billing" --namespace acme --corpus-id help --search-profile balanced --json
""",
    "retrieve-context": """\
Examples:
  rag-core retrieve-context "billing policy" --namespace acme --corpus-id help --qdrant-location :memory: --json
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
