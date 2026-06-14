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
  rag-core local-search examples/demo_corpus "corpus lifecycle" --demo --events-jsonl /tmp/events.jsonl
""",
    "local-eval": """\
Examples:
  rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --json
  rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --search-profile fast
""",
    "ingest": """\
Examples:
  rag-core ingest ./docs --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
  rag-core ingest ./docs.zip --namespace acme --corpus-id help --manifest-dir .rag-core/manifest --plan-json
  rag-core ingest https://example.com/docs/guide --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
  rag-core ingest --url-list ./urls.txt --namespace acme --corpus-id help --manifest-dir .rag-core/manifest --plan-json
""",
    "search": """\
Examples:
  rag-core search "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
  rag-core search "billing" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --search-profile balanced --json
  rag-core search --context "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
""",
    "manifest": """\
Examples:
  rag-core manifest ./docs/guide.md --namespace acme --corpus-id help --json
  rag-core manifest --compact --manifest-dir .rag-core/manifest --namespace acme --corpus-id help
""",
}


def apply_command_examples(
    parser: argparse.ArgumentParser,
    command: CommandName,
) -> None:
    parser.epilog = _EXAMPLES[command]
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
