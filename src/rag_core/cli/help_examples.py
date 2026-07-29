"""Example blocks for core CLI subcommands."""

from __future__ import annotations

import argparse
from typing import Literal

CommandName = Literal[
    "demo",
    "doctor",
    "local-search",
    "local-eval",
    "eval",
    "add",
    "ingest",
    "search",
    "context",
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
  rag-core local-search examples/demo_corpus "corpus lifecycle" --demo --trace-jsonl /tmp/trace.jsonl
""",
    "local-eval": """\
Examples:
  rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --json
  rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --search-profile fast
""",
    "eval": """\
Examples:
  rag-core eval examples/demo_corpus examples/eval_cases.jsonl --json
  rag-core eval examples/demo_corpus examples/eval_cases.jsonl --search-profile fast
""",
    "add": """\
Examples:
  rag-core add ./docs --collection help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
  rag-core add ./docs.zip --collection help --manifest-dir .rag-core/manifest --dry-run --json
  rag-core add https://example.com/docs/guide --collection help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
  rag-core add --url-list ./urls.txt --collection help --manifest-dir .rag-core/manifest --dry-run --json
""",
    "ingest": """\
Examples:
  rag-core ingest ./docs --collection help --dry-run --json
""",
    "search": """\
Examples:
  rag-core search "billing policy" ./docs --json
  rag-core search "billing policy" --collection help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
  rag-core search "billing" --collections help,policies --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --search-profile balanced --json
""",
    "context": """\
Examples:
  rag-core context "billing policy" ./docs --json
  rag-core context "billing policy" --collection help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small
""",
    "manifest": """\
Examples:
  rag-core manifest ./docs/guide.md --collection help --json
  rag-core manifest --compact --manifest-dir .rag-core/manifest --collection help
""",
}


def apply_command_examples(
    parser: argparse.ArgumentParser,
    command: CommandName,
) -> None:
    parser.epilog = _EXAMPLES[command]
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
