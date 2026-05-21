from __future__ import annotations

import argparse


def add_demo_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    demo = subparsers.add_parser(
        "demo",
        help="Run the built-in local demo without vendor API keys or external Qdrant.",
    )
    demo.add_argument("--json", action="store_true", help="Emit JSON output.")


__all__ = ["add_demo_command"]
