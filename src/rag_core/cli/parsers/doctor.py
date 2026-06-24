from __future__ import annotations

import argparse

from rag_core.cli.parsers.config import add_config_flags
from rag_core.cli.help_examples import apply_command_examples


def add_doctor_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    doctor = subparsers.add_parser(
        "doctor",
        help="Print the planned Engine/provider shape and optionally verify the vector store.",
    )
    add_config_flags(doctor)
    doctor.add_argument(
        "--check-store",
        action="store_true",
        help="Create/check the configured vector store and include health data.",
    )
    doctor.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Opt in to creating the collection when missing. Reports a dimension "
            "diff and exits nonzero on mismatch (mismatches are not auto-fixed)."
        ),
    )
    doctor.add_argument("--json", action="store_true", help="Emit JSON output.")
    doctor.description = (
        "Inspect collection/provider shape. This command reports config-level "
        "Engine assembly details, not every programmatic Config field."
    )
    apply_command_examples(doctor, "doctor")


__all__ = ["add_doctor_command"]
