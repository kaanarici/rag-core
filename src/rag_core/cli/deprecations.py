from __future__ import annotations

import argparse
import sys
from typing import Any


def warn_deprecated_cli(old: str, new: str) -> None:
    print(f"warning: {old} is deprecated; use {new}", file=sys.stderr)


class DeprecatedStoreAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        replacement: str,
        **kwargs: Any,
    ) -> None:
        self.replacement = replacement
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser
        warn_deprecated_cli(option_string or self.dest, self.replacement)
        setattr(namespace, self.dest, values)


class DeprecatedAppendAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        replacement: str,
        **kwargs: Any,
    ) -> None:
        self.replacement = replacement
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser
        warn_deprecated_cli(option_string or self.dest, self.replacement)
        current = getattr(namespace, self.dest, None)
        items = [] if current is None else list(current)
        items.append(values)
        setattr(namespace, self.dest, items)


class DeprecatedStoreTrueAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        replacement: str,
        **kwargs: Any,
    ) -> None:
        self.replacement = replacement
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, values
        warn_deprecated_cli(option_string or self.dest, self.replacement)
        setattr(namespace, self.dest, True)


__all__ = [
    "DeprecatedAppendAction",
    "DeprecatedStoreAction",
    "DeprecatedStoreTrueAction",
    "warn_deprecated_cli",
]
