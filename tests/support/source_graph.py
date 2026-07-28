"""Refactor-tolerant structural assertions for product-shape tests.

These helpers replace source-substring / hard-coded-path scraping with two
durable checks:

- the package IMPORT GRAPH (parsed with ``ast``), so architectural-boundary and
  "import the durable owner, not the stale catch-all" invariants survive file
  merges, renames, and reformatting; and
- runtime SYMBOL OWNERSHIP (via ``inspect``), so "this contract lives in its
  owner module" is asserted on where a symbol actually resolves, not on which
  file's text happens to mention it.

Value assertions (``CONST == "value"``) belong in the tests themselves; only the
brittle source-text twins move here. The goal is to keep the architectural
invariants the product-shape tests protect while letting the file layout change.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dotted_module(relative_path: str) -> str:
    parts = Path(relative_path).with_suffix("").parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_parts(relative_path: str) -> tuple[str, ...]:
    parts = tuple(_dotted_module(relative_path).split("."))
    parts = tuple(p for p in parts if p)
    if Path(relative_path).with_suffix("").name == "__init__":
        return parts
    return parts[:-1]


def iter_package_sources(*relative_roots: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(relative_path, dotted_module, source)`` for every .py under the roots."""
    for relative_root in relative_roots:
        for path in sorted((REPO_ROOT / relative_root).rglob("*.py")):
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            yield (
                relative_path,
                _dotted_module(relative_path),
                path.read_text(encoding="utf-8"),
            )


def imported_modules(relative_path: str, source: str) -> set[str]:
    """Fully-qualified modules a source file imports, with relative imports resolved.

    ``from a.b import c`` contributes both ``a.b`` and ``a.b.c`` so that callers can
    assert on either module-level boundaries or symbol-level ownership.
    """
    imported: set[str] = set()
    package = _package_parts(relative_path)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + tuple((node.module or "").split("."))).rstrip(".")
            else:
                module = node.module or ""
            if not module:
                continue
            imported.add(module)
            for alias in node.names:
                imported.add(f"{module}.{alias.name}")
    return imported


def import_graph(*relative_roots: str) -> dict[str, set[str]]:
    """``{dotted_module: set of fully-qualified modules it imports}`` for the roots."""
    return {
        dotted: imported_modules(rel, src)
        for rel, dotted, src in iter_package_sources(*relative_roots)
    }


def modules_importing(
    *relative_roots: str, predicate: Callable[[str], bool]
) -> dict[str, list[str]]:
    """``{module: sorted matching imports}`` for modules whose imports match ``predicate``.

    Empty result means no module under the roots imports anything matching the
    predicate -- the durable form of a ``"forbidden import" not in source`` assert.
    """
    offenders: dict[str, list[str]] = {}
    for dotted, imports in import_graph(*relative_roots).items():
        hits = sorted(module for module in imports if predicate(module))
        if hits:
            offenders[dotted] = hits
    return offenders


def under_module(prefix: str) -> Callable[[str], bool]:
    """Predicate matching ``prefix`` itself or any submodule under it."""
    return lambda module: module == prefix or module.startswith(f"{prefix}.")


def _defines_top_level_name(source: str, name: str) -> bool:
    for node in ast.iter_child_nodes(ast.parse(source)):
        if isinstance(node, ast.Assign):
            targets: tuple[ast.expr, ...] = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                return True
            continue
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return True
    return False


def defining_modules(*relative_roots: str, name: str) -> set[str]:
    """Dotted modules under the roots that bind ``name`` at top level.

    The durable form of "this constant/function has a single owner": its owner
    module appears, and a redefinition elsewhere would add a second module. Works
    for module-level constants (incl. type aliases), which ``symbol_module``
    cannot resolve because they carry no ``__module__``.
    """
    return {
        dotted
        for _rel, dotted, src in iter_package_sources(*relative_roots)
        if _defines_top_level_name(src, name)
    }


_UNSET = object()


def _literal_value(node: ast.expr) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner
    # ``Path("literal")`` / ``PurePath("literal")`` wrap a path constant; treat
    # the wrapped string as the assigned value so path defaults are pinned the
    # same way as bare string literals.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Path", "PurePath", "PurePosixPath"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return _literal_value(node.args[0])
    return _UNSET


def modules_assigning_value(*relative_roots: str, value: object) -> dict[str, list[str]]:
    """``{dotted_module: sorted constant names}`` assigned a literal equal to ``value``.

    Finds module-level ``NAME = <literal>`` / ``NAME: T = <literal>`` bindings whose
    right-hand side is the given string/number literal, across every module under the
    roots. A single-key result means exactly one module owns that literal value -- the
    durable form of the old ``owner.count(...) == 1`` plus ``literal not in consumers``
    pair, without freezing a hand-picked file list or pinning a call-site count.
    """
    owners: dict[str, list[str]] = {}
    for _rel, dotted, src in iter_package_sources(*relative_roots):
        names: list[str] = []
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign):
                targets: tuple[ast.expr, ...] = tuple(node.targets)
                rhs: ast.expr = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = (node.target,)
                rhs = node.value
            else:
                continue
            found = _literal_value(rhs)
            if found is _UNSET:
                continue
            if found != value or isinstance(found, bool) != isinstance(value, bool):
                continue
            names.extend(t.id for t in targets if isinstance(t, ast.Name))
        if names:
            owners[dotted] = sorted(names)
    return owners


def symbol_module(obj: object) -> str:
    """Dotted module where a symbol is actually defined -- its ownership contract."""
    module = inspect.getmodule(obj)
    assert module is not None, f"cannot resolve the owning module for {obj!r}"
    return module.__name__
