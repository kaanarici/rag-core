from __future__ import annotations

from pathlib import Path

import pytest

_META_TEST_FILE_NAMES = frozenset(
    {"test_architecture_pressure.py", "test_product_docs_shape.py"}
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        name = Path(str(item.path)).name
        if name.endswith("_product_shape.py") or name in _META_TEST_FILE_NAMES:
            item.add_marker(pytest.mark.meta)
