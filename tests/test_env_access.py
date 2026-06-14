import pytest

from rag_core.config.env_access import parse_env_bool


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("true", True), ("false", False), ("wat", None)),
)
def test_parse_env_bool(raw: str, expected: bool | None) -> None:
    assert parse_env_bool(raw) is expected
