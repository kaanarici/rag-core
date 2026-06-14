from __future__ import annotations

import pytest

from rag_core.documents.ocr_command_runtime import normalize_ocr_page_indices
from rag_core.documents.ocr_commands import gemini as gemini_command
from rag_core.documents.ocr_commands import mistral as mistral_command


def test_wrapper_page_indices_drop_bool_and_malformed_values() -> None:
    assert normalize_ocr_page_indices([2, True, 0, False, 2, -1, "3", 1.0]) == [0, 2]


@pytest.mark.parametrize("command_module", [gemini_command, mistral_command])
def test_command_page_indices_drop_bool_and_malformed_values(
    command_module: object,
) -> None:
    normalize = getattr(command_module, "_normalize_page_indices")

    assert normalize([2, True, 0, False, 2, -1, "3", 1.0]) == [2, 0]


def test_mistral_collect_markdown_uses_ordinal_for_invalid_response_index() -> None:
    markdown = mistral_command._collect_markdown(
        [
            {"markdown": "ordinal page zero"},
            {"index": True, "markdown": "bool index should not map to zero"},
            {"index": -4, "markdown": "negative index should not clamp to zero"},
            {"index": 1, "markdown": "one based page one"},
        ],
        [0],
    )

    assert markdown == "## Page 1\n\nordinal page zero"
