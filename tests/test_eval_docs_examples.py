from __future__ import annotations

from pathlib import Path

from rag_core.evals import load_cases


def test_public_eval_cases_example_is_parseable() -> None:
    cases = load_cases(Path("examples/eval_cases.jsonl"))
    demo_corpus_files = {path.name for path in Path("examples/demo_corpus").glob("*.md")}

    assert [case.case_id for case in cases] == [
        "help/billing-payment-methods",
        "help/corpus-lifecycle",
        "help/security-scope",
    ]
    assert cases[-1].expected_grades == {
        "security.md": 3,
    }
    for case in cases:
        assert set(case.expected_chunk_ids).issubset(demo_corpus_files)
