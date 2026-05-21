"""Retrieval-quality evaluation surface for the engine.

The eval module *consumes* :class:`rag_core.RAGCore` and never reaches into
internals. It provides a small metrics module (``recall@k`` / ``MRR`` /
``nDCG@k``), a runner that turns a list of :class:`EvalCase`s into
per-case :class:`EvalResult`s by driving ``core.search``, and report builders
for JSON/CLI quality gates.
"""

from rag_core.evals.metrics import mrr, ndcg_at_k, recall_at_k
from rag_core.evals.reporting import (
    EvalReport,
    EvalRunMetadata,
    EvalThresholds,
    add_quality_gate,
    eval_comparison_report,
    eval_exit_code,
    eval_profile_comparison_report,
    eval_report,
    eval_result_payload,
    format_eval_comparison_report,
    format_eval_metrics,
    format_eval_profile_comparison_report,
    format_eval_report,
    format_run_metadata,
    redact_eval_report,
)
from rag_core.evals.runner import EvalCase, EvalResult, load_cases, run_eval

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalRunMetadata",
    "EvalResult",
    "EvalThresholds",
    "add_quality_gate",
    "eval_comparison_report",
    "eval_exit_code",
    "eval_profile_comparison_report",
    "eval_report",
    "eval_result_payload",
    "format_eval_comparison_report",
    "format_eval_metrics",
    "format_eval_profile_comparison_report",
    "format_eval_report",
    "format_run_metadata",
    "load_cases",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "redact_eval_report",
    "run_eval",
]
