"""Retrieval-quality evaluation primitives for library and test use."""

from rag_core.evals.cases import EvalCase, load_cases
from rag_core.evals.metrics import mrr, ndcg_at_k, recall_at_k
from rag_core.evals.quality_gates import add_quality_gate, eval_exit_code
from rag_core.evals.report_models import EvalThresholds
from rag_core.evals.reporting import eval_report, redact_eval_report
from rag_core.evals.runner import EvalResult, run_eval

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalThresholds",
    "add_quality_gate",
    "eval_exit_code",
    "eval_report",
    "load_cases",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "redact_eval_report",
    "run_eval",
]
