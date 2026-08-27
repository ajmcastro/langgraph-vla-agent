"""Evaluation infrastructure for langgraph-vla-agent.

Public surface for M3:
    ActionErrorMetrics  — aggregate action prediction error statistics
    EpisodeEvalResult   — per-episode errors and step count
    OfflineEvalResult   — full offline evaluation result with provenance
    OfflineEvaluator    — runs a policy against recorded episodes, computes errors
"""

from langgraph_vla_agent.evaluation.metrics import (
    ActionErrorMetrics,
    EpisodeEvalResult,
    OfflineEvalResult,
)
from langgraph_vla_agent.evaluation.offline import OfflineEvaluator

__all__ = [
    "ActionErrorMetrics",
    "EpisodeEvalResult",
    "OfflineEvalResult",
    "OfflineEvaluator",
]
