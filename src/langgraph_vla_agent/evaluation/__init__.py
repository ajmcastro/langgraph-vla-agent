"""Evaluation infrastructure for langgraph-vla-agent.

Public surface:
    ActionErrorMetrics          — aggregate action prediction error statistics
    EpisodeEvalResult           — per-episode errors and step count
    OfflineEvalResult           — full offline evaluation result with provenance
    OfflineEvaluator            — runs a policy against recorded episodes
    CheckpointComparisonResult  — side-by-side comparison of two checkpoints
    compare_checkpoints         — evaluate two policies and compute delta metrics
"""

from langgraph_vla_agent.evaluation.comparison import (
    CheckpointComparisonResult,
    compare_checkpoints,
)
from langgraph_vla_agent.evaluation.metrics import (
    ActionErrorMetrics,
    EpisodeEvalResult,
    OfflineEvalResult,
)
from langgraph_vla_agent.evaluation.offline import OfflineEvaluator

__all__ = [
    "ActionErrorMetrics",
    "CheckpointComparisonResult",
    "EpisodeEvalResult",
    "OfflineEvalResult",
    "OfflineEvaluator",
    "compare_checkpoints",
]
