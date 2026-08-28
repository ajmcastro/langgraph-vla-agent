"""Evaluation infrastructure for langgraph-vla-agent.

Public surface:
    ActionErrorMetrics          — aggregate action prediction error statistics
    EpisodeEvalResult           — per-episode errors and step count
    OfflineEvalResult           — full offline evaluation result with provenance
    OfflineEvaluator            — runs a policy against recorded episodes
    CheckpointComparisonResult  — side-by-side comparison of two checkpoints
    compare_checkpoints         — evaluate two policies and compute delta metrics
    ConditionResult             — result of one planning condition on one episode
    ConditionSummary            — aggregate metrics for one condition
    EpisodeScenario             — goal + mock environment settings for M6
    GranularityExperimentResult — full M6 three-condition experiment result
    run_granularity_experiment  — run the M6 planning-granularity experiment
"""

from langgraph_vla_agent.evaluation.comparison import (
    CheckpointComparisonResult,
    compare_checkpoints,
)
from langgraph_vla_agent.evaluation.experiment import (
    ConditionResult,
    ConditionSummary,
    EpisodeScenario,
    GranularityExperimentResult,
    run_granularity_experiment,
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
    "ConditionResult",
    "ConditionSummary",
    "EpisodeEvalResult",
    "EpisodeScenario",
    "GranularityExperimentResult",
    "OfflineEvalResult",
    "OfflineEvaluator",
    "compare_checkpoints",
    "run_granularity_experiment",
]
