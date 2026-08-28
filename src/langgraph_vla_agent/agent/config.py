"""AgentConfig — orchestration-level configuration for the LangGraph agent."""

from enum import StrEnum

from pydantic import BaseModel, Field

from langgraph_vla_agent.domain.context import EvaluationMode


class PlannerType(StrEnum):
    """Which planner the agent uses to decompose goals into subtasks."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"


class Granularity(StrEnum):
    """Subtask decomposition granularity.

    Used by DeterministicPlanner and as a label in M6 experiments.
    VLA_ONLY is a marker for the no-orchestration baseline (M6).
    """

    COARSE = "coarse"
    FINE = "fine"
    VLA_ONLY = "vla_only"


class AgentConfig(BaseModel):
    """Configuration for one agent run.

    Fields
    ------
    max_retries:
        Maximum retry attempts per subtask before giving up or replanning.
    max_replans:
        Maximum full replanning cycles per episode.
    planner_type:
        Which planner to use. DETERMINISTIC requires no LLM or network.
    granularity:
        Decomposition granularity passed to DeterministicPlanner.
    evaluation_mode:
        Propagated to every ExecutionResult for audit labelling.
    safety_check_enabled:
        When False, the safety gate is bypassed (useful for controlled experiments).
    """

    max_retries: int = Field(default=2, ge=0)
    max_replans: int = Field(default=1, ge=0)
    planner_type: PlannerType = PlannerType.DETERMINISTIC
    granularity: Granularity = Granularity.COARSE
    evaluation_mode: EvaluationMode = EvaluationMode.MOCK
    safety_check_enabled: bool = True
