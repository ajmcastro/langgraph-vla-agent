"""ExecutorConfig — runtime limits for the observation→action loop."""

from pydantic import BaseModel, Field

from langgraph_vla_agent.domain.context import EvaluationMode


class ExecutorConfig(BaseModel):
    """Configures the Executor's loop limits and safety gates.

    All limits are enforced as hard upper bounds. Exceeding any of them
    transitions to a terminal failure state rather than silently continuing.
    """

    max_steps: int = Field(default=200, gt=0)
    """Maximum action steps per subtask. Matches docs/safety.md default."""

    timeout_s: float = Field(default=60.0, gt=0.0)
    """Wall-clock seconds before the executor forces a TIMEOUT terminal state."""

    validate_actions: bool = True
    """When True, reject non-finite or wrong-shape actions before they reach
    the environment. Disable only for profiling or replay where validation
    is already guaranteed upstream."""

    evaluation_mode: EvaluationMode = EvaluationMode.MOCK
    """Propagated into every ExecutionResult for audit / evaluation labelling."""
