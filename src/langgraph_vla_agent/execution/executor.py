"""Executor — observation→policy→action→environment loop."""

import time

import numpy as np
import structlog

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.results import (
    ExecutionResult,
    ExecutionStatus,
    FailureReason,
)
from langgraph_vla_agent.domain.tasks import SubTask
from langgraph_vla_agent.environments.base import RobotEnvironment
from langgraph_vla_agent.execution.config import ExecutorConfig
from langgraph_vla_agent.policies.base import RobotPolicy

_log = structlog.get_logger(__name__)


def _is_action_valid(action: RobotAction) -> bool:
    """Return True iff the action vector is 1-D and fully finite."""
    return bool(action.values.ndim == 1 and np.all(np.isfinite(action.values)))


def _make_result(
    *,
    status: ExecutionStatus,
    failure_reason: FailureReason,
    steps_taken: int,
    subtask: SubTask,
    config: ExecutorConfig,
    elapsed_s: float,
) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        failure_reason=failure_reason,
        steps_taken=steps_taken,
        subtask_id=subtask.id,
        evaluation_mode=config.evaluation_mode,
        metrics={"elapsed_s": round(elapsed_s, 4)},
    )


class Executor:
    """Runs the sensorimotor loop for one subtask episode.

    Responsibilities
    ----------------
    - Call policy.reset() once at episode start.
    - Loop: observe → policy.act() → validate → environment.step().
    - Enforce max_steps, wall-clock timeout, and action-validity gates.
    - Catch policy and environment exceptions and map them to terminal states.
    - Return a single ExecutionResult; never expose individual steps to the caller.

    The caller (LangGraph node in M5) operates at the subtask timescale and
    sees only the ExecutionResult — it never sees intermediate observations
    or individual actions.

    Parameters
    ----------
    policy:
        Any object satisfying the RobotPolicy Protocol.
    environment:
        Any object satisfying the RobotEnvironment Protocol.
    config:
        Loop limits and safety-gate settings. Defaults are from docs/safety.md.
    """

    def __init__(
        self,
        policy: RobotPolicy,
        environment: RobotEnvironment,
        config: ExecutorConfig | None = None,
    ) -> None:
        self._policy = policy
        self._environment = environment
        self._config = config or ExecutorConfig()

    def run(self, subtask: SubTask, context: PolicyContext) -> ExecutionResult:
        """Execute one subtask episode and return the terminal result.

        Parameters
        ----------
        subtask:
            The high-level instruction and metadata for this episode.
        context:
            Per-episode metadata (run_id, episode_id, seed) for logging and
            reproducibility. Passed verbatim to policy.reset().

        Returns
        -------
        ExecutionResult
            Terminal state with status, failure reason, step count, and metrics.
            Always returned — exceptions are caught and mapped to failure states.
        """
        log = _log.bind(
            run_id=context.run_id,
            episode_id=context.episode_id,
            subtask_id=subtask.id,
            evaluation_mode=self._config.evaluation_mode.value,
            max_steps=self._config.max_steps,
        )
        log.info("executor.start", instruction=subtask.instruction)

        start_time = time.monotonic()

        self._policy.reset(context)
        obs = self._environment.reset(subtask)

        for step in range(self._config.max_steps):
            elapsed = time.monotonic() - start_time

            if elapsed > self._config.timeout_s:
                log.warning("executor.timeout", step=step, elapsed_s=round(elapsed, 3))
                return _make_result(
                    status=ExecutionStatus.TIMEOUT,
                    failure_reason=FailureReason.TIMEOUT,
                    steps_taken=step,
                    subtask=subtask,
                    config=self._config,
                    elapsed_s=elapsed,
                )

            # --- policy ---
            try:
                action = self._policy.act(obs, subtask.instruction)
            except Exception:
                elapsed = time.monotonic() - start_time
                log.exception("executor.policy_error", step=step)
                return _make_result(
                    status=ExecutionStatus.FAILURE,
                    failure_reason=FailureReason.POLICY_ERROR,
                    steps_taken=step,
                    subtask=subtask,
                    config=self._config,
                    elapsed_s=elapsed,
                )

            # --- action validation ---
            if self._config.validate_actions and not _is_action_valid(action):
                elapsed = time.monotonic() - start_time
                log.warning("executor.invalid_action", step=step, values=action.values.tolist())
                return _make_result(
                    status=ExecutionStatus.INVALID_ACTION,
                    failure_reason=FailureReason.INVALID_ACTION,
                    steps_taken=step + 1,
                    subtask=subtask,
                    config=self._config,
                    elapsed_s=elapsed,
                )

            # --- environment step ---
            try:
                obs, step_result = self._environment.step(action)
            except Exception:
                elapsed = time.monotonic() - start_time
                log.exception("executor.environment_error", step=step)
                return _make_result(
                    status=ExecutionStatus.ENVIRONMENT_ERROR,
                    failure_reason=FailureReason.ENVIRONMENT_ERROR,
                    steps_taken=step + 1,
                    subtask=subtask,
                    config=self._config,
                    elapsed_s=elapsed,
                )

            # --- terminal check ---
            if step_result.terminated:
                elapsed = time.monotonic() - start_time
                if step_result.success:
                    log.info("executor.success", steps=step + 1, elapsed_s=round(elapsed, 3))
                    return _make_result(
                        status=ExecutionStatus.SUCCESS,
                        failure_reason=FailureReason.NONE,
                        steps_taken=step + 1,
                        subtask=subtask,
                        config=self._config,
                        elapsed_s=elapsed,
                    )
                else:
                    log.info("executor.failure", steps=step + 1, elapsed_s=round(elapsed, 3))
                    return _make_result(
                        status=ExecutionStatus.FAILURE,
                        failure_reason=FailureReason.UNKNOWN,
                        steps_taken=step + 1,
                        subtask=subtask,
                        config=self._config,
                        elapsed_s=elapsed,
                    )

            if step_result.truncated:
                elapsed = time.monotonic() - start_time
                log.info("executor.truncated", steps=step + 1)
                return _make_result(
                    status=ExecutionStatus.MAX_STEPS_EXCEEDED,
                    failure_reason=FailureReason.MAX_STEPS_EXCEEDED,
                    steps_taken=step + 1,
                    subtask=subtask,
                    config=self._config,
                    elapsed_s=elapsed,
                )

        # --- exhausted max_steps without a terminal signal ---
        elapsed = time.monotonic() - start_time
        log.warning("executor.max_steps_exceeded", steps=self._config.max_steps)
        return _make_result(
            status=ExecutionStatus.MAX_STEPS_EXCEEDED,
            failure_reason=FailureReason.MAX_STEPS_EXCEEDED,
            steps_taken=self._config.max_steps,
            subtask=subtask,
            config=self._config,
            elapsed_s=elapsed,
        )
