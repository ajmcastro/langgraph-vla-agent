"""SimulationEnvironment — toy closed-loop physics for Milestone 7.

Unlike MockEnvironment (scripted) and ReplayEnvironment (recorded),
this environment is genuinely closed-loop: the action taken at each step
actually changes the world state, which in turn affects the next observation
and the success predicate.  That is the key property M7 adds.

Progress model
--------------
    progress_{t+1} = clip(progress_t + delta, 0, 1)
    delta           = progress_per_step * action_contribution + noise
    action_contribution = (clip(mean(action.values), -1, 1) + 1) / 2  in [0, 1]
    noise           ~ N(0, noise_scale)   if noise_scale > 0

With MockRobotPolicy (all-zero actions): action_contribution = 0.5, so
    delta = 0.5 * progress_per_step per step (baseline rate).

Success predicate: progress >= success_threshold, which fires terminated=True.

The observation encodes current progress in state[0] so that a future policy
that reads observations could exploit it as a closed-loop signal (MockRobotPolicy
ignores observations; SmolVLA could theoretically use it in future work).

Calibration guide
-----------------
Steps needed to succeed with zero actions (MockRobotPolicy):
    n = ceil(success_threshold / (0.5 * progress_per_step))

Example:
    threshold=0.5, progress_per_step=0.15  →  n = ceil(0.5/0.075) ≈ 7 steps
    threshold=0.25, progress_per_step=0.15 →  n = ceil(0.25/0.075) ≈ 4 steps
    threshold=0.1, progress_per_step=0.15  →  n = ceil(0.1/0.075) ≈ 2 steps
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.domain.results import StepResult
from langgraph_vla_agent.domain.tasks import SubTask


class SimulationScenario(BaseModel):
    """Parameters for one SimulationEnvironment configuration.

    Fields
    ------
    total_progress:
        Total progress the task requires across all subtasks.  Each subtask
        receives a threshold of total_progress / n_subtasks, distributed
        equally.  This keeps total task difficulty constant across conditions.
    progress_per_step:
        Maximum progress increment per step (achieved when action_contribution=1).
        With MockRobotPolicy (zero actions), the actual increment is half this.
    max_steps_per_subtask:
        Per-subtask action budget passed to the Executor.
    noise_scale:
        Standard deviation of Gaussian noise added to each progress delta.
        0.0 means deterministic; >0 makes episodes stochastic.
    seed:
        Base RNG seed.  Each subtask adds its own index so subtasks are
        independently seeded while the overall experiment is reproducible.
    state_dim:
        Dimension of the proprioceptive state vector in returned observations.
        state[0] always encodes current progress; remaining dims are zeros.
    """

    total_progress: float = Field(default=0.5, gt=0.0, le=1.0)
    progress_per_step: float = Field(default=0.15, gt=0.0)
    max_steps_per_subtask: int = Field(default=5, gt=0)
    noise_scale: float = Field(default=0.0, ge=0.0)
    seed: int = 42
    state_dim: int = Field(default=6, ge=1)


class SimulationEnvironment:
    """Closed-loop toy physics environment.

    Satisfies the RobotEnvironment Protocol.  No external simulator, GPU,
    or dataset is required — the world model is a scalar progress variable
    updated by the policy's actions.

    Parameters
    ----------
    success_threshold:
        Progress value at which the subtask is considered complete.
        Typically set to scenario.total_progress / n_subtasks by the
        experiment runner.
    progress_per_step:
        Maximum progress increment per step.
    noise_scale:
        Gaussian noise std added to each step's delta (0.0 = deterministic).
    seed:
        RNG seed.  A fixed seed gives deterministic results for noise_scale=0.
    state_dim:
        Observation state vector dimension.
    """

    def __init__(
        self,
        success_threshold: float = 0.5,
        progress_per_step: float = 0.15,
        noise_scale: float = 0.0,
        seed: int = 42,
        state_dim: int = 6,
    ) -> None:
        self._success_threshold = success_threshold
        self._progress_per_step = progress_per_step
        self._noise_scale = noise_scale
        self._base_seed = seed
        self._state_dim = state_dim

        self._progress: float = 0.0
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._current_subtask: SubTask | None = None

    # ------------------------------------------------------------------
    # RobotEnvironment Protocol
    # ------------------------------------------------------------------

    def reset(self, subtask: SubTask) -> RobotObservation:
        """Reset progress to 0 and return the initial observation."""
        self._progress = 0.0
        self._step_count = 0
        self._current_subtask = subtask
        # Re-seed per subtask so repeated resets are reproducible
        self._rng = np.random.default_rng(self._base_seed + hash(subtask.id) % (2**32))
        return self._make_observation()

    def step(self, action: RobotAction) -> tuple[RobotObservation, StepResult]:
        """Apply action: update progress and return next observation + result.

        The action is NOT ignored — its mean value scales how much progress
        is made this step.  This is the fundamental difference from
        MockEnvironment and ReplayEnvironment.
        """
        self._step_count += 1

        # Map mean action value from [-1,1] to action_contribution in [0,1]
        raw_mean = float(np.mean(action.values))
        clipped = float(np.clip(raw_mean, -1.0, 1.0))
        action_contribution = (clipped + 1.0) / 2.0  # [0, 1]

        delta = self._progress_per_step * action_contribution
        if self._noise_scale > 0.0:
            delta += float(self._rng.normal(0.0, self._noise_scale))

        self._progress = float(np.clip(self._progress + delta, 0.0, 1.0))

        terminated = self._progress >= self._success_threshold
        obs = self._make_observation()
        result = StepResult(
            terminated=terminated,
            truncated=False,
            success=terminated,
            info={
                "progress": round(self._progress, 4),
                "threshold": self._success_threshold,
                "step": self._step_count,
                "action_contribution": round(action_contribution, 4),
            },
        )
        return obs, result

    def observe(self) -> RobotObservation:
        """Return current observation without advancing world state."""
        return self._make_observation()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def progress(self) -> float:
        """Current progress value in [0, 1]."""
        return self._progress

    @property
    def success_threshold(self) -> float:
        """The progress threshold that triggers success."""
        return self._success_threshold

    @property
    def step_count(self) -> int:
        """Number of step() calls since last reset()."""
        return self._step_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> RobotObservation:
        state = np.zeros(self._state_dim, dtype=np.float32)
        state[0] = float(self._progress)  # progress encoded in first dim
        return RobotObservation(
            state=state,
            images={},
            timestamp=float(self._step_count),
        )
