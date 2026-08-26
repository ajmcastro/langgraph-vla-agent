"""Unit tests for domain models.

Verifies: construction, field defaults, validators, and enum membership.
No mocks, no network, no GPU.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from langgraph_vla_agent.domain import (
    EvaluationMode,
    ExecutionResult,
    ExecutionStatus,
    FailureReason,
    PolicyContext,
    RobotAction,
    RobotObservation,
    StepResult,
    SubTask,
)

# ---------------------------------------------------------------------------
# EvaluationMode
# ---------------------------------------------------------------------------


def test_evaluation_mode_values() -> None:
    assert EvaluationMode.MOCK == "mock"
    assert EvaluationMode.REPLAY == "replay"
    assert EvaluationMode.SIMULATION == "simulation"
    assert EvaluationMode.HARDWARE == "hardware"


# ---------------------------------------------------------------------------
# PolicyContext
# ---------------------------------------------------------------------------


def test_policy_context_construction() -> None:
    ctx = PolicyContext(
        run_id="run-001",
        episode_id="ep-001",
        evaluation_mode=EvaluationMode.MOCK,
    )
    assert ctx.run_id == "run-001"
    assert ctx.episode_id == "ep-001"
    assert ctx.seed is None
    assert ctx.extra == {}


def test_policy_context_with_seed() -> None:
    ctx = PolicyContext(
        run_id="r",
        episode_id="e",
        evaluation_mode=EvaluationMode.MOCK,
        seed=42,
    )
    assert ctx.seed == 42


# ---------------------------------------------------------------------------
# RobotObservation
# ---------------------------------------------------------------------------


def test_robot_observation_construction() -> None:
    state = np.zeros(6, dtype=np.float32)
    obs = RobotObservation(state=state, timestamp=1.0)
    assert obs.state.shape == (6,)
    assert obs.images == {}
    assert obs.timestamp == 1.0


def test_robot_observation_rejects_2d_state() -> None:
    with pytest.raises(ValidationError, match="1-D"):
        RobotObservation(state=np.zeros((6, 1)))


def test_robot_observation_rejects_negative_timestamp() -> None:
    with pytest.raises(ValidationError, match=">="):
        RobotObservation(state=np.zeros(6), timestamp=-0.1)


def test_robot_observation_with_image() -> None:
    obs = RobotObservation(
        state=np.zeros(6, dtype=np.float32),
        images={"front": np.zeros((480, 640, 3), dtype=np.uint8)},
    )
    assert "front" in obs.images
    assert obs.images["front"].shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# RobotAction
# ---------------------------------------------------------------------------


def test_robot_action_construction() -> None:
    action = RobotAction(values=np.zeros(6, dtype=np.float32))
    assert action.values.shape == (6,)


def test_robot_action_rejects_2d_values() -> None:
    with pytest.raises(ValidationError, match="1-D"):
        RobotAction(values=np.zeros((6, 1)))


def test_robot_action_allows_nan_values() -> None:
    # The model itself does NOT reject NaN — the Executor's validation gate does.
    action = RobotAction(values=np.array([np.nan, 0.0, 0.0], dtype=np.float32))
    assert not np.all(np.isfinite(action.values))


# ---------------------------------------------------------------------------
# SubTask
# ---------------------------------------------------------------------------


def test_subtask_auto_id() -> None:
    st = SubTask(instruction="grasp the cube")
    assert len(st.id) > 0
    st2 = SubTask(instruction="place in bin")
    assert st.id != st2.id  # UUIDs are unique


def test_subtask_defaults() -> None:
    st = SubTask(instruction="push block left")
    assert st.success_criteria == ""
    assert st.attempt == 0


def test_subtask_explicit_id() -> None:
    st = SubTask(id="my-fixed-id", instruction="test")
    assert st.id == "my-fixed-id"


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


def test_step_result_success() -> None:
    r = StepResult(terminated=True, truncated=False, success=True)
    assert r.terminated
    assert r.success
    assert r.info == {}


def test_step_result_non_terminal() -> None:
    r = StepResult(terminated=False, truncated=False, success=False)
    assert not r.terminated
    assert not r.truncated


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


def test_execution_result_succeeded_property() -> None:
    r = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        failure_reason=FailureReason.NONE,
        steps_taken=5,
        subtask_id="st-1",
        evaluation_mode=EvaluationMode.MOCK,
    )
    assert r.succeeded


def test_execution_result_not_succeeded() -> None:
    r = ExecutionResult(
        status=ExecutionStatus.FAILURE,
        failure_reason=FailureReason.UNKNOWN,
        steps_taken=3,
        subtask_id="st-1",
        evaluation_mode=EvaluationMode.MOCK,
    )
    assert not r.succeeded


def test_execution_result_metrics_default_empty() -> None:
    r = ExecutionResult(
        status=ExecutionStatus.TIMEOUT,
        failure_reason=FailureReason.TIMEOUT,
        steps_taken=0,
        subtask_id="st-x",
        evaluation_mode=EvaluationMode.MOCK,
    )
    assert r.metrics == {}
    assert r.artifact_references == []
