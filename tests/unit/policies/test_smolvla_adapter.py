"""Unit tests for SmolVLAPolicyAdapter (no [vla] extra required)."""

import numpy as np
import pytest

from langgraph_vla_agent.domain.context import EvaluationMode, PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.policies import RobotPolicy, SmolVLAPolicyAdapter, vla_available
from langgraph_vla_agent.policies.smolvla import _StubSmolVLAModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_adapter(action_dim: int = 6) -> tuple[SmolVLAPolicyAdapter, _StubSmolVLAModel]:
    stub = _StubSmolVLAModel(action_dim=action_dim)
    adapter = SmolVLAPolicyAdapter(_model=stub)
    return adapter, stub


def _obs(state_dim: int = 6) -> RobotObservation:
    return RobotObservation(state=np.zeros(state_dim, dtype=np.float32))


def _obs_with_image(state_dim: int = 6, h: int = 4, w: int = 4) -> RobotObservation:
    return RobotObservation(
        state=np.zeros(state_dim, dtype=np.float32),
        images={"front": np.zeros((h, w, 3), dtype=np.uint8)},
    )


def _context() -> PolicyContext:
    return PolicyContext(run_id="r", episode_id="e", evaluation_mode=EvaluationMode.REPLAY)


# ---------------------------------------------------------------------------
# vla_available()
# ---------------------------------------------------------------------------


def test_vla_available_returns_bool() -> None:
    assert isinstance(vla_available(), bool)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_smolvla_adapter_satisfies_robot_policy_protocol() -> None:
    adapter, _ = _stub_adapter()
    assert isinstance(adapter, RobotPolicy)


# ---------------------------------------------------------------------------
# Instantiation guards
# ---------------------------------------------------------------------------


def test_smolvla_adapter_raises_without_vla_and_no_stub() -> None:
    if vla_available():
        pytest.skip("lerobot installed — ImportError path not reachable")
    with pytest.raises(ImportError, match="lerobot"):
        SmolVLAPolicyAdapter()


def test_smolvla_adapter_accepts_stub_without_vla() -> None:
    stub = _StubSmolVLAModel()
    adapter = SmolVLAPolicyAdapter(_model=stub)
    assert adapter is not None


# ---------------------------------------------------------------------------
# reset() — delegates to model
# ---------------------------------------------------------------------------


def test_reset_calls_model_reset() -> None:
    adapter, stub = _stub_adapter()
    assert stub.reset_call_count == 0
    adapter.reset(_context())
    assert stub.reset_call_count == 1


def test_reset_called_multiple_times() -> None:
    adapter, stub = _stub_adapter()
    for _ in range(3):
        adapter.reset(_context())
    assert stub.reset_call_count == 3


# ---------------------------------------------------------------------------
# act() — shape and dtype of returned RobotAction
# ---------------------------------------------------------------------------


def test_act_returns_correct_dim() -> None:
    adapter, _ = _stub_adapter(action_dim=6)
    adapter.reset(_context())
    action = adapter.act(_obs(), "move the block")
    assert action.values.shape == (6,)


def test_act_returns_float32() -> None:
    adapter, _ = _stub_adapter()
    adapter.reset(_context())
    action = adapter.act(_obs(), "move")
    assert action.values.dtype == np.float32


def test_act_with_different_action_dim() -> None:
    adapter, _ = _stub_adapter(action_dim=4)
    adapter.reset(_context())
    action = adapter.act(_obs(state_dim=4), "move")
    assert action.values.shape == (4,)


def test_act_increments_stub_call_count() -> None:
    adapter, stub = _stub_adapter()
    adapter.reset(_context())
    adapter.act(_obs(), "a")
    adapter.act(_obs(), "b")
    assert stub.act_call_count == 2


# ---------------------------------------------------------------------------
# act() — observation with and without image
# ---------------------------------------------------------------------------


def test_act_handles_observation_without_image() -> None:
    adapter, _ = _stub_adapter()
    adapter.reset(_context())
    obs = _obs()  # no images dict
    action = adapter.act(obs, "move the block")
    assert action.values.shape == (6,)


def test_act_handles_observation_with_image() -> None:
    adapter, _ = _stub_adapter()
    adapter.reset(_context())
    obs = _obs_with_image()
    action = adapter.act(obs, "move the block")
    assert action.values.shape == (6,)


# ---------------------------------------------------------------------------
# _to_robot_action — numpy vs tensor-like dispatch
# ---------------------------------------------------------------------------


def test_to_robot_action_with_numpy() -> None:
    adapter, _ = _stub_adapter()
    raw = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    result = adapter._to_robot_action(raw)
    assert result.values.dtype == np.float32
    assert result.values.shape == (3,)


def test_to_robot_action_with_tensor_like() -> None:
    """Uses an object with .cpu() / .detach() / .numpy() (duck-typed tensor)."""

    class _FakeTensor:
        def cpu(self) -> "_FakeTensor":
            return self

        def detach(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:  # type: ignore[override]
            return np.array([1.0, 2.0], dtype=np.float32)

        def flatten(self) -> "_FakeTensor":
            return self

    adapter, _ = _stub_adapter()
    result = adapter._to_robot_action(_FakeTensor())
    assert result.values.shape == (2,)


def test_to_robot_action_raises_on_unexpected_type() -> None:
    adapter, _ = _stub_adapter()
    with pytest.raises(TypeError, match="Unexpected action type"):
        adapter._to_robot_action("not-an-array")


# ---------------------------------------------------------------------------
# model_id property
# ---------------------------------------------------------------------------


def test_model_id_default() -> None:
    adapter, _ = _stub_adapter()
    assert adapter.model_id == "lerobot/smolvla_base"


def test_model_id_custom() -> None:
    stub = _StubSmolVLAModel()
    adapter = SmolVLAPolicyAdapter(model_id="my/custom-model", _model=stub)
    assert adapter.model_id == "my/custom-model"
