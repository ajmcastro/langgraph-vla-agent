"""Integration tests for SmolVLAPolicyAdapter with the real lerobot model.

These tests are skipped unless the [vla] extra is installed AND the
SMOLVLA_INTEGRATION_TESTS environment variable is set to "1".

They require:
  - uv sync --extra dev --extra vla
  - internet access to download the model checkpoint on first run
  - SMOLVLA_INTEGRATION_TESTS=1 (to avoid accidental Hub downloads in CI)

Run manually:
    SMOLVLA_INTEGRATION_TESTS=1 uv run pytest tests/integration/policies/ -v
"""

import os

import numpy as np
import pytest

from langgraph_vla_agent.domain.context import EvaluationMode, PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.policies import SmolVLAPolicyAdapter, vla_available

_RUN_INTEGRATION = os.getenv("SMOLVLA_INTEGRATION_TESTS") == "1"
_skip_reason = (
    "Set SMOLVLA_INTEGRATION_TESTS=1 and install [vla] extra to run SmolVLA integration tests"
)

pytestmark = pytest.mark.skipif(
    not _RUN_INTEGRATION or not vla_available(),
    reason=_skip_reason,
)


# ---------------------------------------------------------------------------
# Fixtures — module-scoped so the model is loaded once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter() -> SmolVLAPolicyAdapter:
    return SmolVLAPolicyAdapter(
        model_id="lerobot/smolvla_base",
        device="cpu",
        image_key="front",
    )


@pytest.fixture
def context() -> PolicyContext:
    return PolicyContext(
        run_id="integration-test",
        episode_id="ep-smolvla-001",
        evaluation_mode=EvaluationMode.REPLAY,
    )


@pytest.fixture
def obs() -> RobotObservation:
    return RobotObservation(
        state=np.zeros(6, dtype=np.float32),
        images={"front": np.zeros((480, 640, 3), dtype=np.uint8)},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_loads_without_error(adapter: SmolVLAPolicyAdapter) -> None:
    assert adapter is not None


def test_reset_runs_without_error(adapter: SmolVLAPolicyAdapter, context: PolicyContext) -> None:
    adapter.reset(context)


def test_act_returns_6d_action(
    adapter: SmolVLAPolicyAdapter,
    context: PolicyContext,
    obs: RobotObservation,
) -> None:
    adapter.reset(context)
    action = adapter.act(obs, "pick up the red cube and place it in the bin")
    assert action.values.shape == (6,)
    assert action.values.dtype == np.float32


def test_act_returns_finite_values(
    adapter: SmolVLAPolicyAdapter,
    context: PolicyContext,
    obs: RobotObservation,
) -> None:
    adapter.reset(context)
    action = adapter.act(obs, "pick up the cube")
    assert np.all(np.isfinite(action.values)), "SmolVLA returned non-finite action values"


def test_chunk_buffer_across_calls(
    adapter: SmolVLAPolicyAdapter,
    context: PolicyContext,
    obs: RobotObservation,
) -> None:
    """SmolVLA predicts an action chunk; verify multiple calls return valid actions."""
    adapter.reset(context)
    for _ in range(5):
        action = adapter.act(obs, "pick up the block")
        assert action.values.shape == (6,)
