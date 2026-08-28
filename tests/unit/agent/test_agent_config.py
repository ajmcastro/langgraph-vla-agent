"""Unit tests for AgentConfig."""

import pytest
from pydantic import ValidationError

from langgraph_vla_agent.agent.config import AgentConfig, Granularity, PlannerType
from langgraph_vla_agent.domain.context import EvaluationMode


def test_default_config_is_valid() -> None:
    cfg = AgentConfig()
    assert cfg.max_retries == 2
    assert cfg.max_replans == 1
    assert cfg.planner_type == PlannerType.DETERMINISTIC
    assert cfg.granularity == Granularity.COARSE
    assert cfg.evaluation_mode == EvaluationMode.MOCK
    assert cfg.safety_check_enabled is True


def test_rejects_negative_max_retries() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(max_retries=-1)


def test_rejects_negative_max_replans() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(max_replans=-1)


def test_zero_retries_is_valid() -> None:
    cfg = AgentConfig(max_retries=0)
    assert cfg.max_retries == 0


def test_zero_replans_is_valid() -> None:
    cfg = AgentConfig(max_replans=0)
    assert cfg.max_replans == 0


def test_fine_granularity_is_accepted() -> None:
    cfg = AgentConfig(granularity=Granularity.FINE)
    assert cfg.granularity == Granularity.FINE


def test_llm_planner_type_is_accepted() -> None:
    cfg = AgentConfig(planner_type=PlannerType.LLM)
    assert cfg.planner_type == PlannerType.LLM


def test_safety_check_can_be_disabled() -> None:
    cfg = AgentConfig(safety_check_enabled=False)
    assert cfg.safety_check_enabled is False
