"""Policy implementations.

Public surface for M1:
    RobotPolicy         — structural Protocol
    MockPolicyBehavior  — scenario enum for MockRobotPolicy
    MockRobotPolicy     — deterministic implementation for tests

Public surface for M2:
    ReplayRobotPolicy   — replays recorded actions from a ReplayEpisode

Public surface for M3:
    SmolVLAPolicyAdapter — wraps lerobot/smolvla_base (requires [vla] extra)
    vla_available        — True when lerobot + torch are installed
"""

from langgraph_vla_agent.policies.base import RobotPolicy
from langgraph_vla_agent.policies.mock import MockPolicyBehavior, MockRobotPolicy
from langgraph_vla_agent.policies.replay import ReplayRobotPolicy
from langgraph_vla_agent.policies.smolvla import SmolVLAPolicyAdapter, vla_available

__all__ = [
    "MockPolicyBehavior",
    "MockRobotPolicy",
    "ReplayRobotPolicy",
    "RobotPolicy",
    "SmolVLAPolicyAdapter",
    "vla_available",
]
