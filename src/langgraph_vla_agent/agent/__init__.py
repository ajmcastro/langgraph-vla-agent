"""LangGraph orchestration agent.

The public API for M5:
- AgentConfig        — configures the agent (planner, retries, evaluation mode)
- AgentRunner        — convenience wrapper around the compiled graph
- AgentState         — LangGraph TypedDict (the durable orchestration state)
- AgentStatus        — terminal status values for the full agent run
- SafetyChecker      — software-layer safety gate applied before each subtask
- build_agent_graph  — assembles and compiles the StateGraph (requires [agent] extra)
- make_mock_runner   — factory for a fully wired mock runner (no LLM, no GPU)
"""

from langgraph_vla_agent.agent.config import AgentConfig, Granularity, PlannerType
from langgraph_vla_agent.agent.runner import AgentRunner, make_mock_runner
from langgraph_vla_agent.agent.safety import SafetyChecker
from langgraph_vla_agent.agent.state import AgentState, AgentStatus

__all__ = [
    "AgentConfig",
    "AgentRunner",
    "AgentState",
    "AgentStatus",
    "Granularity",
    "PlannerType",
    "SafetyChecker",
    "build_agent_graph",
    "make_mock_runner",
]
