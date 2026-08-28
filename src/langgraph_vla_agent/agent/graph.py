"""build_agent_graph — assembles and compiles the LangGraph StateGraph.

This is the only module in the agent package that imports from langgraph.
All other modules (nodes, state, safety, config) are importable without
the [agent] extra, keeping unit tests fast and dependency-free.

Dependency injection
--------------------
build_agent_graph() accepts the executor, planner, config, and safety
checker as plain Python objects. Each LangGraph node that needs one
receives it via a closure, keeping the node functions in nodes.py
pure and independently testable.
"""

from langgraph_vla_agent.agent.config import AgentConfig
from langgraph_vla_agent.agent.nodes import (
    create_plan,
    diagnose_failure,
    execute_policy,
    route_after_create_plan,
    route_after_diagnose,
    route_after_safety,
    route_after_select,
    route_after_understand,
    route_after_verify,
    safety_check,
    select_next_subtask,
    understand_goal,
    verify_result,
)
from langgraph_vla_agent.agent.safety import SafetyChecker
from langgraph_vla_agent.agent.state import AgentState
from langgraph_vla_agent.execution.executor import Executor
from langgraph_vla_agent.planning.base import TaskPlanner

try:
    from langgraph.graph import START, StateGraph
except ImportError as _exc:
    raise ImportError("LangGraph is not installed. Run: uv sync --extra agent") from _exc


def build_agent_graph(
    executor: Executor,
    planner: TaskPlanner,
    config: AgentConfig,
    safety_checker: SafetyChecker | None = None,
) -> object:
    """Assemble and compile the VLA orchestration graph.

    Parameters
    ----------
    executor:
        Wired with the desired policy and environment.
    planner:
        DeterministicPlanner or LLMTaskPlanner.
    config:
        Controls max_retries, max_replans, safety, and evaluation_mode.
    safety_checker:
        Custom SafetyChecker; defaults to SafetyChecker() with standard lists.

    Returns
    -------
    CompiledStateGraph
        Call `.invoke(initial_state)` to run one full episode.

    Graph topology
    --------------
    START
      → understand_goal
      → create_plan         [→ END on planning failure]
      → select_next_subtask [→ END on completion or no plan]
      → safety_check        [→ END on safety rejection]
      → execute_policy
      → verify_result
          ├─ success → select_next_subtask
          └─ failure → diagnose_failure
              ├─ retry  → execute_policy
              ├─ replan → create_plan
              └─ give_up → END
    """
    checker = safety_checker or SafetyChecker()

    # --- Closures that inject dependencies into node functions. ---
    def _create_plan(state: AgentState) -> dict:  # type: ignore[type-arg]
        return create_plan(state, planner=planner)

    def _safety_check(state: AgentState) -> dict:  # type: ignore[type-arg]
        if not config.safety_check_enabled:
            return {"safety_status": "ok", "safety_rejection_reason": ""}
        return safety_check(state, checker=checker)

    def _execute_policy(state: AgentState) -> dict:  # type: ignore[type-arg]
        return execute_policy(state, executor=executor)

    def _diagnose_failure(state: AgentState) -> dict:  # type: ignore[type-arg]
        return diagnose_failure(state, config=config)

    # --- Build graph. ---
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("understand_goal", understand_goal)
    graph.add_node("create_plan", _create_plan)
    graph.add_node("select_next_subtask", select_next_subtask)
    graph.add_node("safety_check", _safety_check)
    graph.add_node("execute_policy", _execute_policy)
    graph.add_node("verify_result", verify_result)
    graph.add_node("diagnose_failure", _diagnose_failure)

    graph.add_edge(START, "understand_goal")
    graph.add_conditional_edges("understand_goal", route_after_understand)
    graph.add_conditional_edges("create_plan", route_after_create_plan)
    graph.add_conditional_edges("select_next_subtask", route_after_select)
    graph.add_conditional_edges("safety_check", route_after_safety)
    graph.add_edge("execute_policy", "verify_result")
    graph.add_conditional_edges("verify_result", route_after_verify)
    graph.add_conditional_edges("diagnose_failure", route_after_diagnose)

    return graph.compile()
