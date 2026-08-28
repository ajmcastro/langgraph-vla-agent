"""AgentRunner — convenience wrapper around the compiled LangGraph.

Usage (mock mode, no LLM or GPU required):
    runner = make_mock_runner()
    final_state = runner.run("pick up the cube and place it in the bin")
    print(final_state["final_status"])

Usage (custom config):
    config = AgentConfig(granularity=Granularity.FINE, max_retries=3)
    runner = make_mock_runner(config=config)
    final_state = runner.run("grab the block and put it in the tray")
"""

from typing import Protocol, cast

import structlog

from langgraph_vla_agent.agent.config import AgentConfig, Granularity
from langgraph_vla_agent.agent.graph import build_agent_graph
from langgraph_vla_agent.agent.state import AgentState, make_initial_state
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.domain.tasks import TaskGoal
from langgraph_vla_agent.environments.mock import MockEnvironment, MockScenario
from langgraph_vla_agent.execution.config import ExecutorConfig
from langgraph_vla_agent.execution.executor import Executor
from langgraph_vla_agent.planning.deterministic import DeterministicPlanner
from langgraph_vla_agent.policies.mock import MockPolicyBehavior, MockRobotPolicy

_log = structlog.get_logger(__name__)


class _Graph(Protocol):
    def invoke(self, input: AgentState) -> AgentState: ...


class AgentRunner:
    """Wraps a compiled LangGraph for easy single-call episode execution."""

    def __init__(self, graph: _Graph, config: AgentConfig) -> None:
        self._graph = graph
        self._config = config

    def run(self, goal_text: str, run_id: str | None = None) -> AgentState:
        """Execute one full agent episode and return the terminal AgentState.

        Parameters
        ----------
        goal_text:
            Natural-language goal (e.g. "pick up the cube and place it in the bin").
        run_id:
            Optional correlation ID; auto-generated (UUID4) if not provided.

        Returns
        -------
        AgentState
            The final state after the graph reaches END.
            Inspect state["final_status"] for the terminal outcome.
        """
        goal = TaskGoal(text=goal_text, evaluation_mode=self._config.evaluation_mode)
        if run_id is not None:
            goal = TaskGoal(
                text=goal_text,
                run_id=run_id,
                evaluation_mode=self._config.evaluation_mode,
            )

        initial = make_initial_state(goal)
        _log.info("agent_runner.start", run_id=initial["run_id"], goal=goal_text[:120])

        result: AgentState = self._graph.invoke(initial)

        _log.info(
            "agent_runner.done",
            run_id=initial["run_id"],
            final_status=result.get("final_status"),
            completed=len(result.get("completed_subtask_ids", [])),
            failed=len(result.get("failed_subtask_ids", [])),
        )
        return result


def make_mock_runner(
    config: AgentConfig | None = None,
    scenario: MockScenario = MockScenario.SUCCEED_AT_STEP,
    succeed_at_step: int = 2,
) -> AgentRunner:
    """Factory for a fully wired mock runner — no LLM, no GPU, no robot required.

    Uses DeterministicPlanner + MockRobotPolicy + MockEnvironment. Suitable for
    `make evaluate-agent`, tests, and local development without extras beyond [agent].

    Parameters
    ----------
    config:
        Agent configuration; defaults to mock mode with coarse granularity.
    scenario:
        Controls when MockEnvironment signals episode termination.
    succeed_at_step:
        For SUCCEED_AT_STEP scenario: which step number triggers success.
    """
    if config is None:
        config = AgentConfig(
            evaluation_mode=EvaluationMode.MOCK,
            granularity=Granularity.COARSE,
        )

    granularity_str = "fine" if config.granularity.value == "fine" else "coarse"
    planner = DeterministicPlanner(granularity=granularity_str)  # type: ignore[arg-type]
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID)
    environment = MockEnvironment(scenario=scenario, n=succeed_at_step)
    executor_config = ExecutorConfig(
        max_steps=50,
        evaluation_mode=config.evaluation_mode,
    )
    executor = Executor(policy, environment, executor_config)

    graph = cast(
        _Graph,
        build_agent_graph(executor=executor, planner=planner, config=config),
    )
    return AgentRunner(graph=graph, config=config)
