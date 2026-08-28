# Architecture

## Guiding principle

The LLM/LangGraph layer and the VLA policy layer must remain **strictly separated**. LangGraph operates at the goal and subtask timescale. The policy executor operates at the observation→action timescale. They communicate only through typed protocol boundaries (`RobotPolicy`, `RobotEnvironment`).

**The LLM must never produce joint torques, servo positions, motor commands, or high-frequency trajectories.** Any design that allows this boundary to dissolve is a regression.

---

## Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│  User / Application                                         │
│  "Pick up the red cube and place it in the bin."            │
└──────────────────────────┬──────────────────────────────────┘
                           │ natural-language goal
┌──────────────────────────▼──────────────────────────────────┐
│  LangGraph Agent  (goal/subtask timescale)                  │
│                                                             │
│  understand_goal → create_plan → select_next_subtask        │
│  → safety_check → execute_policy → verify_result            │
│      ├─ success → select_next_subtask (→ END when all done) │
│      └─ failure → diagnose_failure                          │
│           ├─ retry_count < max_retries → execute_policy     │
│           ├─ replan_count < max_replans → create_plan       │
│           └─ both exhausted → END (FAILED)                  │
│                                                             │
│  State: TaskGoal, TaskPlan, SubTask, ExecutionResult,       │
│         retry_count, replan_count, safety_status            │
└──────────────────────────┬──────────────────────────────────┘
                           │ high-level subtask instruction
┌──────────────────────────▼──────────────────────────────────┐
│  Executor  (observation/action timescale)                   │
│                                                             │
│  for each step:                                             │
│    obs  = environment.observe()                             │
│    act  = policy.act(obs, instruction)                      │
│    result = environment.step(act)                           │
│  until: max_steps | terminal condition                      │
│                                                             │
│  Returns: ExecutionResult (status, metrics, artifact refs)  │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
┌──────────▼──────────┐       ┌───────────▼─────────────────┐
│  RobotPolicy        │       │  RobotEnvironment           │
│  (Protocol)         │       │  (Protocol)                 │
│                     │       │                             │
│  MockRobotPolicy    │       │  MockEnvironment            │
│  ReplayRobotPolicy  │       │  ReplayEnvironment          │
│  SmolVLAAdapter     │       │  [SimulationEnvironment]    │
│  [HardwareAdapter]  │       │  [HardwareEnvironment]      │
└─────────────────────┘       └─────────────────────────────┘
```

---

## LangGraph state

Graph state stores **orchestration metadata only** — never raw image tensors or full trajectory arrays. Large data is referenced by stable IDs (file paths, artifact IDs).

Implemented in `src/langgraph_vla_agent/agent/state.py` as a `TypedDict`. Fields with `Annotated[list[str], operator.add]` use LangGraph's append reducer — nodes return only new items; LangGraph concatenates.

| Field | Type | Notes |
|---|---|---|
| `run_id` | `str` | UUID4 correlation ID for the episode |
| `goal` | `TaskGoal` | Typed goal (text + run_id + evaluation_mode) |
| `plan` | `TaskPlan \| None` | Structured subtask plan; cleared on replan |
| `current_subtask` | `SubTask \| None` | Active subtask |
| `completed_subtask_ids` | `Annotated[list[str], add]` | Appended by `verify_result` on success |
| `failed_subtask_ids` | `Annotated[list[str], add]` | Appended by `diagnose_failure` on replan |
| `execution_history_references` | `Annotated[list[str], add]` | Artifact refs, not raw data |
| `retry_count` | `int` | Attempts on current subtask; reset on success |
| `replan_count` | `int` | Full replanning cycles this episode |
| `last_execution_result` | `ExecutionResult \| None` | Most recent executor output |
| `safety_status` | `str` | `"ok"` or `"rejected"` |
| `safety_rejection_reason` | `str` | Set by `safety_check` on rejection |
| `evaluation_mode` | `EvaluationMode` | mock / replay / simulation / hardware |
| `final_status` | `AgentStatus \| None` | Terminal state when set (COMPLETED / FAILED / SAFETY_STOP) |
| `error_message` | `str` | Human-readable failure description |

`AgentStatus` values: `RUNNING`, `COMPLETED`, `FAILED`, `SAFETY_STOP`.

**What does NOT belong in graph state:** image tensors, trajectory arrays, model weights, raw sensor streams, telemetry blobs. These live in the executor context, artifact store, or observability system.

---

## Planning layer

Two implementations sharing the same `TaskPlan` return type:

- **`DeterministicPlanner`** — scripted plans for tests, baselines, and reproducibility. No LLM calls. The default in all non-integration tests.
- **`LLMTaskPlanner`** — validates the LLM output against a Pydantic schema before it enters the graph. Provider-agnostic via `langchain-core`.

Plans are never parsed from free-form prose. Invalid plan output triggers a controlled error node, not an unhandled exception.

---

## Policy abstraction

```python
class RobotPolicy(Protocol):
    def reset(self, context: PolicyContext) -> None: ...
    def act(self, observation: RobotObservation, instruction: str) -> PolicyAction: ...
```

Implementations (introduced per milestone):

| Class | Deps | Purpose |
|---|---|---|
| `MockRobotPolicy` | none | Deterministic scripted behavior for agent tests |
| `ReplayRobotPolicy` | none (local files) | Replays recorded actions for offline evaluation |
| `SmolVLAPolicyAdapter` | `lerobot[smolvla,dataset]`, GPU | Actual SmolVLA inference |

Hardware adapters are isolated in `src/langgraph_vla_agent/policies/hardware/` and are never imported by core agent or evaluation code.

---

## Environment abstraction

```python
class RobotEnvironment(Protocol):
    def reset(self, scenario: Scenario) -> RobotObservation: ...
    def step(self, action: RobotAction) -> tuple[RobotObservation, StepResult]: ...
    def observe(self) -> RobotObservation: ...
```

---

## Dependency injection

External dependencies (LLM client, policy, environment, artifact store) are injected at the graph's entry point. Tests substitute deterministic implementations without any mocking framework. This makes all agent behavior testable without GPU, network, or robot.

---

## Configuration

All runtime configuration comes from environment variables and YAML config files in `configs/`. No hardcoded credentials, endpoints, or seeds. Reproducible experiments record the full config alongside results.
