# LangGraph VLA Agent — Project Implementation Prompt

I want you to act as a **Staff-level Machine Learning Engineer, Robotics/Physical AI Engineer, and Agentic AI Architect** and help me build an educational but production-quality open-source project called **LangGraph VLA Agent**.

The project must demonstrate how an agentic planning layer can orchestrate a Vision-Language-Action (VLA) policy while keeping high-level reasoning separate from low-level sensorimotor control.

I do **not** own a LeRobot/SO-101 physical arm and I am skipping the hardware-focused Phase 2 of my learning plan. Therefore:

- physical hardware must not be required for any core milestone, test, demo, or evaluation;
- the primary path must use public LeRobot-compatible datasets, offline/replay evaluation, deterministic mocks, simulation where practical, and cloud NVIDIA GPUs for fine-tuning;
- real-robot execution, teleoperation, data collection, calibration, and SO-101 integration are optional future work only;
- never invent or imply real-world results when experiments were performed offline, in replay, or in simulation.

The final repository should be strong enough to serve as a **Staff/Senior-level Agentic AI + Robotics portfolio project**. I want to understand the implementation, not merely receive generated code. Work incrementally, explain important decisions, and avoid unnecessary complexity.

---

## Central architecture

```text
Natural-language goal
        ↓
Agentic reasoning / planning
        ↓
LangGraph orchestration
        ↓
High-level physical subtask
        ↓
RobotPolicy abstraction
        ↓
SmolVLA / replay / mock policy
        ↓
Dataset replay, mock environment, or simulator
        ↓
Observation + execution result
        ↓
Verification and replanning
```

LangGraph is the brain responsible for goal interpretation, decomposition, orchestration, state tracking, verification, and recovery. The VLA is the sensorimotor policy that maps observations and language instructions to actions.

Do **not** turn the LLM into a low-level controller. It must never generate servo positions, joint torques, motor commands, or high-frequency trajectories. LangGraph operates at the goal/subtask timescale; the policy executor operates at the repeated observation/action timescale.

---

## Core research question

> When does adding an agentic planning and orchestration layer on top of a learned VLA policy improve multi-step task completion compared with giving the entire natural-language task directly to the VLA?

Evaluate possible gains in:

- success rate and robustness;
- language and scene generalisation;
- failure recovery;
- interpretability and observability.

Also measure costs in:

- latency and execution steps;
- LLM calls and policy invocations;
- implementation and operational complexity.

The comparison must be scientifically honest about what offline/replay evaluation can and cannot establish. Offline action-prediction metrics are not equivalent to closed-loop task success.

---

## Primary development and execution environments

### Local development

Assume a MacBook Pro M4 Max with 64 GB unified memory. Use it for:

- LangGraph and API development;
- dataset inspection and preprocessing;
- unit and integration tests;
- deterministic mock and replay execution;
- lightweight inference when dependencies support Apple Silicon;
- evaluation analysis and plots.

### Cloud GPU environment

Use provider-neutral NVIDIA GPU infrastructure for:

- SmolVLA fine-tuning;
- GPU-intensive inference;
- batch evaluation;
- checkpoint export and artifact generation.

Training jobs must be reproducible from committed configuration. Do not couple the repository to one cloud vendor, and do not commit credentials or large model/data artifacts.

### Optional future physical environment

An SO-101 or other LeRobot-compatible robot may be added later behind the existing interfaces. This is **not** part of the required implementation or acceptance criteria. Keep future hardware adapters isolated, and mark hardware-in-the-loop tests separately so normal CI never requires a robot.

---

## Default VLA and abstractions

Use **Hugging Face LeRobot + SmolVLA** as the default real VLA implementation, subject to current compatibility and checkpoint/dataset availability.

The architecture must not depend directly on SmolVLA. Start with the smallest useful abstraction, approximately:

```python
class RobotPolicy(Protocol):
    def reset(self, context: PolicyContext) -> None: ...

    def act(
        self,
        observation: RobotObservation,
        instruction: str,
    ) -> PolicyAction: ...
```

Adapt the exact API to LeRobot rather than forcing an artificial interface. Expected implementations, introduced only when needed, are:

- `MockRobotPolicy` for deterministic graph and recovery tests;
- `ReplayRobotPolicy` for recorded trajectories and offline scenarios;
- `SmolVLAPolicyAdapter` for actual model inference.

Other VLA adapters are optional future extensions. The repository must remain fully usable when a VLA checkpoint, CUDA GPU, simulator, or network connection is unavailable.

Create a separate environment boundary, such as `RobotEnvironment`, with mock, replay, and optional simulation implementations. Do not put simulator-specific or future hardware-specific behavior into domain or agent code.

---

## Data strategy

Use public, documented, LeRobot-compatible manipulation datasets as the primary data source. Before selecting a dataset, verify its license, schema, embodiment/action space, observation modalities, language annotations, size, and compatibility with the chosen SmolVLA/LeRobot versions.

Do not assume an SO-101 dataset is automatically compatible merely because it is hosted in the LeRobot ecosystem. Record any conversions or limitations explicitly.

The data workflow must support:

1. metadata-only inspection or a tiny sample;
2. validation of episode boundaries, timestamps, observations, actions, and language annotations;
3. deterministic train/validation/test or evaluation splits with leakage checks;
4. cached/local references without committing large datasets;
5. replay scenarios derived from held-out episodes;
6. provenance, license, version/revision, checksums where feasible, and preprocessing configuration.

Never require locally collected demonstrations. A future personal dataset may be supported, but it is not the default path.

---

## Evaluation modes and claims

Keep these modes distinct in code, reports, and metrics:

### Mock evaluation

Deterministic scripted scenarios for validating plans, graph routing, retries, safety decisions, and terminal states. This proves software behavior, not robot capability.

### Offline/replay evaluation

Evaluate model predictions or orchestration decisions against held-out recorded episodes. Appropriate metrics may include action error, action-chunk error, likelihood-based metrics when supported, instruction-conditioned performance, retrieval/trajectory agreement, and replay-defined subtask outcomes.

State clearly that logged-data evaluation cannot fully measure counterfactual closed-loop behavior.

### Simulation evaluation

Use a mock or a practical simulator such as MuJoCo only if it adds credible closed-loop evidence without dominating the project. Report simulator, task definitions, seeds, resets, success predicates, and limitations. Do not claim sim-to-real performance.

### Optional future hardware evaluation

Document a proposed protocol, safety checklist, and adapter boundary. Do not fabricate measurements.

Every result must identify its evaluation mode.

---

## Python and dependency management

Use Python and use **`uv` for all Python version, virtual-environment, and package management**.

Do not use Poetry, Pipenv, Conda, `requirements.txt` as the primary dependency mechanism, or direct pip-based project management. Maintain dependencies in `pyproject.toml` and commit `uv.lock`.

Before initializing the project, verify current compatibility among Python, PyTorch, LeRobot, SmolVLA dependencies, LangGraph, Apple Silicon, and CUDA. Do not blindly choose the newest Python version.

Use optional dependency groups where useful so contributors can run core tests without downloading GPU/VLA/simulation dependencies.

Core technologies may include:

- Python, uv, PyTorch;
- Hugging Face LeRobot, SmolVLA, and Hub;
- LangGraph;
- Pydantic;
- pytest, Ruff, and mypy where useful;
- structured logging;
- NumPy and matplotlib;
- FastAPI only when an API milestone needs it.

Optional tools include MuJoCo, OpenTelemetry, Prometheus, MLflow, or Weights & Biases. Do not add all optional infrastructure immediately.

---

## Architecture principles

- clean architecture without ceremony;
- typed Python and explicit domain models;
- strict separation of agent reasoning, policy inference, environment execution, and evaluation;
- dependency injection at external boundaries;
- deterministic implementations for testing;
- provider abstraction for external LLMs;
- structured LLM outputs validated against schemas;
- configuration through settings/environment variables;
- `.env.example`, never real credentials;
- feature flags and graceful degradation;
- reproducible experiments with seeds and versioned configs;
- structured logging and trace correlation;
- unit and integration tests, with future hardware tests isolated;
- a required Makefile;
- README and architecture/evaluation/safety documentation kept current;
- ADRs for significant decisions;
- no empty modules or abstractions created only to match a diagram.

Avoid storing image tensors or large trajectories directly in persistent LangGraph state; store stable references and metadata instead.

---

## Domain and state models

Design explicit models for concepts such as:

```text
RobotObservation
RobotAction
WorldState
TaskGoal
SubTask
TaskPlan
ExecutionResult
ExecutionStatus
FailureReason
RecoveryDecision
PolicyContext
PolicyResult
EvaluationMode
AgentState
```

Avoid arbitrary dictionaries throughout the graph.

The LangGraph state should contain only durable orchestration information, approximately:

```text
original_goal
world_state_reference
plan
current_subtask
completed_subtasks
failed_subtasks
execution_history_references
retry_count
replan_count
last_execution_result
safety_status
evaluation_mode
final_status
```

Explain what belongs in graph state versus an execution context, model runtime, artifact store, or telemetry system.

---

## Planning and graph design

Create a planner boundary supporting both:

- `DeterministicPlanner` for tests, reproducibility, baselines, and isolating graph behavior;
- `LLMTaskPlanner` for validated natural-language planning.

Both must return the same structured `TaskPlan`. Do not parse arbitrary prose. Plans should capture subtasks, dependencies, success criteria, assumptions, and attempt limits where appropriate. Invalid plans must trigger controlled recovery.

The graph may eventually resemble:

```text
START → understand_goal → validate_goal → create_plan
      → select_next_subtask → safety_check → execute_policy
      → observe_result → verify_result
          ├─ success → update_plan → next subtask / complete
          └─ failure → diagnose → retry / replan / request approval / fail
```

Grow this graph milestone by milestone. A graph node should invoke an executor that owns the repeated observation → policy → action → environment loop. Do not model every low-level action as a LangGraph transition.

---

## Planning-granularity experiments

Experimentally compare at least three levels:

1. **VLA-only:** pass the full multi-step instruction directly to the policy.
2. **Coarse agentic plan:** LangGraph produces a small number of meaningful manipulation subtasks.
3. **Fine agentic plan:** LangGraph decomposes the goal into smaller physical skills, without descending into actuator control.

Use the same scenario set, seeds/splits, checkpoint, budgets, and success predicates wherever possible. Measure:

- overall and subtask success;
- plan validity and completion;
- retry/replan/recovery rates;
- policy calls, LLM calls, action steps, and latency;
- failure categories;
- performance by task length, language variation, and scene variation.

Include ablations for deterministic versus LLM planning, verification on/off, and recovery on/off where feasible. Explain confounders and statistical uncertainty. Do not overstate findings from a small sample.

---

## Fine-tuning

Fine-tuning begins only after dataset inspection, a reproducible base-model inference path, and an evaluation pipeline exist.

Use cloud GPU training as the primary route. Record:

- exact base model and revision;
- dataset identity, revision, license, size, and split;
- preprocessing and normalization;
- number of episodes and task/language variations;
- seed, training steps, batch size, optimizer, and learning rate;
- precision, GPU type/count, wall-clock time, and approximate cost;
- checkpoints and artifact references;
- evaluation protocol and limitations.

Compare the base and fine-tuned checkpoint using the same held-out evaluation. Avoid leakage from evaluation episodes or paraphrases into training. Fine-tuning must be configurable and skippable so the rest of the repository remains runnable without cloud resources.

---

## Safety

Safety remains a first-class architectural concern even without hardware. Implement and test:

- allowlisted task/skill categories;
- bounded retries, replans, action horizons, and timeouts;
- schema and range validation for actions;
- explicit cancellation and safety-stop terminal states;
- simulation/mock workspace constraints;
- human approval checkpoints for risky or ambiguous operations;
- fail-closed behavior when state, model output, or policy confidence is invalid;
- redaction of credentials and sensitive telemetry.

Clearly distinguish software safety guards tested in mocks/simulation from guarantees that would require hardware validation. Document future physical checks such as joint limits, collision handling, emergency stops, calibration, and supervised first execution, but do not implement fake hardware safety.

---

## Observability

Use structured, correlated events spanning one episode:

```text
run_id / episode_id
goal and plan version
subtask and attempt
planner/provider/model
policy/checkpoint/dataset revision
evaluation mode and scenario seed
observation/artifact references
execution outcome and failure reason
retry/replan decisions
LLM and policy latency
action count and total duration
safety decisions
```

Make traces useful for reconstructing why a run succeeded or failed without logging huge tensors or secrets. Persist machine-readable evaluation outputs and produce concise summary tables/plots.

---

## Suggested repository structure

```text
langgraph-vla-agent/
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── configs/
├── data/                    # metadata/small fixtures only
├── artifacts/               # ignored generated outputs
├── docs/
│   ├── architecture.md
├── ├── PROJECT_SPEC.md
├── ├── RATIONALE_PER_MILESTONE.md
│   ├── data.md
│   ├── evaluation.md
│   ├── experiments.md
│   ├── safety.md
│   └── decisions/
├── scripts/
├── src/langgraph_vla_agent/
│   ├── config/
│   ├── domain/
│   ├── datasets/
│   ├── policies/            # mock, replay, SmolVLA adapter
│   ├── environments/        # mock, replay, optional simulation
│   ├── planning/
│   ├── agent/
│   ├── execution/
│   ├── evaluation/
│   ├── safety/
│   ├── observability/
│   ├── llm/
│   └── api/                 # only when needed
└── tests/
    ├── unit/
    ├── integration/
    ├── simulation/
    └── hardware/            # optional future work, excluded by default
```

Do not create empty files and directories merely to match this tree. Introduce components only when a milestone requires them, and improve the structure when there is a clear reason.

---

## Makefile and developer experience

Provide clear, documented targets as they become relevant, such as:

```text
make setup
make sync
make format
make lint
make typecheck
make test
make test-unit
make test-integration
make inspect-data
make evaluate-mock
make evaluate-replay
make evaluate-policy
make evaluate-agent
make train
make run-demo
make check
```

Targets must call `uv run` and committed scripts/configuration rather than hiding essential logic in shell recipes. `make check` should run the practical local quality gate. GPU, simulator, network, and future hardware targets must be opt-in and clearly labeled.

---

## Milestones

### Milestone 0 — Foundation and verified project plan

Goal: create a minimal, runnable, well-documented foundation without implementing the VLA or full LangGraph agent.

Required work:

1. Inspect the current compatibility and official usage requirements for Python, uv, PyTorch, LeRobot/SmolVLA, and LangGraph.
2. Select and justify a compatible Python version.
3. Initialize the project with uv; create `pyproject.toml` and `uv.lock`.
4. Add only essential development dependencies.
5. Create the minimal package and one meaningful smoke test.
6. Add Ruff and pytest configuration; add mypy only if useful now.
7. Create a Makefile with working setup/check/test targets.
8. Add `.gitignore` and `.env.example` with no secrets.
9. Write a README explaining the research question, offline/simulation-first scope, architecture boundary, environments, and quick start.
10. Write initial architecture, evaluation, data, and safety notes.
11. Record an ADR explaining why hardware is optional and why mock/replay are first-class backends.
12. Identify one or more candidate public LeRobot-compatible datasets based on verified metadata and license. Do not download a large dataset yet.
13. Define milestone acceptance checks and list unresolved compatibility/data risks.

Acceptance criteria:

- a fresh clone can be set up with documented uv/Make commands;
- lint and tests pass locally;
- no GPU, model download, simulator, cloud account, or robot is required;
- documentation makes no claim of real-world execution;
- dependency and dataset choices cite verified sources or clearly mark open questions;
- the next milestone has an explicit, reviewable plan.

**STOP after Milestone 0.**

Do not begin Milestone 1, download large datasets/checkpoints, start cloud training, build the full graph, or create placeholder architecture. Summarize what was created, show verification results, explain key choices and risks, and wait for my review and explicit approval.

### Milestone 1 — Domain contracts and deterministic mock loop

After approval only: introduce the smallest domain models, policy/environment protocols, deterministic mock policy and environment, executor semantics, and unit tests.

### Milestone 2 — Public dataset inspection and replay backend

Build dataset adapters, inspect a small public sample, validate schema/provenance, construct held-out replay scenarios, and document the limits of replay evaluation.

### Milestone 3 — SmolVLA baseline

Add the thin SmolVLA adapter, reproducible base-checkpoint inference, and VLA-only offline evaluation. Keep GPU/model dependencies optional.

### Milestone 4 — Cloud GPU fine-tuning

Fine-tune SmolVLA on the selected public dataset, preserve configurations and provenance, and compare base versus fine-tuned checkpoints on held-out data.

### Milestone 5 — LangGraph orchestration

Implement deterministic planning first, then structured LLM planning, subtask execution, verification, bounded retry/replanning, persistence choices, and observable traces.

### Milestone 6 — Planning-granularity experiments

Compare VLA-only, coarse decomposition, and fine decomposition under a controlled protocol. Add ablations and analyze gains, costs, and failure modes.

### Milestone 7 — Optional closed-loop simulation

If justified by earlier findings, add a simulator-backed environment and closed-loop evaluation without weakening the offline-first path.

### Milestone 8 — Portfolio hardening

Polish documentation, diagrams, reproducibility, experiment reports, API/demo, CI, and interview-oriented architectural explanations.

### Optional future milestone — Physical robot integration

Only if hardware becomes available: add an SO-101/LeRobot environment adapter, calibration and supervised execution workflow, hardware safety gates, hardware-in-the-loop tests, real-data collection, and sim/offline-to-real comparison. This milestone must not require redesigning the agent or policy abstractions.

---

## Required working style

For every approved milestone:

1. inspect the existing repository before editing;
2. state the milestone goal and proposed changes;
3. verify unstable dependency/API facts against current primary documentation;
4. implement the smallest coherent increment;
5. add tests alongside behavior;
6. run the relevant quality checks;
7. update README/docs/configuration;
8. summarize files changed, commands run, results, tradeoffs, and remaining risks;
9. stop at the milestone boundary and wait for approval.

Ask focused questions only when a decision would materially change scope, cost, evaluation validity, or architecture. Otherwise make a reasonable, documented assumption.

Prefer transparent implementations over fashionable complexity. Do not hide limitations, fabricate metrics, silently substitute datasets/models, or claim physical-robot validation.

Begin with **Milestone 0 only**, then stop.
