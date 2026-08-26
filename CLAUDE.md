# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**LangGraph VLA Agent** is an educational, production-quality open-source project demonstrating how an agentic planning layer (LangGraph + LLM) can orchestrate a Vision-Language-Action (VLA) sensorimotor policy (SmolVLA via Hugging Face LeRobot) for multi-step manipulation tasks.

**No physical robot is required.** The primary path is offline/replay evaluation using public LeRobot-compatible datasets and deterministic mocks. Hardware integration is optional future work behind isolated adapters.

The full project spec lives in [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

---

## Dependency and environment management

- **All** package management goes through `uv`. Never use pip, Poetry, Conda, or requirements.txt.
- Python version and venv are managed by uv; see `pyproject.toml`.
- Lock file (`uv.lock`) is committed and must stay in sync.
- Optional dependency groups keep GPU/VLA/simulation deps out of the core test path.

Common commands (once `pyproject.toml` and `Makefile` exist):

```bash
make setup          # uv sync --all-extras
make sync           # uv sync (core deps only)
make check          # format + lint + typecheck + test-unit (local quality gate)
make test           # all tests
make test-unit      # uv run pytest tests/unit
make test-integration  # uv run pytest tests/integration
make format         # uv run ruff format .
make lint           # uv run ruff check .
make typecheck      # uv run mypy src
```

GPU, simulator, network, and hardware targets are opt-in and labeled accordingly. `tests/hardware/` is excluded from normal CI and requires an explicit marker.

---

## Architecture

The project is milestone-driven. **Do not create files or modules that a current milestone does not require.** Check the current milestone status before adding anything.

### Layered separation (strictly enforced)

```
LangGraph agent (goal/subtask timescale)
    └─ Planning: DeterministicPlanner | LLMTaskPlanner → TaskPlan
    └─ Graph nodes: understand_goal → create_plan → select_subtask
                    → safety_check → execute_policy → verify_result
                    └─ failure path: diagnose → retry/replan/fail
          ↓
Executor (observation → policy → action → environment loop)
          ↓
RobotPolicy (Protocol)
    └─ MockRobotPolicy      ← always available, no deps
    └─ ReplayRobotPolicy    ← public datasets, no GPU
    └─ SmolVLAPolicyAdapter ← optional GPU dep
          ↓
RobotEnvironment (Protocol)
    └─ MockEnvironment
    └─ ReplayEnvironment
    └─ SimulationEnvironment (optional, future)
    └─ HardwareEnvironment  (optional, future, isolated)
```

**The LLM/LangGraph layer must never produce joint torques, servo positions, or motor commands.** It operates at the goal and subtask level only.

### LangGraph state

Store only durable orchestration metadata in graph state — never raw image tensors or full trajectories. Use references (e.g., file paths, artifact IDs) for large data.

Key state fields: `original_goal`, `plan`, `current_subtask`, `completed_subtasks`, `failed_subtasks`, `execution_history_references`, `retry_count`, `replan_count`, `last_execution_result`, `safety_status`, `evaluation_mode`, `final_status`.

### Domain models

Use typed Pydantic models throughout. Avoid plain dicts in graph state or between layers. Core models (introduced per milestone, not all at once): `RobotObservation`, `RobotAction`, `WorldState`, `TaskGoal`, `SubTask`, `TaskPlan`, `ExecutionResult`, `ExecutionStatus`, `FailureReason`, `PolicyContext`, `AgentState`.

---

## Evaluation modes — keep distinct in code and claims

| Mode | What it proves | Key constraint |
|---|---|---|
| **Mock** | Software behavior (graph routing, retries, safety) | Deterministic; never implies robot capability |
| **Offline/Replay** | Action prediction quality against recorded episodes | Cannot measure closed-loop counterfactuals |
| **Simulation** | Credible closed-loop evidence | Must cite simulator, seeds, success predicates |
| **Hardware** | Real-world performance | Optional future; never fabricate |

Every result, metric, and claim must identify its evaluation mode.

---

## Safety (software layer)

Even without hardware, safety is first-class:
- Allowlisted task/skill categories
- Bounded retries, replans, action horizons, and timeouts
- Schema and range validation on all policy outputs
- Explicit cancellation and safety-stop terminal states
- Fail-closed behavior on invalid state or model output
- Human approval checkpoints for ambiguous operations

Distinguish software guards (testable in mocks) from physical guarantees (require hardware validation). Do not implement fake hardware safety.

---

## Data

- Use public LeRobot-compatible datasets; verify license, schema, embodiment, and LeRobot version compatibility before selecting.
- Never commit large datasets or model checkpoints. Store only metadata, small fixtures, and cache references.
- Dataset provenance (source, revision, license, checksum) must be recorded.
- `data/` holds metadata and small fixtures only. `artifacts/` holds generated outputs and is git-ignored.

---

## Key working rules

- Inspect the existing repository before proposing or making changes.
- Implement the smallest coherent increment a milestone requires; stop at milestone boundaries.
- Add tests alongside new behavior; do not create test files without corresponding implementation.
- Run `make check` before marking work complete.
- `docs/decisions/` holds ADRs for significant architectural choices.
- Do not fabricate metrics, claim physical-robot validation, or silently substitute datasets or models.
- Configuration via environment variables and `configs/`; `.env.example` for documentation, never real credentials.
- Structured logging with correlated `run_id`/`episode_id`; no secrets or large tensors in logs.
