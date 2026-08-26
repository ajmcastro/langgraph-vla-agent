# Rationale per Milestone

Design decisions made at each milestone boundary — the "why" that should survive code churn.

---

## Milestone 0 — Foundation

### Python 3.12

**Decision:** Require Python ≥ 3.12.

**Why:** LeRobot (the library wrapping SmolVLA) explicitly requires Python ≥ 3.12 as of mid-2026. Its PyTorch requirement (≥ 2.7) also resolves cleanly with 3.12. LangGraph (≥ 1.2.0) supports Python ≥ 3.10, so 3.12 is a strict subset of both. Choosing 3.12 means the same virtual environment can host LeRobot, LangGraph, and all dev tools without a compatibility shim.

**Risk considered:** Python 3.12 type annotation syntax (`type X = Y`) is new; we use `TypeAlias` from `typing` for compatibility where needed.

### Minimal core dependencies

**Decision:** Core install (`uv sync` without extras) pulls in only `pydantic`, `structlog`, and `numpy`. LeRobot and LangGraph are optional extras.

**Why:** The mock evaluation path — which covers all agent graph tests — needs none of the ML dependencies. Keeping them optional means contributors can run the full unit test suite on any laptop without downloading gigabytes of model weights or dealing with PyTorch/CUDA setup.

**Trade-off:** The `[vla]` extra is large. First-time setup for Milestone 3+ takes time and requires either Apple Silicon + ffmpeg or a CUDA GPU. This is documented and expected.

### uv for all dependency management

**Decision:** All Python version, venv, and package operations go through `uv`. No pip, Poetry, Conda, or requirements.txt.

**Why:** `uv` provides deterministic installs from `uv.lock`, is significantly faster than pip, handles Python version management, and supports the `--torch-backend` flag for correct CUDA wheel selection on Linux. The LeRobot installation docs explicitly show `uv` as the recommended non-conda path.

### Ruff over flake8 + black + isort

**Decision:** `ruff` handles linting and formatting. `black` and `isort` are not added.

**Why:** Ruff replaces all three in a single tool with near-identical rules and is significantly faster. It's already the standard in the HuggingFace ecosystem (LeRobot uses it). mypy remains separate because ruff does not do type checking.

### Primary dataset: `lerobot/svla_so100_pickplace`

**Decision:** Identify `lerobot/svla_so100_pickplace` as the primary candidate dataset for Milestones 2–4.

**Why:** This is the reference dataset from the SmolVLA paper (arXiv:2506.01844), explicitly designed for fine-tuning `lerobot/smolvla_base`. Using it creates a direct baseline: the paper's own data → the paper's own model → our orchestration layer on top. Any improvement we measure is relative to the same starting point used to evaluate SmolVLA itself.

**Risk:** 50 episodes across 5 cube positions is small. Generalisation claims must be qualified accordingly. This is flagged as an open question for Milestone 2.

---

## Milestone 1 — Domain contracts and deterministic mock loop

### Protocol over ABC for policy and environment boundaries

**Decision:** `RobotPolicy` and `RobotEnvironment` are `typing.Protocol` classes, not abstract base classes (ABC).

**Why:** Protocol uses structural subtyping — any class with the right method signatures satisfies the interface, without needing to inherit from it. This means `SmolVLAPolicyAdapter` can satisfy `RobotPolicy` without importing from the agent codebase. It also means the Protocol is checkable at static analysis time (mypy) and optionally at runtime (`@runtime_checkable`). ABCs would force every future implementation to import and inherit from our base, creating an unnecessary coupling.

### Validation gate location: Executor, not RobotAction

**Decision:** `RobotAction` allows NaN/Inf values in its `values` field. The Executor validates finiteness before passing the action to the environment.

**Why:** The domain model needs to be constructable with invalid values so test code can exercise the Executor's validation path. If the model itself rejected NaN, there would be no way to test `INVALID_ACTION` behavior without bypassing Pydantic. The Executor's gate is also configurable (`validate_actions=False`) which allows replay evaluation to skip validation for performance where actions are pre-validated upstream.

### StrEnum over (str, Enum)

**Decision:** All enums inherit from `StrEnum` (Python 3.11+) rather than the older `(str, Enum)` pattern.

**Why:** `StrEnum` is the modern, explicit way to create string enums in Python 3.12. It is more readable, is the ruff-recommended form (UP042), and works identically in all contexts where string comparison or JSON serialization is needed. Since we require Python ≥ 3.12, there is no compatibility cost.

### npt.NDArray[Any] for numpy type annotations

**Decision:** numpy array fields use `npt.NDArray[Any]` rather than bare `np.ndarray`.

**Why:** mypy strict mode requires type arguments for generic types. `np.ndarray` is generic over the dtype and shape. Using `npt.NDArray[Any]` satisfies mypy while remaining flexible — the dtype is embodiment-specific (float32 for SO-100 joint positions, uint8 for camera images) and will be tightened per-field in M3 when SmolVLA's observation space is known precisely.

### Domain models deferred to M5: TaskGoal, TaskPlan, WorldState, AgentState

**Decision:** These models are not created in M1.

**Why:** They require the LangGraph orchestration layer (M5) to be meaningful. `TaskPlan` is a list of `SubTask`s — without a planner, there's nothing to produce it. `AgentState` is the LangGraph state TypedDict — it requires `langgraph` as a dependency. Creating stubs now would be empty modules without tests, violating the "no abstractions created only to match a diagram" rule.

## Milestone 2 — (Pending)
## Milestone 3 — (Pending)
## Milestone 4 — (Pending)
## Milestone 5 — (Pending)
## Milestone 6 — (Pending)
## Milestone 7 — (Pending)
## Milestone 8 — (Pending)
