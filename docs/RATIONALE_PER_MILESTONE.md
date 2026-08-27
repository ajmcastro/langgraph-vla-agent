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

## Milestone 2 — Public dataset inspection and replay backend

### JSON fixtures for unit tests; YAML for provenance

**Decision:** Replay episode fixtures are committed as small JSON files in `data/fixtures/episodes/`. Dataset provenance metadata is stored in `data/provenance/*.yaml`.

**Why:** JSON round-trips through Pydantic's `model_validate_json()` without any extra library. It is human-readable and small enough to commit. YAML is preferred for the provenance record because it supports inline comments (open questions, notes) that JSON does not. Neither format requires downloading data. Large dataset files (parquet, video frames) are never committed — only the metadata.

### ReplayEnvironment ignores the action argument

**Decision:** `ReplayEnvironment.step()` accepts a `RobotAction` argument (to satisfy the `RobotEnvironment` Protocol) but discards it. The next observation always comes from the recorded trajectory, not from simulating the action.

**Why:** This is the defining semantic of offline/replay evaluation. The environment is a data-playback device, not a physics simulator. Pretending the action influences the next state would be incorrect and misleading — the only honest behaviour is to play back the pre-recorded observation regardless of what action was taken. This constraint is documented explicitly in the class docstring and in `docs/evaluation.md` so evaluators cannot mistake replay results for closed-loop performance.

### ReplayRobotPolicy and ReplayEnvironment have independent step pointers

**Decision:** Both objects take the same `ReplayEpisode` but each maintains its own `_ptr: int`. Both reset to `ptr=0` on their respective `reset()` calls.

**Why:** The Executor calls `policy.reset()` and `env.reset()` independently at the start of each `run()`. If a shared pointer were used, one `reset()` call could corrupt the other's state. Independent pointers keep the objects loosely coupled and let either be swapped out without coordination (e.g. replace `ReplayRobotPolicy` with `SmolVLAPolicyAdapter` in M3 while keeping `ReplayEnvironment`).

### EpisodeStore as a structural Protocol

**Decision:** `EpisodeStore` uses `typing.Protocol` with `@runtime_checkable`, matching the M1 pattern for `RobotPolicy` and `RobotEnvironment`.

**Why:** The same reasoning applies: `FixtureEpisodeStore` (M2) and a future `HubEpisodeStore` (M3) satisfy the Protocol without inheriting from it. The Executor and evaluation code import only the Protocol — never a concrete implementation — keeping layers decoupled.

### [datasets] optional extra for huggingface_hub

**Decision:** `huggingface_hub>=0.24.0` is placed in a new `[datasets]` optional extra, separate from `[vla]`.

**Why:** `[vla]` pulls in all of LeRobot and PyTorch (gigabytes). Dataset metadata inspection only needs `huggingface_hub` (~MB). Separating these lets contributors run `make inspect-data` to fetch Hub metadata without setting up the full ML stack. `HubDatasetInspector` guards its import with a `try/except ImportError` and raises a clear message if the extra is missing.

## Milestone 3 — SmolVLA baseline and offline evaluation

### SmolVLAPolicyAdapter: adapter over ABC

**Decision:** `SmolVLAPolicyAdapter` satisfies `RobotPolicy` via structural typing (Protocol), not by inheriting from it.

**Why:** The same argument as M1 Protocols — future checkpoints or alternative VLA models can satisfy the interface without importing from the agent codebase. The adapter holds `self._model: _SmolVLAModel` (an internal Protocol), so the real `SmolVLAPolicy` and the test stub `_StubSmolVLAModel` are interchangeable at the adapter boundary.

### Dependency injection via `_model` for unit tests

**Decision:** `SmolVLAPolicyAdapter.__init__` accepts a `_model` keyword argument that bypasses the lerobot import entirely. `_StubSmolVLAModel` (defined in the same module) satisfies the internal `_SmolVLAModel` Protocol and returns zero vectors.

**Why:** The [vla] extra requires a CUDA or Apple-Silicon GPU and a ~500 MB model download. Unit tests must run on any developer laptop without those requirements. Dependency injection is the minimal, non-fragile way to achieve this — no monkeypatching, no mocking `import lerobot`, no conftest magic.

### OfflineEvaluator is separate from Executor

**Decision:** Action prediction error is computed in `OfflineEvaluator`, which runs its own per-step loop. It does not reuse `Executor`.

**Why:** `Executor.run()` returns a single opaque `ExecutionResult` — there is no hook for per-step inspection of policy outputs. The evaluator needs to see every predicted action alongside the corresponding ground-truth action to compute L1/L2 error. Exposing this through the Executor would require adding evaluation-specific logic to a general-purpose component, violating the layered separation. The cleaner design is a dedicated evaluation class that owns the comparison loop.

### L1 and L2 error: per-step scalars, aggregated across episodes

**Decision:** At each step, L1 = `mean(|predicted - gt|)` and L2 = `sqrt(mean((predicted - gt)²))` over action dimensions. These are then aggregated (mean ± std) across all steps of all evaluated episodes.

**Why:** Computing per-step scalars (not per-joint arrays) keeps `ActionErrorMetrics` simple and readable in logs. Aggregating across all steps (not per-episode means of means) gives the correct overall statistics — each step is equally weighted regardless of episode length. Both metrics are standard in offline imitation learning evaluation and directly comparable to the SmolVLA paper's reported baselines.

### Structural evaluation_note in OfflineEvalResult

**Decision:** `OfflineEvalResult` has a `evaluation_note: str = _OFFLINE_NOTE` field with a default value. It cannot be set to an empty string by normal construction.

**Why:** The fundamental limitation of offline evaluation — that it cannot measure closed-loop counterfactual behavior — must be structurally attached to every result object. If it were only in documentation, it would be easy to strip when reporting metrics. Making it a model field ensures that any consumer (a script, a report generator, a future M6 dashboard) sees the limitation alongside the numbers.

### lerobot 0.6.1 API differences discovered during M3

Three deviations from the originally anticipated API were found when running `make evaluate-policy` against the installed lerobot 0.6.1. These required fixes to `SmolVLAPolicyAdapter._build_batch`.

**1. Import path changed — `lerobot.common.policies` no longer exists.**

lerobot 0.6 removed the `common/` prefix from its package layout. The correct import is `lerobot.policies.smolvla.modeling_smolvla`. The fix is a one-line change in the adapter; the `lerobot.*` mypy override already covers both paths.

**2. Image keys and shapes are checkpoint-specific, not dataset-generic.**

`smolvla_base` expects `observation.images.camera1/2/3` at shape `(3, 256, 256)`, not `observation.images.front` at `(3, 480, 640)` as the SO-100 dataset documentation suggests. `_build_batch` now introspects `model.config.image_features` to discover the expected keys and shapes at runtime, building a dummy black image for each key the checkpoint declares. This makes the adapter portable across checkpoints without hardcoding any key or size.

**3. Language input must be pre-tokenized; `select_action` does not accept raw strings.**

`SmolVLAPolicy.select_action` reads `observation.language.tokens` (Long tensor) and `observation.language.attention_mask` (**boolean** tensor) directly from the batch. It does not accept a `task` string and tokenize internally. `_build_batch` now retrieves the GPT-2 tokenizer from `model.model.vlm_with_expert.processor.tokenizer` at init time, tokenizes the instruction, and adds both tensors. The attention mask must be `.bool()` — passing a Long (0/1) mask causes a `RuntimeError` inside the model's attention implementation.

These three fixes were driven by actual runtime errors against the installed package, not assumptions. They are load-bearing details for M4 fine-tuning and any future checkpoint swap.

## Milestone 4 — (Pending)
## Milestone 5 — (Pending)
## Milestone 6 — (Pending)
## Milestone 7 — (Pending)
## Milestone 8 — (Pending)
