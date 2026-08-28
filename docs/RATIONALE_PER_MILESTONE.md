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

## Milestone 4 — Cloud GPU fine-tuning

### SmolVLATrainingConfig: our schema over lerobot's TrainPipelineConfig

**Decision:** M4 defines its own Pydantic `SmolVLATrainingConfig` rather than directly using lerobot's `TrainPipelineConfig` dataclass.

**Why:** lerobot's config uses `draccus` (a dataclass-based CLI parser), not Pydantic. It cannot be serialized to JSON, validated at construction time, or embedded inside `TrainingRunProvenance` as a typed field. Our schema serves a different purpose — reproducibility and provenance — while `lerobot_train_args()` translates it into the actual CLI invocation. If lerobot's CLI changes, only `lerobot_train_args()` needs updating; the provenance schema stays stable.

### lerobot-train CLI entry point

**Decision:** The training script calls `lerobot-train` via subprocess, not lerobot's Python API.

**Why:** lerobot 0.6.1 exposes training via a `draccus`-decorated CLI entry point (`lerobot-train`), not a stable importable Python function. Calling via subprocess is more robust across lerobot minor versions — the command-line contract changes less frequently than internal Python APIs. A `--dry-run` flag prints the command without executing, enabling pre-flight validation on local machines without a GPU.

### eval_split vs test split — two separate holdouts

**Decision:** `eval_split` (a `DatasetConfig` field in lerobot) holds out a fraction of episodes for online val-loss monitoring *during* training. The test split used for the final base-vs-fine-tuned comparison is a separate holdout managed by `compare_checkpoints.py`.

**Why:** These serve different purposes. The val split detects overfitting mid-training and influences early stopping. The test split must never be seen during training in any form (not even for val-loss), so the honest comparison is fully uncontaminated. With 50 episodes, keeping them separate is even more important — using the same data for both would make the comparison meaningless.

### CheckpointComparisonResult: testable without a trained model

**Decision:** `compare_checkpoints()` accepts any `RobotPolicy`, including stubs. `make compare-checkpoints` runs in fixture mode (stub policies, no GPU) to verify infrastructure.

**Why:** The actual training run requires cloud GPU access and takes hours. The comparison infrastructure should be developed, tested, and confident *before* spending compute budget. Fixture mode verifies that `OfflineEvaluator` is called correctly for both policies, delta metrics are computed correctly, and `CheckpointComparisonResult` serializes cleanly — all without a trained checkpoint.

### Camera key mismatch: resolved via rename_map

**Decision:** The M3 camera key mismatch (dataset uses `observation.image.front`; base model expects `camera1/2/3`) is resolved by `rename_map: {"observation.image.front": "observation.images.camera1"}` in `configs/training/smolvla_so100.yaml` and the corresponding `rename_map` field in `SmolVLATrainingConfig`. The config emits `--rename_map=<json>` to `lerobot-train`.

**Why:** lerobot's `TrainPipelineConfig` has a `rename_map: dict[str, str]` field that remaps dataset observation keys to the names the policy expects at training time. This is the lerobot-native solution: the mapping is applied before any batch construction, so the fine-tuned checkpoint will use `observation.images.camera1` (the base model's key) rather than the dataset's `observation.image.front`. Discovered by inspecting `lerobot-train --help` output and `TrainPipelineConfig` source during `make train` debugging. This was the correct fix — adding evidence before acting (running a broken `make train`) surfaced the exact mechanism.

### lerobot-train CLI arg format: `--key=value` (draccus requires double-dash prefix)

**Decision:** All args passed to `lerobot-train` use the `--key=value` format (double-dash prefix). `push_to_hub` is scoped to the policy as `--policy.push_to_hub=false`.

**Why:** Discovered empirically by running `make train` and getting exit code 2. draccus (the arg-parsing library lerobot uses) requires `--key=value` format; bare `key=value` strings are silently unrecognised. Three specific corrections were needed:
1. All args need `--` prefix (`--batch_size=8`, not `batch_size=8`).
2. Policy loading uses `--policy.type=smolvla` (select architecture) + `--policy.pretrained_path=<id>` (load weights), not the non-existent `policy.path`.
3. `push_to_hub` lives under the policy config (`--policy.push_to_hub=false`), not at the top level — lerobot's default is `push_to_hub=True`, which triggers a `ValueError` if `repo_id` is not also set.

### Do not pre-create the output directory before calling lerobot-train

**Decision:** `scripts/train_smolvla.py` does not call `out_dir.mkdir(parents=True, exist_ok=True)` before invoking `lerobot-train`.

**Why:** lerobot's `TrainPipelineConfig.validate()` raises `FileExistsError` if the output directory already exists and `resume=False`. An early `mkdir` from our script triggers this error on the very first run. The fix was simply to remove the pre-creation — lerobot creates the directory itself. If resuming an interrupted run, pass `--resume=true` (a lerobot flag), not by pre-creating the directory.

### Checkpoint path structure: `checkpoints/<step>/pretrained_model/`

**Decision:** The `--finetuned` argument to `compare_checkpoints.py --mode vla` must point to `artifacts/training/<run_name>/checkpoints/<step>/pretrained_model/`, not the run root.

**Why:** lerobot saves each checkpoint in its own numbered subdirectory (`checkpoints/010000/pretrained_model/`), following the `save_freq` schedule. The run root directory does not itself contain a `config.json`, so passing the root path raises `FileNotFoundError`. `make compare-checkpoints-vla` hardcodes the final step path (`010000`) for reproducibility.

### M4 training result (2026-08-27, Apple M4 Max MPS)

10,000 steps on MPS completed in 2 h 23 min at $0 cost. L1 error on 3 synthetic fixture episodes: base=0.1893, fine-tuned=0.1436, delta=−24%. L1 std halved (0.048 → 0.023), indicating more consistent predictions. Provenance: `data/provenance/training/smolvla_so100_m4.yaml`.

---

## Milestone 5 — LangGraph orchestration

### Protocol-based TaskPlanner and LanguageModel

**Decision:** `TaskPlanner` and `LanguageModel` are `typing.Protocol` classes with `@runtime_checkable`. Both `DeterministicPlanner` and `LLMTaskPlanner` satisfy `TaskPlanner` structurally.

**Why:** Same reasoning as M1 Protocols — structural subtyping lets tests inject any object with a compatible `plan()` method without importing from the planning package. This enables 15 deterministic-planner tests and 11 LLM-planner tests to run without langchain, langchain-anthropic, or any LLM key. The same pattern makes `AgentRunner` injectable with any future planner (LLM or otherwise) without touching the graph.

### LangGraph import isolation in graph.py only

**Decision:** Only `src/langgraph_vla_agent/agent/graph.py` imports from `langgraph`. All other agent modules (`state`, `nodes`, `config`, `safety`, `runner`) are importable without the `[agent]` extra.

**Why:** The 30 node unit tests and 9 safety tests run under `make check` — the core dev gate that has no `[agent]` extra. If `nodes.py` or `state.py` imported from `langgraph`, those tests would fail on a fresh checkout. Isolation means the unit-test path stays zero-dependency. The 10 integration tests in `tests/integration/agent/` use `pytest.importorskip("langgraph")` and skip cleanly when the extra is absent.

### Closure-based dependency injection into node functions

**Decision:** `build_agent_graph()` wraps each node function in a closure that closes over the concrete `executor`, `planner`, `config`, and `checker`. Node functions in `nodes.py` accept these as keyword-only arguments.

**Why:** LangGraph node functions have the signature `(state: AgentState) -> dict`. They cannot accept extra arguments at call time. Two alternatives were considered: (1) attach dependencies to the state TypedDict — rejected because state is a public data contract between nodes, not a dependency container, and LangGraph serializes state; (2) use class instances — rejected because they are harder to test independently. Closures are the minimal, non-magical solution: `nodes.py` functions are pure state-in / dict-out and independently testable; `graph.py` is the wiring layer.

### AgentStatus as a separate enum from ExecutionStatus

**Decision:** `AgentStatus` (RUNNING / COMPLETED / FAILED / SAFETY_STOP) is defined in `agent/state.py`. It is distinct from `ExecutionStatus` (SUCCESS / FAILURE / INVALID_ACTION / MAX_STEPS_EXCEEDED / POLICY_ERROR) in `domain/results.py`.

**Why:** These enums operate at different timescales and have different consumers. `ExecutionStatus` describes the outcome of a single policy-execution loop for one subtask. `AgentStatus` describes the terminal state of the entire multi-subtask orchestration episode. Mixing them would force the graph layer to understand Executor internals, violating the layered separation. The `verify_result` node translates from `ExecutionStatus` to the appropriate `AgentStatus` transition.

### Annotated[list[str], operator.add] for append fields in AgentState

**Decision:** `completed_subtask_ids`, `failed_subtask_ids`, and `execution_history_references` use `Annotated[list[str], operator.add]` as their type in `AgentState`.

**Why:** LangGraph's default state merging overwrites fields with the value returned by the node. A node that appends to a list must read the current list, extend it, and return the full new list — a read-before-write anti-pattern that breaks under concurrent execution and requires every node to know the full state. The `Annotated` reducer tells LangGraph to apply `operator.add` (list concatenation) on the returned list. Nodes return only the new items (`[subtask_id]`), and LangGraph merges them. This is the idiomatic LangGraph pattern for accumulator fields.

### DeterministicPlanner with coarse/fine granularity

**Decision:** `DeterministicPlanner` supports two granularity modes: `"coarse"` (2 subtasks: approach-and-grasp, move-and-place) and `"fine"` (5 subtasks: approach, grasp-and-lift, move-to-target, lower-and-place, release-and-retract).

**Why:** M6's research question compares coarse vs fine agentic decomposition against VLA-only. To run that comparison, we need two concrete agentic conditions. The coarse plan mirrors the minimal subtask decomposition that a human might verbally describe. The fine plan breaks the same task into skills closer to the policy's training distribution (individual approach, grasp, move, place, release steps). Both are deterministic (no LLM call), reproducible, and testable — which is essential for the M6 comparison to be fair and repeatable.

### Bounded retry and replan with fail-closed behavior

**Decision:** `AgentConfig` has `max_retries: int` (per subtask) and `max_replans: int` (per episode). Exhausting both sets `final_status=FAILED`. There is no infinite retry.

**Why:** Unbounded retry is unsafe in any agent system — it can loop indefinitely on a permanently failing state. Fail-closed is the correct default: if the agent cannot succeed within the configured budget, it halts and reports failure rather than continuing to consume resources. The bounds also make the graph termination-provable: the maximum path length is `max_replans × (subtasks × max_retries) + subtasks`, which is finite and computable from the config. This property is critical for the M6 experiment where all three conditions must be run with identical budgets.

---

## Milestone 6 — (Pending)
## Milestone 7 — (Pending)
## Milestone 8 — (Pending)
