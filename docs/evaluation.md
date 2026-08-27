# Evaluation

Every result, metric, and claim in this project must identify its evaluation mode. The four modes are strictly separated in code, test fixtures, configuration, and reporting.

---

## Evaluation modes

### Mock evaluation

**What it is:** Deterministic scripted scenarios executed against `MockRobotPolicy` and `MockEnvironment`. No network, no GPU, no dataset.

**What it proves:** Software behavior — graph routing, retry logic, replanning decisions, safety gate triggers, terminal state transitions, and plan validation.

**What it does NOT prove:** Any capability related to real sensorimotor execution, policy quality, or task success on real or simulated scenes.

**Use it for:** All unit and integration tests. Every agent behavior must be testable in mock mode.

---

### Offline / replay evaluation

**What it is:** Evaluation of policy predictions or orchestration decisions against held-out episodes from a recorded dataset. The `ReplayRobotPolicy` re-serves recorded actions; the `ReplayEnvironment` replays recorded observations.

**What it can measure:**
- Action prediction error (L1/L2 on action chunks)
- Action-chunk mean error over a horizon
- Instruction-conditioned performance (does accuracy vary by task language?)
- Subtask completion as defined by replay-level success predicates
- Plan validity and plan completion rates against replay-defined outcomes
- Retry and replan rates under various orchestration strategies

**What it CANNOT measure:** Closed-loop counterfactual behavior. If the agent takes a different action than the recorded one, the replay environment cannot simulate what would happen next. Offline metrics are necessary but not sufficient evidence of task success.

**Required disclosure:** All offline metrics must state: *"Measured in offline/replay mode. Results cannot be extrapolated to closed-loop task performance without further simulation or hardware experiments."*

---

### Simulation evaluation

**What it is:** Closed-loop execution in a physics simulator (e.g., MuJoCo via `gym-pusht` or `gym-aloha`). The agent takes real actions and observes simulated consequences.

**What it can measure:** Closed-loop task success rate, recovery behavior under perturbation, and multi-step planning effectiveness in a controlled world.

**Required disclosure:** Report the simulator name and version, task definitions, episode seeds, success predicates, and the known gap between the simulated and real task. Do not claim sim-to-real transfer without separate hardware experiments.

**When to use:** Only if a simulator adds credible closed-loop evidence that offline evaluation cannot provide and the additional complexity is justified by the research question. This is Milestone 7 and explicitly optional.

---

### Hardware evaluation (future)

**Status:** Not yet implemented. Requires physical robot, safety review, and supervised execution.

**What it would measure:** Real closed-loop task success under real-world conditions, including sensing noise, calibration error, and physical interaction.

**Required disclosure:** A hardware result section must describe the robot model, calibration procedure, safety checks performed, number of trials, evaluator supervision level, and any task simplifications relative to the full spec.

**Policy:** No hardware metrics may be fabricated or implied from offline/simulation results. All hardware experiments are explicitly out of scope until physical hardware is available and safety review is completed.

---

## Metrics

| Metric | Mode | Notes |
|---|---|---|
| Action L1/L2 error | Offline | Per-joint and aggregate; report mean ± std |
| Action-chunk error | Offline | Mean error over the prediction horizon |
| Subtask success rate | Offline / Sim | Fraction of subtasks reaching the defined terminal condition |
| Task success rate | Offline (proxy) / Sim / Hardware | Fraction of full multi-step tasks completed |
| Plan validity rate | Mock / All | Fraction of LLM plans that pass schema validation |
| Retry rate | All | Mean retries per subtask |
| Replan rate | All | Mean full replanning cycles per episode |
| LLM call count | All | Total LLM calls per episode |
| Policy call count | All | Total policy.act() calls per episode |
| Episode latency | All | Wall-clock time from goal to terminal state |
| Failure categories | All | Proportion of failures by type (plan invalid, policy error, safety stop, timeout) |

---

## Comparison protocol (Milestone 6)

The planning-granularity experiment compares three conditions:
1. **VLA-only** — the full multi-step instruction is passed directly to the policy; no LangGraph orchestration.
2. **Coarse agentic** — LangGraph decomposes the goal into 2–4 meaningful manipulation subtasks (e.g. "approach → grasp → lift → place").
3. **Fine agentic** — LangGraph decomposes into smaller physical skills (e.g. "open gripper → move to pre-grasp → close gripper → …").

### Planned experimental design

**Dataset and task:** `lerobot/svla_so100_pickplace` (50 episodes, Apache-2.0). Task: pick a cube from one of 5 positions and place it in a target zone. This is the reference task from the SmolVLA paper, making results directly comparable to the published baseline.

**Policy checkpoint:** The M4 fine-tuned checkpoint (`smolvla_so100_m4`, step 010000) for all three conditions. The base model (`lerobot/smolvla_base`) is evaluated separately as an additional baseline to isolate the effect of fine-tuning from the effect of orchestration.

**Evaluation mode:** Offline/replay. The test split (15% holdout, ~7–8 episodes not seen during training) is the scenario set. Each episode is run under all three conditions against the same recorded observations.

**Success predicate:** Action prediction L1 error below the base model's per-episode mean on the same split. Subtask completion is defined by replay-level episode termination signals.

**Action budget:** Max 200 steps per episode (matching the longest episode in `svla_so100_pickplace`). Subtask-level budget allocated proportionally by the planner.

**Sample size:** With ~7–8 test episodes, n < 10 per condition. The small-sample disclaimer applies to all M6 results. Statistical tests (Wilcoxon signed-rank) are included for completeness but effect sizes and direction are the primary evidence.

**What would constitute a meaningful result:**
- VLA-only vs coarse agentic: a consistent L1 reduction (>10%) or improved subtask completion rate across all test episodes would support the orchestration hypothesis.
- Coarse vs fine agentic: a tradeoff — lower error but more LLM calls and higher latency — would support the granularity hypothesis.
- No measurable difference would be an honest and publishable null result: it would suggest that SmolVLA's native language understanding is sufficient for this task without decomposition.

### What must be finalized in M5 (before M6 can run)

The LangGraph graph (M5) determines what "a coarse subtask decomposition" and "a fine subtask decomposition" actually mean in code. The subtask vocabulary, plan schema, and orchestration logic are M5 deliverables. M6 cannot be designed beyond the structural level above until M5 exists. Specifically:
- Exact subtask labels and granularity levels (defined by the planner in M5)
- How the agent decides a subtask is complete in replay mode (must be replay-compatible — no live camera feedback)
- Whether the LLM planner is deterministic or sampled (affects reproducibility)

### Consistency requirements (all conditions)

- Same test split and episode order
- Same policy checkpoint
- Same action budget
- Same success predicates
- Same evaluation mode (offline/replay)

Results are reported with mean ± std and sample size. Statistical tests are applied when n ≥ 10 per condition. The small-sample disclaimer is included when n < 10.

---

## What offline evaluation proves in M3

Milestone 3 adds `SmolVLAPolicyAdapter`, `OfflineEvaluator`, and action-prediction error metrics. With M3 running, these claims are testable and verifiable **without a GPU, network, or robot**:

| Claim | How it is tested |
|---|---|
| `SmolVLAPolicyAdapter` satisfies `RobotPolicy` protocol (structurally) | `test_smolvla_adapter_satisfies_robot_policy_protocol` |
| `SmolVLAPolicyAdapter` accepts a test stub via `_model` injection (no [vla] needed) | `test_smolvla_adapter_accepts_stub_without_vla` |
| `reset()` delegates to the underlying model's `reset()` | `test_reset_calls_model_reset` |
| `act()` returns a 6-D float32 `RobotAction` | `test_act_returns_correct_dim`, `test_act_returns_float32` |
| `act()` creates a black dummy image when no camera image is present | `test_act_handles_observation_without_image` |
| `OfflineEvaluator` produces L1=0 when `ReplayRobotPolicy` is used (baseline sanity check) | `test_offline_evaluator_zero_error_with_replay_policy` |
| `OfflineEvaluator` produces L1>0 when a stub predicts zeros against non-zero recorded actions | `test_offline_evaluator_nonzero_error_with_stub` |
| L1 and L2 values match manual calculation | `test_l1_l2_error_values_are_correct` |
| Every `OfflineEvalResult` carries a structural disclaimer | `evaluation_note` field, `test_offline_eval_result_always_has_evaluation_note` |
| `ActionErrorMetrics` rejects negative means, stds, and step counts | `test_action_error_metrics_rejects_negative_*` |
| Aggregate covers all steps across multiple episodes | `test_offline_evaluator_aggregates_multiple_episodes` |

**What M3 does NOT prove:**

- That `lerobot/smolvla_base` makes *good* predictions on `svla_so100_pickplace`. The integration tests (`SMOLVLA_INTEGRATION_TESTS=1`) only verify that the model produces *valid* outputs (correct shape, finite values) — not that the predictions are close to ground truth. Meaningful accuracy requires running `make evaluate-policy` against the actual held-out dataset split, which is M4 work.
- That low action prediction error implies task success in closed-loop execution (requires simulation or hardware).
- That the SmolVLA model generalises beyond the 3 synthetic fixture episodes used in unit and script tests (the fixtures are hand-crafted JSON files, not real robot data from `svla_so100_pickplace`).

All M3 evaluation results are labeled `evaluation_mode=REPLAY` and carry `OfflineEvalResult.evaluation_note` stating the closed-loop limitation explicitly.

---

## What replay evaluation proves in M2

With the M2 replay backend running, these claims are testable and verifiable:

| Claim | How it is tested |
|---|---|
| `ReplayRobotPolicy` serves recorded actions in episode order | `test_replay_policy_returns_actions_in_order` |
| `ReplayRobotPolicy` resets to step 0 on `reset()` — independent episodes | `test_replay_policy_resets_ptr_to_zero`, `test_replay_policy_second_run_starts_from_beginning` |
| `ReplayRobotPolicy` raises `IndexError` when exhausted | `test_replay_policy_raises_when_exhausted` |
| `ReplayEnvironment` ignores the action argument and replays recorded observations | `test_replay_environment_action_is_ignored` |
| `ReplayEnvironment` returns `terminated=True, success=True` at the last step of a successful episode | `test_replay_environment_step_returns_success_on_terminal` |
| `ReplayEnvironment` over-run truncates gracefully (no crash) | `test_replay_environment_overrun_truncates` |
| `FixtureEpisodeStore` round-trips episode data from JSON without data loss | `test_load_episode_returns_replay_episode`, `test_load_episode_has_correct_action_dim` |
| `EpisodeSplitter` produces the same split for the same seed | `test_split_is_deterministic_with_same_seed` |
| `EpisodeSplitter` splits are leak-free — no episode in two partitions | `test_split_has_no_leakage` |
| The Executor with replay backends returns SUCCESS matching the recorded episode outcome | `test_executor_replay_success_episode` |
| The Executor with replay backends returns FAILURE matching the recorded episode outcome | `test_executor_replay_failure_episode` |

None of these claims require a GPU, network, dataset, or robot. They prove the **infrastructure** for replay evaluation is correct, not that any model performs well on a task.

**What replay infrastructure does NOT prove:**

- That `lerobot/smolvla_base` or any fine-tuned checkpoint makes good predictions (M3).
- That a policy would succeed or fail if allowed to take actions that differ from the recorded ones (closed-loop evidence requires simulation or hardware).
- That the recorded task outcomes generalise beyond the 50 episodes in `svla_so100_pickplace`.

---

## What mock evaluation proves in M1

With the M1 mock loop running, these claims are testable and verifiable:

| Claim | How it is tested |
|---|---|
| The executor returns `SUCCESS` when the environment signals success | `SUCCEED_AT_STEP` scenario, confirmed by `test_executor_success_path` |
| The executor returns `INVALID_ACTION` when the policy returns NaN | `INVALID_AFTER_N` policy + `test_executor_invalid_action_path` |
| The executor returns `MAX_STEPS_EXCEEDED` when the loop budget is exhausted | `NEVER_TERMINATE` env + `max_steps=5`, confirmed by step count |
| The executor catches policy exceptions and returns `POLICY_ERROR` | `RAISE_AFTER_N` policy + `test_executor_policy_error_path` |
| `validate_actions=False` passes NaN to the environment | Confirmed by `test_executor_validation_disabled_passes_nan` |
| `reset()` is called with the correct `PolicyContext` | Confirmed by `test_executor_passes_context_to_policy_reset` |
| Two consecutive runs are independent (no shared state) | Confirmed by `test_executor_runs_are_independent` |

None of these claims require a GPU, network, dataset, or robot. They prove software behavior, not sensorimotor capability.

---

## What fine-tuning comparison proves in M4

Milestone 4 adds `SmolVLATrainingConfig`, `TrainingRunProvenance`, `CheckpointComparisonResult`, and `compare_checkpoints()`. The comparison infrastructure is testable without a GPU:

| Claim | How it is tested |
|---|---|
| `SmolVLATrainingConfig` rejects invalid hyperparameters (batch_size=0, steps<0, eval_split≥1) | `test_rejects_zero_batch_size`, `test_rejects_negative_steps`, `test_rejects_eval_split_ge_one` |
| `lerobot_train_args()` produces the correct CLI argument list | `test_lerobot_train_args_contains_required_keys`, `test_lerobot_train_args_includes_revision_when_set` |
| `lerobot_train_command()` produces a multi-line shell command | `test_lerobot_train_command_is_multiline_string` |
| `TrainingRunProvenance` round-trips through JSON without data loss | `test_provenance_round_trip` |
| `compare_checkpoints()` returns `delta=0` when both policies are identical | `test_delta_is_zero_when_both_policies_identical` |
| `compare_checkpoints()` returns negative delta when fine-tuned is better | `test_negative_delta_when_finetuned_is_better` |
| `compare_checkpoints()` returns positive delta when fine-tuned is worse | `test_positive_delta_when_finetuned_is_worse` |
| `improvement_pct_l1 = 0.0` (not a crash) when base L1 = 0 | `test_improvement_pct_zero_when_base_l1_is_zero` |
| `CheckpointComparisonResult` carries a structural `evaluation_note` | `test_comparison_result_has_evaluation_note` |

**M4 training run completed (2026-08-27):**

Training ran for 10,000 steps on Apple M4 Max (MPS) in 2 h 23 min at $0 cost. Comparison against `lerobot/smolvla_base` on 3 synthetic fixture episodes:

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
| L1 mean | 0.1893 | 0.1436 | −0.046 (−24%) |
| L2 mean | 0.2175 | 0.1812 | −0.036 |
| L1 std | 0.0481 | 0.0225 | halved |

Full provenance: `data/provenance/training/smolvla_so100_m4.yaml`.

**What M4 still does NOT prove:**

- That the 24% L1 improvement holds on real held-out episodes from `svla_so100_pickplace` (fixture episodes are synthetic — evaluating on real test-split episodes requires downloading the full dataset and building a `HubEpisodeStore` or similar).
- That lower L1/L2 error translates to higher task success (requires simulation or hardware — M7+).
- That the training config is optimal (10 k steps on MPS is a reasonable baseline, not a tuned recipe; a cloud CUDA run at full batch size may produce different results).

All comparison results are labeled `evaluation_mode=REPLAY` and carry `CheckpointComparisonResult.evaluation_note`.

---

## Anti-patterns to avoid

- Reporting simulation or offline results as "robot performance"
- Tuning evaluation scenarios on the same data used for training
- Omitting the evaluation mode label from any published metric
- Comparing conditions that used different seeds, checkpoints, or splits
- Claiming recovery performance when recovery was never triggered in evaluation
