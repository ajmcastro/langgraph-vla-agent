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

The planning-granularity experiment compares three conditions, all routed through the same compiled LangGraph graph with identical `AgentConfig`, `max_retries`, `max_replans`, and environment settings:

| Condition | Planner | Subtasks | Policy calls (mock) |
|---|---|---|---|
| `vla_only` | `VlaOnlyPlanner` | 1 (full goal as one subtask) | 1 |
| `coarse_agentic` | `DeterministicPlanner("coarse")` | 2 | 2 |
| `fine_agentic` | `DeterministicPlanner("fine")` | 5 | 5 |

### M6 implementation (completed)

**What runs:** `run_granularity_experiment(scenarios, max_retries, max_replans)` iterates all three conditions over a shared list of `EpisodeScenario` objects. Each (condition, episode) pair builds a fresh `AgentRunner` with `MockRobotPolicy` and `MockEnvironment`. Results are collected as `ConditionResult` per episode and aggregated into `GranularityExperimentResult`.

**Mock evaluation mode:** All M6 results are in mock mode. `MockEnvironment` is deterministic — all three conditions complete at 100% on success scenarios. The informative comparison is orchestration **cost** (policy calls, subtask overhead). Run with `make run-experiment`.

**Failure path:** With `MockScenario.FAIL_AT_STEP` and `max_retries=0`, all three conditions reach `AgentStatus.FAILED`. The retry/replan paths are exercised correctly regardless of planning granularity. Run with `make run-experiment-fail`.

### Requirements for M7 closed-loop comparison

The real question — does finer decomposition improve task success? — cannot be answered in mock mode. The M7 protocol must satisfy:

- **Same test split and episode order** across conditions
- **Same policy checkpoint** (`smolvla_so100_m4`, step 010000, or the base model as an additional baseline)
- **Same action budget** (max 200 steps, matching the longest episode in `svla_so100_pickplace`)
- **Same success predicates** and evaluation mode (offline/replay or simulation)
- **Statistical tests** applied only when n ≥ 10 per condition; small-sample disclaimer included otherwise

**What would constitute a meaningful M7 result:**
- VLA-only vs coarse agentic: a consistent L1 reduction (>10%) or improved subtask completion rate across test episodes supports the orchestration hypothesis.
- Coarse vs fine agentic: a tradeoff — lower error but more LLM calls and higher latency — supports the granularity hypothesis.
- No measurable difference would be an honest null result: SmolVLA's native language understanding may be sufficient for this task without decomposition.

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

## What mock evaluation proves in M5

Milestone 5 adds the full LangGraph StateGraph: `understand_goal → create_plan → select_next_subtask → safety_check → execute_policy → verify_result → diagnose_failure`, with retry/replan/fail terminal paths. All M5 claims are verified in mock mode — no LLM, no GPU, no dataset.

### What M5 proves

| Claim | How it is tested |
|---|---|
| `understand_goal` fails with FAILED status on empty or whitespace-only goals | `test_understand_goal_fails_empty_goal`, `test_understand_goal_fails_whitespace_only` |
| `create_plan` delegates to the injected planner and returns a `TaskPlan` | `test_create_plan_produces_plan`, `test_create_plan_uses_injected_planner` |
| `create_plan` sets FAILED status on `PlanningError` | `test_create_plan_wraps_planning_error` |
| `select_next_subtask` picks the first pending subtask and skips completed ones | `test_select_next_subtask_returns_first_pending`, `test_select_next_subtask_skips_completed` |
| `select_next_subtask` sets COMPLETED when all subtasks are done | `test_select_next_subtask_sets_completed_when_all_done` |
| `safety_check` passes allowed verbs and rejects blocked terms | `test_safety_check_passes_valid_instruction`, `test_safety_check_rejects_blocked_instruction` |
| `safety_check` sets SAFETY_STOP on rejection | `test_safety_check_rejects_blocked_instruction` |
| `diagnose_failure` increments retry_count and subtask.attempt when retries remain | `test_diagnose_increments_retry_when_retries_remain`, `test_diagnose_increments_attempt_on_retry` |
| `diagnose_failure` clears the plan and increments replan_count after max retries | `test_diagnose_triggers_replan_after_max_retries` |
| `diagnose_failure` sets FAILED when both retries and replans are exhausted | `test_diagnose_fails_after_max_retries_and_replans` |
| `DeterministicPlanner` produces 2 subtasks in coarse mode, 5 in fine mode | `test_coarse_plan_has_two_subtasks`, `test_fine_plan_has_five_subtasks` |
| `DeterministicPlanner` raises `PlanningError` for unrecognised goals | `test_unknown_goal_raises_planning_error` |
| `LLMTaskPlanner` parses valid JSON responses into `TaskPlan` | `test_parses_valid_json_response`, `test_subtask_instructions_match_llm_response` |
| `LLMTaskPlanner` raises `PlanningError` on LLM failure, bad JSON, or empty subtask list | `test_raises_on_llm_call_failure`, `test_raises_on_invalid_json`, `test_raises_on_empty_subtask_list` |
| Full compiled graph completes a pick-and-place goal in coarse mode (2 subtasks) | `test_agent_completes_pick_and_place_goal` |
| Full compiled graph completes a goal in fine mode (5 subtasks) | `test_agent_completes_with_fine_granularity` |
| Graph halts with SAFETY_STOP when a planner injects a blocked-keyword subtask | `test_agent_safety_stops_on_blocked_instruction` |
| Graph exhausts retries and replans and returns FAILED | `test_agent_fails_after_exhausting_retries_and_replans` |
| `execution_history_references` is appended by LangGraph for each subtask run | `test_agent_execution_history_is_recorded` |

### What M5 does NOT prove

- That the `DeterministicPlanner`'s subtask vocabulary matches what SmolVLA was trained on. The coarse plan (["approach and grasp", "move and place"]) and fine plan (5 steps) are keyword templates — they are plausible decompositions, not ground-truth instruction labels from `svla_so100_pickplace`. Whether these subtask instructions produce lower action error than the full goal string is the M6 experiment.
- That the `LLMTaskPlanner` produces better plans than `DeterministicPlanner` for this task. It may — but the evidence requires running M6 with both planners against the same recorded episodes.
- That retry and replan logic improves task success in closed-loop execution. The mock tests verify the *software behavior* (correct state transitions) but not the *performance impact* (whether a retry actually leads to success on a real task).
- That the graph executes correctly in replay or VLA mode. M5 and M6 integration tests both use `MockRobotPolicy` and `MockEnvironment`. Connecting the graph to `ReplayRobotPolicy`/`ReplayEnvironment` is M7 work.

All M5 results are labeled `evaluation_mode=MOCK`.

---

## What mock evaluation proves in M6

Milestone 6 adds `VlaOnlyPlanner`, `run_granularity_experiment()`, and the three-condition planning-granularity comparison infrastructure. All M6 claims are verified in mock mode — no LLM, no GPU, no dataset, no robot.

### What M6 proves

| Claim | How it is tested |
|---|---|
| `VlaOnlyPlanner` wraps the full goal as exactly one subtask | `test_vla_only_planner_produces_single_subtask` |
| `VlaOnlyPlanner.plan()` sets the subtask instruction to the full goal text | `test_vla_only_planner_uses_full_goal_as_instruction` |
| `VlaOnlyPlanner.planner_id` is `"vla_only"` | `test_vla_only_planner_id_is_vla_only` |
| `run_granularity_experiment()` returns results for all three conditions | `test_experiment_returns_all_three_conditions` |
| All three conditions complete at 100% on success scenarios (mock is deterministic) | `test_all_conditions_complete_on_success_scenario` |
| VLA-only condition produces exactly 1 planned subtask per episode | `test_vla_only_has_one_subtask` |
| Coarse condition produces exactly 2 planned subtasks per episode | `test_coarse_has_two_subtasks` |
| Fine condition produces exactly 5 planned subtasks per episode | `test_fine_has_five_subtasks` |
| Policy calls equal subtasks planned on a clean success (no retries or replans) | `test_policy_calls_equal_subtasks_on_clean_success` |
| VLA-only has fewer policy calls than coarse, which has fewer than fine | `test_vla_only_has_fewer_policy_calls_than_coarse`, `test_coarse_has_fewer_policy_calls_than_fine` |
| All conditions fail when mock environment never succeeds (max_retries=0) | `test_all_conditions_fail_when_mock_never_succeeds` |
| `mean_subtasks_planned` and `mean_policy_calls` are strictly ordered vla < coarse < fine | `test_mean_subtasks_differ_across_conditions`, `test_mean_policy_calls_differ_across_conditions` |
| `ConditionSummary.completion_rate` is 1.0 on all-success scenarios | `test_condition_summary_completion_rate_is_one_on_success` |
| `GranularityExperimentResult.summary_lines()` includes a disclaimer note | `test_summary_lines_has_evaluation_note` |

### What M6 does NOT prove

- That any planning condition improves real-world task success. `MockEnvironment` is deterministic — it succeeds whenever `succeed_at_step` is reached, regardless of what the policy predicted. All three conditions complete at 100% on success scenarios because the environment is scripted, not because the planning helps.
- That coarse or fine decomposition improves action quality. The policy in all M6 runs is `MockRobotPolicy` (returns constant valid zero actions). Connecting the graph to `ReplayRobotPolicy` or `SmolVLAPolicyAdapter` in offline/replay mode is M7 work.
- That the subtask vocabulary in `DeterministicPlanner` matches SmolVLA's training distribution. Whether the subtask instructions ("approach and grasp", "move and place") produce lower action error than the full goal string requires running the experiment with a real policy checkpoint against real dataset episodes.
- Statistical significance. The default scenario set has N=5–6 episodes. Results are descriptive statistics only. Statistical tests (Wilcoxon) are appropriate only when N ≥ 10 per condition, which requires simulation (M7+).

All M6 results are labeled `evaluation_mode=MOCK`. The informative comparison in mock mode is orchestration **cost** (policy calls, subtask overhead), not success rate.

---

## Anti-patterns to avoid

- Reporting simulation or offline results as "robot performance"
- Tuning evaluation scenarios on the same data used for training
- Omitting the evaluation mode label from any published metric
- Comparing conditions that used different seeds, checkpoints, or splits
- Claiming recovery performance when recovery was never triggered in evaluation
