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
1. VLA-only (no orchestration)
2. Coarse agentic plan (few meaningful subtasks)
3. Fine agentic plan (smaller physical skills)

All conditions must use:
- The same scenario set and seeds / replay split
- The same policy checkpoint
- The same action budget (max steps per subtask / episode)
- The same success predicates
- The same evaluation mode

Results are reported with mean ± std and sample size. Statistical tests are applied when n ≥ 10 per condition. The small sample disclaimer is included when n < 10.

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

## Anti-patterns to avoid

- Reporting simulation or offline results as "robot performance"
- Tuning evaluation scenarios on the same data used for training
- Omitting the evaluation mode label from any published metric
- Comparing conditions that used different seeds, checkpoints, or splits
- Claiming recovery performance when recovery was never triggered in evaluation
