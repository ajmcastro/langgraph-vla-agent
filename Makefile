# =============================================================================
# LangGraph VLA Agent — Developer Makefile
# All targets use `uv run` so the project venv is always active.
# GPU, simulator, network, and hardware targets are marked [OPTIONAL] and
# require additional setup (see docs/data.md, docs/evaluation.md).
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help setup sync setup-datasets setup-agent setup-vla \
        format lint typecheck test test-unit test-integration \
        check inspect-data evaluate-mock evaluate-replay evaluate-policy \
        evaluate-agent evaluate-agent-fine run-experiment run-experiment-fail \
        run-simulation run-simulation-hard \
        train train-cloud validate-train-config \
        compare-checkpoints compare-checkpoints-vla run-demo clean

PYTHON := uv run python
UV     := uv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

## setup: Install core + dev deps (run once after cloning)
setup:
	$(UV) sync --extra dev

## sync: Sync venv to pyproject.toml without optional extras
sync:
	$(UV) sync

## setup-agent: Install core + dev + LangGraph agent deps (Milestone 5+)
setup-agent:
	$(UV) sync --extra dev --extra agent

## setup-datasets: Install dataset inspection deps (huggingface_hub; no GPU needed)
setup-datasets:
	$(UV) sync --extra dev --extra datasets

## setup-vla: [OPTIONAL] Install VLA/LeRobot deps — large download, needs GPU for training
setup-vla:
	$(UV) sync --extra dev --extra agent --extra vla
	@echo "NOTE: For macOS/Apple Silicon, run: brew install ffmpeg"
	@echo "NOTE: For Linux CUDA, override torch: uv pip install --torch-backend cu128 lerobot"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

## format: Auto-format with ruff
format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

## lint: Lint with ruff (no auto-fix)
lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

## typecheck: Static type checking with mypy
typecheck:
	$(UV) run mypy src

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

## test-unit: Fast unit tests (no external deps, no GPU, no robot)
test-unit:
	$(UV) run pytest tests/unit -m "not integration and not simulation and not hardware"

## test-integration: Integration tests (may use disk/network)
test-integration:
	$(UV) run pytest tests/integration

## test: All non-hardware tests
test:
	$(UV) run pytest tests/unit tests/integration -m "not hardware"

## check: Full local quality gate (format + lint + typecheck + unit tests)
check: lint typecheck test-unit

# ---------------------------------------------------------------------------
# Data (Milestone 2+)
# ---------------------------------------------------------------------------

## inspect-data: [OPTIONAL/NETWORK] Inspect dataset metadata without downloading data
inspect-data:
	$(PYTHON) scripts/inspect_dataset.py

# ---------------------------------------------------------------------------
# Evaluation (Milestone 3+)
# ---------------------------------------------------------------------------

## evaluate-mock: Offline evaluation with MockRobotPolicy (no VLA extra needed)
evaluate-mock:
	$(PYTHON) scripts/evaluate_policy.py --mode mock

## evaluate-replay: Offline evaluation with ReplayRobotPolicy on fixture episodes
evaluate-replay:
	$(PYTHON) scripts/evaluate_policy.py --mode replay

## evaluate-policy: [OPTIONAL] VLA policy evaluation (requires vla extra + model download)
evaluate-policy:
	$(PYTHON) scripts/evaluate_policy.py --mode vla

## evaluate-agent: Run LangGraph agent in mock mode (requires agent extra: make setup-agent)
evaluate-agent:
	$(PYTHON) scripts/run_agent.py --granularity coarse

## evaluate-agent-fine: Run LangGraph agent with fine-grained subtask decomposition
evaluate-agent-fine:
	$(PYTHON) scripts/run_agent.py --granularity fine

## run-experiment: Run M6 planning-granularity experiment (requires agent extra: make setup-agent)
run-experiment:
	$(PYTHON) scripts/run_experiment.py

## run-experiment-fail: Run M6 experiment with a failure scenario to see retry/replan
run-experiment-fail:
	$(PYTHON) scripts/run_experiment.py --fail-scenario

## run-simulation: Run M7 simulation experiment — easy mode (all conditions succeed)
run-simulation:
	$(PYTHON) scripts/run_simulation.py

## run-simulation-hard: Run M7 simulation experiment — hard mode (VLA-only fails, agentic succeeds)
run-simulation-hard:
	$(PYTHON) scripts/run_simulation.py --hard

# ---------------------------------------------------------------------------
# Training (Milestone 4+, Cloud GPU required)
# ---------------------------------------------------------------------------

## validate-train-config: Validate training YAML without running training
validate-train-config:
	$(PYTHON) scripts/train_smolvla.py --dry-run

## train: [OPTIONAL][GPU] Fine-tune SmolVLA locally (requires GPU + vla extra)
train:
	$(PYTHON) scripts/train_smolvla.py

## train-cloud: [OPTIONAL][CLOUD GPU] Submit SmolVLA fine-tuning to HF Jobs
train-cloud:
	$(PYTHON) scripts/train_smolvla.py --config configs/training/smolvla_so100.yaml

## compare-checkpoints: Compare base vs fine-tuned checkpoint (fixture mode, no GPU)
compare-checkpoints:
	$(PYTHON) scripts/compare_checkpoints.py --mode fixture

## compare-checkpoints-vla: [OPTIONAL][GPU] Compare base vs fine-tuned with real VLA models
compare-checkpoints-vla:
	$(PYTHON) scripts/compare_checkpoints.py \
		--mode vla \
		--base lerobot/smolvla_base \
		--finetuned artifacts/training/smolvla_so100_m4/checkpoints/010000/pretrained_model

# ---------------------------------------------------------------------------
# Demo (Milestone 8+)
# ---------------------------------------------------------------------------

## run-demo: Run the M8 portfolio demo (replay + mock agent + simulation, requires setup-agent)
run-demo:
	$(PYTHON) scripts/run_demo.py

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

## clean: Remove generated artifacts and caches
clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

## help: Show this help
help:
	@grep -E '^## ' Makefile | sed 's/## //' | column -t -s ':'
