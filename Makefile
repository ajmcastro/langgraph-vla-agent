# =============================================================================
# LangGraph VLA Agent — Developer Makefile
# All targets use `uv run` so the project venv is always active.
# GPU, simulator, network, and hardware targets are marked [OPTIONAL] and
# require additional setup (see docs/data.md, docs/evaluation.md).
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help setup sync format lint typecheck test test-unit test-integration \
        check inspect-data evaluate-mock evaluate-replay evaluate-policy \
        evaluate-agent train run-demo clean

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

## inspect-data: [OPTIONAL] Inspect dataset metadata without full download
inspect-data:
	@echo "Milestone 2 target — not yet implemented"

# ---------------------------------------------------------------------------
# Evaluation (Milestone 3+)
# ---------------------------------------------------------------------------

## evaluate-mock: Run mock/deterministic evaluation scenarios
evaluate-mock:
	@echo "Milestone 3 target — not yet implemented"

## evaluate-replay: [OPTIONAL] Offline replay evaluation on held-out episodes
evaluate-replay:
	@echo "Milestone 3 target — not yet implemented"

## evaluate-policy: [OPTIONAL] VLA policy evaluation (requires vla extra + model checkpoint)
evaluate-policy:
	@echo "Milestone 3 target — not yet implemented"

## evaluate-agent: [OPTIONAL] Full agent evaluation (requires agent + vla extras)
evaluate-agent:
	@echo "Milestone 5 target — not yet implemented"

# ---------------------------------------------------------------------------
# Training (Milestone 4+, Cloud GPU required)
# ---------------------------------------------------------------------------

## train: [OPTIONAL][CLOUD GPU] Fine-tune SmolVLA on the selected dataset
train:
	@echo "Milestone 4 target — requires cloud GPU and VLA extra"
	@echo "See docs/data.md and docs/evaluation.md for setup instructions"

# ---------------------------------------------------------------------------
# Demo (Milestone 8+)
# ---------------------------------------------------------------------------

## run-demo: Run the interactive agent demo
run-demo:
	@echo "Milestone 8 target — not yet implemented"

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
