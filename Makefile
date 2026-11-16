.DEFAULT_GOAL := help
.PHONY: help setup sync lock upgrade list run test lint format check clean clean-all

# uv installer drops the binary here; make must find it right after install
export PATH := $(HOME)/.local/bin:$(PATH)

# Optional target for run/format/lint, e.g. FILE=03_chunking_strategies/chunking_strategies.py
# Defaults to the whole project (or the first lesson for `run`).
FILE ?=
TARGET := $(or $(FILE),.)

help: ## Show targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
	@echo ""
	@echo "  Examples:"
	@echo "    make run FILE=03_chunking_strategies/chunking_strategies_ollama.py"
	@echo "    make format FILE=03_chunking_strategies/chunking_strategies_ollama.py"
	@echo "    make lint FILE=02_embeddings_similarity"

setup: ## First-time setup: install uv if missing, Python, venv, deps
	@command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install
	uv sync

sync: ## Install deps exactly as locked
	uv sync --locked

lock: ## Refresh uv.lock after editing pyproject.toml
	uv lock

upgrade: ## Upgrade all deps to latest allowed
	uv lock --upgrade
	uv sync

list: ## List runnable Python scripts
	@find . -name '*.py' -not -path './.venv/*' | sed 's|^\./||' | sort

run: ## Run a script: make run FILE=path/to/script.py
	@test -f "$(or $(FILE),01_rag_explained/rag_explained.py)" || { echo "No such file: $(FILE) (try: make list)"; exit 1; }
	uv run python $(or $(FILE),01_rag_explained/rag_explained.py)

test: ## Run pytest (exit 5 = no tests collected, not a failure)
	uv run pytest || [ $$? -eq 5 ]

lint: ## Check with ruff + mypy (optional FILE=)
	uv run ruff check $(TARGET)
	uv run ruff format --check $(TARGET)
	uv run mypy $(FILE)

format: ## Auto-fix and format with ruff (optional FILE=)
	uv run ruff check --fix $(TARGET)
	uv run ruff format $(TARGET)

check: lint test ## Lint and test

clean: ## Delete caches, __pycache__, notebook checkpoints, .DS_Store
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -name .DS_Store -type f -exec rm -f {} +

clean-all: clean ## Also delete .venv and uv.lock (run `make setup` after)
	rm -rf .venv uv.lock
