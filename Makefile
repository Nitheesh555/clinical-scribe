# Clinical Scribe — developer tasks.
# Usage: make <target>. On Windows, run under Git Bash or WSL.

.PHONY: help install install-dev lint format typecheck test smoke lock prepare clean

help:
	@echo "install      install runtime + train/eval extras"
	@echo "install-dev  install dev tooling (lint, test, pre-commit)"
	@echo "lint         ruff + black --check"
	@echo "format       ruff --fix + black"
	@echo "typecheck    mypy on src/"
	@echo "test         pytest (excludes gpu-marked tests)"
	@echo "smoke        run the 2-step smoke-train test (needs GPU)"
	@echo "lock         freeze resolved versions to requirements.lock (run on target runtime)"
	@echo "prepare      build JSONL datasets from config"

install:
	pip install -e ".[train,eval]"

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests scripts
	black --check src tests scripts

format:
	ruff check --fix src tests scripts
	black src tests scripts

typecheck:
	mypy

test:
	pytest -m "not gpu"

smoke:
	pytest -m "slow and gpu" -s

lock:
	pip freeze > requirements.lock
	@echo "Wrote requirements.lock — commit this for reproducibility."

prepare:
	python scripts/prepare_data.py --config configs/phase1_t4.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
