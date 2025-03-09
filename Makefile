SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# Default: create the dev environment
uv.lock: pyproject.toml
	uv sync --dev

# Automatic make target for scripts with locking
%.py.lock: %.py
	uv lock --script $<

lint:
	uvx ruff format 
	uvx ruff check --fix --exit-non-zero-on-fix .
	uvx mypy src examples
.PHONY: lint

test:
	uvx pytest
.PHONY: test

docs:
	PYTHONPATH=src uvx mkdocs build
.PHONY: docs

serve-docs:
	PYTHONPATH=src mkdocs serve
.PHONY: serve-docs

hscc25experiments: ./examples/swarm-monitoring/run_hscc_experiments.py
	uv run --group examples --script $<
