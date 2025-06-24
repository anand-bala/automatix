SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

EXTRA_UV_FLAGS =
USE_CUDA ?=
# If we want to use CUDA, the USE_CUDA variable should not be empty
ifneq (,$(USE_CUDA))
	EXTRA_UV_FLAGS += --extra cuda
endif

# Default: create the dev environment
dev: uv.lock | .venv
.PHONY: dev

lint:
	uv run --frozen ruff format 
	uv run --frozen ruff check --fix --exit-non-zero-on-fix .
	uv run --frozen --all-groups --extra polynomials  mypy src examples
.PHONY: lint

test:
	uv run --dev --frozen pytest
.PHONY: test

docs:
	PYTHONPATH=src uv run --dev mkdocs build
.PHONY: docs

serve-docs:
	PYTHONPATH=src uv run --dev mkdocs serve
.PHONY: serve-docs

hscc25experiments: ./examples/swarm-monitoring/run_hscc_experiments.py
	uv run --group examples --script $<

uv.lock .venv &: pyproject.toml
	uv sync --frozen --dev ${EXTRA_UV_FLAGS}

# Automatic make target for scripts with locking
%.py.lock: %.py
	uv lock --script $<
