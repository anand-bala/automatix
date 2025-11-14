SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
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

ruff-check:
	uv run --frozen ruff format
	uv run --frozen ruff check --fix --exit-non-zero-on-fix .
.PHONY: fmt

type-check:
	uv run --frozen zmypy
.PHONY: type-check

lint: ruff-check type-check
.PHONY: lint

test:
	uv run --dev --frozen pytest
.PHONY: test

hscc25experiments: ./examples/swarm-monitoring/run_hscc_experiments.py
	uv run --group examples --script $<

uv.lock .venv &: pyproject.toml
	uv sync --all-packages --frozen --dev ${EXTRA_UV_FLAGS}

# Automatic make target for scripts with locking
%.py.lock: %.py
	uv lock --script $<
