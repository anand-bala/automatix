SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

EXTRA_UV_FLAGS =
# If we want to use CUDA, the CUDA_VERSION variable should not be empty and contain the
# version number (either 12 or 13)
CUDA_VERSION ?=

_DEV_TARGETS = uv.lock
ifneq ($(CUDA_VERSION),)
_DEV_TARGETS += cuda-packages
endif

# Default: create the dev environment
dev: $(_DEV_TARGETS) | .venv
.PHONY: dev

fmt:
	uv run --frozen ruff format
	uv run --frozen ruff check --fix --exit-non-zero-on-fix .
.PHONY: fmt

type-check:
	uv run --frozen ty check
	uv run --frozen zmypy
.PHONY: type-check

lint: fmt type-check
.PHONY: lint

test:
	uv run --dev --frozen pytest
.PHONY: test

hscc25experiments: ./examples/swarm-monitoring/run_hscc_experiments.py
	uv run --group examples --script $<

uv.lock .venv &: pyproject.toml
	uv sync --all-packages --frozen --inexact --dev

export CUDA_VERSION
cuda-packages: uv.lock | .venv
	uv pip install jax[cuda$(CUDA_VERSION)]
.PHONY: cuda-packages

# Automatic make target for scripts with locking
%.py.lock: %.py
	uv lock --script $<
