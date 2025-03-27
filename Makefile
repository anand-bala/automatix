SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

USE_CUDA ?= $(shell type -p "nvidia-smi")
# If we want to use CUDA, the USE_CUDA variable should not be empty
ifneq (,$(USE_CUDA))
	EXTRA_UV_FLAGS = --extra cuda
else
	EXTRA_UV_FLAGS = 
endif

# Default: create the dev environment
dev: uv.lock | .venv
.PHONY: dev

lint:
	uvx ruff format 
	uvx ruff check --fix --exit-non-zero-on-fix .
	uvx mypy src examples
.PHONY: lint

test:
	uv run --dev pytest
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
