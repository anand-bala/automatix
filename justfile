# Set shell options for safety

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# If we want to use CUDA, the CUDA_VERSION variable should not be empty and contain the
# version number (either 12 or 13)

CUDA_VERSION := env("CUDA_VERSION", "")

# Default: create the dev environment
default: dev

# Set up development environment
dev: sync-venv cuda-packages-if-needed

# Format and lint code
[no-cd]
fmt:
    uv run --frozen ruff format
    uv run --frozen ruff check --output-format concise --fix --exit-non-zero-on-fix 

# Run type checkers
[no-cd]
[private]
ty-check:
    uv run --frozen ty check --output-format concise

[no-cd]
[private]
pyrefly-check:
    uv run --frozen pyrefly check --output-format min-text

[no-cd]
mypy-check:
    uv run --frozen mypy --strict

[parallel]
type-check: ty-check pyrefly-check

# Run both formatting and type checking
[no-cd]
lint: fmt type-check

# Run tests
[no-cd]
test:
    uv run --dev --frozen pytest --lf

# Run HSCC25 experiments
hscc25experiments:
    uv run --group examples --script ./examples/swarm-monitoring/run_hscc_experiments.py

# Sync virtual environment
sync-venv:
    uv sync --all-packages --frozen --inexact --dev

# Install CUDA packages if CUDA_VERSION is set
[private]
cuda-packages-if-needed:
    #!/usr/bin/env bash
    set -eu -o pipefail
    if [[ -n "{{ CUDA_VERSION }}" ]]; then
        uv pip install "jax[cuda{{ CUDA_VERSION }}]"
    fi

# Install CUDA packages (call directly if needed)
cuda-packages:
    uv pip install "jax[cuda{{ CUDA_VERSION }}]"

# Lock a Python script's dependencies
lock-script script:
    uv lock --script {{ script }}

# Release workflow

@bump-version package *args:
    #!/usr/bin/env bash
    set -euo pipefail
    read -p "Are you sure? [y/n] " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      uv version --directory packages/{{ package }} --bump {{ args }}
    fi

@tag-package package:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Will run: git tag $(printf "{{ package }}-v%s" $(uv version --short))"
    read -p "Are you sure? [y/n] " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      git tag $(printf "{{ package }}-v%s" $(uv version --short))
    fi
