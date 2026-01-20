# Set shell options for safety
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# If we want to use CUDA, the CUDA_VERSION variable should not be empty and contain the
# version number (either 12 or 13)
CUDA_VERSION := env_var_or_default("CUDA_VERSION", "")

# Default: create the dev environment
default: dev

# Set up development environment
dev: sync-venv cuda-packages-if-needed

# Format and lint code
[no-cd]
fmt:
    uv run --frozen ruff format
    uv run --frozen ruff check --output-format concise --fix --exit-non-zero-on-fix .

# Run type checkers
[no-cd]
type-check:
    uv run --frozen ty check --output-format concise
    uv run --frozen mypy --strict .

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
tag-package package:
    echo git tag $(printf "{{ package }}-v%s" $(uv version --directory packages/{{ package }} --short))
