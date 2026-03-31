set unstable := true

# If we want to use CUDA, the CUDA_VERSION variable should not be empty and contain the
# version number (either 12 or 13)

CUDA_VERSION := env("CUDA_VERSION", "")

# Default: create the dev environment
default: dev

# Set up development environment
dev: sync-venv cuda-packages

# Format and lint code
[no-cd]
fmt:
    uv run --dev --frozen ruff format
    uv run --dev --frozen ruff check --output-format concise --fix --exit-non-zero-on-fix 

# Run type checkers
[no-cd]
[private]
ty-check:
    uv run --dev --frozen ty check --output-format concise

[no-cd]
[private]
pyrefly-check:
    uv run --dev --frozen pyrefly check --output-format min-text

[no-cd]
mypy-check:
    uv run --dev --frozen mypy --strict

[parallel]
type-check: ty-check pyrefly-check mypy-check

# Run both formatting and type checking
[no-cd]
lint: fmt type-check

# Run tests
[no-cd]
test:
    uv run --dev --frozen pytest --lf

# Sync virtual environment
sync-venv:
    uv sync --all-packages --frozen --inexact --all-groups --all-extras

[private]
_jax_extra := if which("nvidia-smi") == "" { "cpu" } else { f"cuda{{CUDA_VERSION}}" }

# Install CUDA packages (call directly if needed)
cuda-packages:
    uv pip install "jax[{{ _jax_extra }}]" torch --torch-backend=auto

# Build Sphinx HTML docs for a subproject (or all if none specified)
[arg("format", long="format", short="f")]
docs project format="html":
    uv run --dev --group docs --frozen sphinx-build -b "{{ format }}" "docs/{{ project }}" "docs/_build/html/{{ project }}"

all-docs format="html": (docs "algebraic" format) (docs "morphata" format) (docs "automatix" format)

# Upload a generated HDF5 dataset to Hugging Face Hub.
# Requires authentication: run `uv run --dev --frozen hf auth login` or set HF_TOKEN.
# Usage: just upload-dataset <dataset_path> <hf_repo_id> [path_in_repo]
# Example: just upload-dataset data/trivial-21k.hdf5 myuser/afa-dataset

[no-cd]
upload-dataset dataset_path hf_repo_id="anand-bala/automata-embeddings" path_in_repo=(shell("basename $1", dataset_path)):
    uv run --dev --frozen hf upload "{{ hf_repo_id }}" "{{ dataset_path }}" "{{ path_in_repo }}" --type dataset
