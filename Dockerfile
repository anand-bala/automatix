FROM ghcr.io/astral-sh/uv:debian as builder

# Enable bytecode compilation
# Copy from the cache instead of linking since it's a mounted volume
# Only use the managed Python version
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed

# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR /python

# Install Python before the project for caching
RUN uv python install 3.12

# Install the project into `/app`
WORKDIR /app
COPY uv.lock pyproject.toml /app/

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-groups

ADD . /app
# Installing separately from its dependencies allows optimal layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups

# Then, use a final image without uv
FROM debian:bookworm-slim

# Copy the Python version
COPY --from=builder /python /python

# Copy the application from the builder
COPY --from=builder /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "/app/examples/swarm-monitoring/run_hscc_experiments.py"]
