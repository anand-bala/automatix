"""Shared fixtures and helpers for algebraic tests."""

from __future__ import annotations

import pytest
from algebraic.types import Backend

BACKENDS = list(str(b) for b in Backend)


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture that yields each backend name, skipping if unavailable."""
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture
def jax_backend() -> str:
    """Fixture that yields 'jax', skipping if JAX is not installed."""
    pytest.importorskip("jax")
    return "jax"
