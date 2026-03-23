"""Shared testing utils for algebraic"""

from typing import Any

import numpy as np

from algebraic import AlgebraicArray
from algebraic.types import Array, Backend


def maybe_unwrap(x: AlgebraicArray | Array | Any) -> np.ndarray:  # noqa: ANN401
    if isinstance(x, AlgebraicArray):
        x = x.data
    return np.asanyarray(x)


def assert_allclose(actual: AlgebraicArray, desired: AlgebraicArray, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    assert actual.semiring == desired.semiring, f"Semirings differ: {actual.semiring} != {desired.semiring}"
    assert actual._vdot == desired._vdot and actual._matmul == desired._matmul

    np.testing.assert_allclose(actual.data, desired.data, rtol=rtol, atol=atol)


def assert_close(a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:  # noqa: ANN401
    """Assert two arrays are element-wise close after converting to NumPy."""
    np.testing.assert_allclose(maybe_unwrap(a), maybe_unwrap(b), rtol=rtol, atol=atol)


def assert_equal(a: Any, b: Any) -> None:  # noqa: ANN401
    """Assert two arrays are element-wise equal after converting to NumPy."""
    np.testing.assert_array_equal(maybe_unwrap(a), maybe_unwrap(b))


def make_array(value: Any, backend: str | Backend | None = None) -> Array:  # noqa: ANN401
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(value)
    if backend == "torch":
        import torch

        return torch.as_tensor(value)
    # Default fallback
    return np.asarray(value)
