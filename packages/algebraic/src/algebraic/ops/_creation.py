"""Backend-agnostic array creation functions for `AlgebraicArray`.

All backend-specific imports are lazy (inside function bodies) so that
importing this module never triggers `jax`, `torch`, or `numpy` imports.
"""

from __future__ import annotations

import typing

from algebraic.spec import Semiring
from algebraic.types import Array, Backend, Number

if typing.TYPE_CHECKING:
    from algebraic.array.base import AlgebraicArray


def array(data: Array, *, semiring: Semiring, backend: str | Backend | None = None) -> AlgebraicArray:
    """Create an `AlgebraicArray` from an existing backend array.

    Args:
        data: Backend array (`jax.Array`, `torch.Tensor`, or `numpy.ndarray`).
        semiring: Semiring for the algebraic structure.
        backend: Backend to use. If `None`, auto-detected from `data`.

    Returns:
        A concrete `AlgebraicArray` backed by the appropriate backend.
    """
    if backend is None:
        backend = Backend.from_array(data)
    backend = Backend(backend)
    if backend == Backend.JAX:
        import jax.numpy as jnp

        from algebraic.array._jax import JaxAlgebraicArray

        return JaxAlgebraicArray(jnp.asarray(data), semiring)  # ty: ignore[too-many-positional-arguments]
    elif backend == Backend.TORCH:
        import torch

        from algebraic.array._torch import TorchAlgebraicArray

        return TorchAlgebraicArray(torch.asarray(data), semiring)
    elif backend == Backend.NUMPY:
        import numpy as np

        from algebraic.array._numpy import NumpyAlgebraicArray

        return NumpyAlgebraicArray(np.asarray(data), semiring)
    raise ValueError(f"Unsupported backend: {backend!r}")


def zeros(shape: tuple[int, ...], *, semiring: Semiring, backend: str | Backend) -> AlgebraicArray:
    """Create an `AlgebraicArray` filled with the semiring's additive identity.

    Args:
        shape: Shape of the output array.
        semiring: Semiring for the algebraic structure.
        backend: Backend to use (`"jax"`, `"torch"`, or `"numpy"`).

    Returns:
        A concrete `AlgebraicArray` filled with `semiring.zero`.
    """
    b = Backend(backend)
    data: Array
    match b:
        case Backend.JAX:
            import jax.numpy as jnp

            data = jnp.full(shape, jnp.asarray(semiring.zero))

        case Backend.TORCH:
            import torch

            data = torch.full(shape, semiring.zero)
        case Backend.NUMPY:
            import numpy as np

            data = np.full(shape, semiring.zero)
        case _:
            raise ValueError(f"Unsupported backend: {b!r}")
    return array(data, semiring=semiring, backend=b)


def ones(shape: tuple[int, ...], *, semiring: Semiring, backend: str | Backend) -> AlgebraicArray:
    """Create an `AlgebraicArray` filled with the semiring's multiplicative identity.

    Args:
        shape: Shape of the output array.
        semiring: Semiring for the algebraic structure.
        backend: Backend to use (`"jax"`, `"torch"`, or `"numpy"`).

    Returns:
        A concrete `AlgebraicArray` filled with `semiring.one`.
    """
    semiring = semiring
    b = Backend(backend)
    data: Array
    match b:
        case Backend.JAX:
            import jax.numpy as jnp

            data = jnp.full(shape, jnp.asarray(semiring.one))

        case Backend.TORCH:
            import torch

            data = torch.full(shape, semiring.one)
        case Backend.NUMPY:
            import numpy as np

            data = np.full(shape, semiring.one)
        case _:
            raise ValueError(f"Unsupported backend: {b!r}")
    return array(data, semiring=semiring, backend=b)


def zeros_like(arr: AlgebraicArray) -> AlgebraicArray:
    """Create an `AlgebraicArray` of zeros with the same shape and backend as *arr*.

    Args:
        arr: Source `AlgebraicArray` whose shape, semiring, and backend are used.

    Returns:
        A concrete `AlgebraicArray` filled with `arr.semiring.zero`.
    """
    from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

    assert isinstance(arr, BaseAlgebraicArray)
    backend = Backend.from_array(arr.data)
    return zeros(arr.shape, semiring=arr.semiring, backend=backend)


def ones_like(arr: AlgebraicArray) -> AlgebraicArray:
    """Create an `AlgebraicArray` of ones with the same shape and backend as *arr*.

    Args:
        arr: Source `AlgebraicArray` whose shape, semiring, and backend are used.

    Returns:
        A concrete `AlgebraicArray` filled with `arr.semiring.one`.
    """
    from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

    assert isinstance(arr, BaseAlgebraicArray)
    backend = Backend.from_array(arr.data)
    return ones(arr.shape, semiring=arr.semiring, backend=backend)


def full(shape: tuple[int, ...], fill_value: Number, *, semiring: Semiring, backend: str | Backend) -> AlgebraicArray:
    """Create an `AlgebraicArray` filled with *fill_value*.

    Args:
        shape: Shape of the output array.
        fill_value: Value to fill the array with.
        semiring: Semiring for the algebraic structure.
        backend: Backend to use (`"jax"`, `"torch"`, or `"numpy"`).

    Returns:
        A concrete `AlgebraicArray` filled with *fill_value*.
    """
    semiring = semiring
    b = Backend(backend)

    data: Array
    if b == Backend.JAX:
        import jax.numpy as jnp

        data = jnp.full(shape, fill_value)
    elif b == Backend.TORCH:
        import torch

        data = torch.full(shape, fill_value)
    elif b == Backend.NUMPY:
        import numpy as np

        data = np.full(shape, fill_value)
    else:
        raise ValueError(f"Unsupported backend: {b!r}")
    return array(data, semiring=semiring, backend=b)
