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


def array(data: Array | Number, *, semiring: Semiring, backend: str | Backend | None = None) -> AlgebraicArray:
    """Create an :class:`AlgebraicArray` from an existing backend array.

    Parameters
    ----------
    data : Array or Number
        Backend array (``numpy.ndarray``, ``jax.Array``, or ``torch.Tensor``)
        or a Python number. Defaults to the NumPy backend when *backend* is
        ``None`` and *data* is a plain number.
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend or None, optional
        Backend to use. If ``None``, auto-detected from *data*.

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` backed by the appropriate backend.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> algebraic.array([1.0, 2.0], semiring=sr, backend="numpy")  # doctest: +SKIP
    AlgebraicArray([1., 2.], semiring=Semiring(...))
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
    """Create an :class:`AlgebraicArray` filled with the semiring's additive identity.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array.
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with ``semiring.zero``.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> algebraic.zeros((2,), semiring=sr, backend="numpy").data
    array([inf, inf])
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
    """Create an :class:`AlgebraicArray` filled with the semiring's multiplicative identity.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array.
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with ``semiring.one``.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> algebraic.ones((2,), semiring=sr, backend="numpy").data
    array([0., 0.])
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
    """Create an :class:`AlgebraicArray` of zeros with the same shape and backend as *arr*.

    Parameters
    ----------
    arr : AlgebraicArray
        Source array whose shape, semiring, and backend are used.

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with ``arr.semiring.zero``.
    """
    from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

    assert isinstance(arr, BaseAlgebraicArray)
    backend = Backend.from_array(arr.data)
    return zeros(arr.shape, semiring=arr.semiring, backend=backend)


def ones_like(arr: AlgebraicArray) -> AlgebraicArray:
    """Create an :class:`AlgebraicArray` of ones with the same shape and backend as *arr*.

    Parameters
    ----------
    arr : AlgebraicArray
        Source array whose shape, semiring, and backend are used.

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with ``arr.semiring.one``.
    """
    from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

    assert isinstance(arr, BaseAlgebraicArray)
    backend = Backend.from_array(arr.data)
    return ones(arr.shape, semiring=arr.semiring, backend=backend)


def full(shape: tuple[int, ...], fill_value: Number, *, semiring: Semiring, backend: str | Backend) -> AlgebraicArray:
    """Create an :class:`AlgebraicArray` filled with *fill_value*.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array.
    fill_value : Number
        Value to fill the array with.
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with *fill_value*.
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
