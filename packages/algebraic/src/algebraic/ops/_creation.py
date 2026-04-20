"""Backend-agnostic array creation functions for `AlgebraicArray`.

All backend-specific imports are lazy (inside function bodies) so that
importing this module never triggers `jax`, `torch`, or `numpy` imports.
"""

from __future__ import annotations

from collections.abc import Iterable

from algebraic.array.base import AlgebraicArray
from algebraic.spec import Semiring
from algebraic.types import Array, Backend, Number, is_array, is_scalar


def array(
    data: Array | Number | Iterable[Array | Number],
    *,
    semiring: Semiring,
    backend: str | Backend | None = None,
    device: object | None = None,
) -> AlgebraicArray:
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
    device : object or None, optional
        Target device for the array (e.g. ``torch.device("cuda:0")``).
        If ``None``, the backend's default device is used.

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` backed by the appropriate backend.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> algebraic.array([1.0, 2.0], semiring=sr, backend="numpy")
    AlgebraicArray(data=array([1., 2.], dtype=float32), semiring=Semiring(...
    """
    if backend is None:
        if not is_array(data) and not is_scalar(data):
            assert isinstance(data, Iterable)
            # Must be an iterable...
            data = list(data)
            backend = Backend.from_array(data[0])
        else:
            backend = Backend.from_array(data)
    backend = Backend(backend)
    if backend == Backend.JAX:
        import jax.numpy as jnp

        result = AlgebraicArray(jnp.asarray(data, dtype=jnp.float32), semiring)
        if device is not None:
            result = result.to_device(device)
        return result
    elif backend == Backend.TORCH:
        import torch

        kwarg: dict[str, object] = dict(dtype=torch.float32)
        if device is not None:
            kwarg["device"] = device

        t = torch.as_tensor(data, **kwarg)  # type: ignore[arg-type]
        return AlgebraicArray(t, semiring)
    elif backend == Backend.NUMPY:
        import numpy as np

        return AlgebraicArray(np.asarray(data, dtype=np.float32), semiring)
    raise ValueError(f"Unsupported backend: {backend!r}")


def zeros(
    shape: tuple[int, ...], *, semiring: Semiring, backend: str | Backend, device: object | None = None
) -> AlgebraicArray:
    """Create an :class:`AlgebraicArray` filled with the semiring's additive identity.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array.
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).
    device : object or None, optional
        Target device for the array.  If ``None``, uses the backend default.

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with ``semiring.zero``.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> algebraic.zeros((2,), semiring=sr, backend="numpy").data
    array([inf, inf], dtype=float32)
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
    return array(data, semiring=semiring, backend=b, device=device)


def ones(
    shape: tuple[int, ...], *, semiring: Semiring, backend: str | Backend, device: object | None = None
) -> AlgebraicArray:
    """Create an :class:`AlgebraicArray` filled with the semiring's multiplicative identity.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array.
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).
    device : object or None, optional
        Target device for the array.  If ``None``, uses the backend default.

    Returns
    -------
    AlgebraicArray
        A concrete :class:`AlgebraicArray` filled with ``semiring.one``.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> algebraic.ones((2,), semiring=sr, backend="numpy").data
    array([0., 0.], dtype=float32)
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
    return array(data, semiring=semiring, backend=b, device=device)


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
    return zeros(arr.shape, semiring=arr.semiring, backend=backend, device=arr.device)


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
    return ones(arr.shape, semiring=arr.semiring, backend=backend, device=arr.device)


def full(
    shape: tuple[int, ...],
    fill_value: Number,
    *,
    semiring: Semiring,
    backend: str | Backend,
    device: object | None = None,
) -> AlgebraicArray:
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
    device : object or None, optional
        Target device for the array.  If ``None``, uses the backend default.

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
    return array(data, semiring=semiring, backend=b, device=device)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    semiring: Semiring,
    backend: str | Backend,
    k: int = 0,
    device: object | None = None,
) -> AlgebraicArray:
    """Returns a two-dimensional array with ones on the ``k``th diagonal and zeros elsewhere.

    Parameters
    ----------

    n_rows : int
        number of rows in the output array.
    n_cols : int or None, optional
        number of columns in the output array. If ``None``, the default number of
        columns in the output array is equal to ``n_rows``. Default: ``None``.
    k : int
        index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal. (default = ``0``)
    semiring : Semiring
        Semiring for the algebraic structure.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).
    device : object or None, optional
        Target device for the array.  If ``None``, uses the backend default.

    """
    backend = Backend(backend)
    xp = backend.get_array_namespace()

    mask: Array = xp.eye(n_rows, n_cols, k=k)

    result: Array = xp.where(mask > 0, xp.asarray(semiring.one), xp.asarray(semiring.zero))
    return array(result, semiring=semiring, backend=backend, device=device)
