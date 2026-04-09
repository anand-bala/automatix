"""Pass-through array manipulation operations for AlgebraicArray.

All functions in this module operate on the underlying data via
`array_api_compat` and wrap the result back into an `AlgebraicArray`
without involving semiring operations. Operations that take multiple
`AlgebraicArray` inputs validate that they share the same semiring instance
before proceeding.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

import array_api_compat

from algebraic.array.base import AlgebraicArray
from algebraic.types import Array, Number
from algebraic.utils import maybe_unwrap, validate_semiring


class UniqueAllResult(typing.NamedTuple):
    """Return type for :func:`unique_all`."""

    values: AlgebraicArray
    """Unique values (wrapped AlgebraicArray)."""
    indices: Array
    """Indices of the first occurrence of each unique value in the input."""
    inverse_indices: Array
    """Indices such that ``values[inverse_indices] == x.ravel()``."""
    counts: Array
    """Number of times each unique value appears."""


class UniqueCountsResult(typing.NamedTuple):
    """Return type for :func:`unique_counts`."""

    values: AlgebraicArray
    """Unique values (wrapped AlgebraicArray)."""
    counts: Array
    """Number of times each unique value appears."""


class UniqueInverseResult(typing.NamedTuple):
    """Return type for :func:`unique_inverse`."""

    values: AlgebraicArray
    """Unique values (wrapped AlgebraicArray)."""
    inverse_indices: Array
    """Indices such that ``values[inverse_indices] == x.ravel()``."""


def broadcast_arrays(*arrays: AlgebraicArray) -> Sequence[AlgebraicArray]:
    """Broadcast a collection of arrays to a common shape.

    Parameters
    ----------
    *arrays : AlgebraicArray
        One or more arrays with the same semiring.

    Returns
    -------
    tuple of AlgebraicArray
        Each array broadcast to the common shape.
    """
    validate_semiring(*arrays)
    xp = array_api_compat.array_namespace(arrays[0].data)
    raw: list[Array] = list(xp.broadcast_arrays(*[a.data for a in arrays]))
    return tuple(arrays[0]._wrap(r) for r in raw)


def broadcast_to(x: AlgebraicArray, shape: tuple[int, ...]) -> AlgebraicArray:
    """Broadcast *x* to the given *shape*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    shape : tuple of int
        Target shape (must be broadcast-compatible with ``x.shape``).
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.broadcast_to(x.data, shape))


def concat(arrays: Sequence[AlgebraicArray], *, axis: int = 0) -> AlgebraicArray:
    """Concatenate arrays along an existing axis.

    Parameters
    ----------
    arrays : sequence of AlgebraicArray
        Arrays with the same semiring.
    axis : int, default 0
        Axis along which to concatenate.
    """
    arr_list = list(arrays)
    validate_semiring(*arr_list)
    xp = array_api_compat.array_namespace(arr_list[0].data)
    result: Array = xp.concat([a.data for a in arr_list], axis=axis)
    return arr_list[0]._wrap(result)


def expand_dims(x: AlgebraicArray, *, axis: int) -> AlgebraicArray:
    """Insert a size-1 axis at position *axis*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int
        Position of the new axis.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.expand_dims(x.data, axis=axis))


def flip(x: AlgebraicArray, *, axis: int | tuple[int, ...] | None = None) -> AlgebraicArray:
    """Reverse the order of elements along one or more axes.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int or tuple of int or None, optional
        Axis or axes to flip. ``None`` flips all axes.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.flip(x.data, axis=axis))


def moveaxis(
    x: AlgebraicArray,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> AlgebraicArray:
    """Move axes of an array to new positions.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    source : int or sequence of int
        Original axis position(s).
    destination : int or sequence of int
        Destination axis position(s).
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.moveaxis(x.data, source, destination))


def permute_dims(x: AlgebraicArray, axes: tuple[int, ...]) -> AlgebraicArray:
    """Permute array dimensions.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axes : tuple of int
        Permutation of ``range(x.ndim)``.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.permute_dims(x.data, axes))


def repeat(
    x: AlgebraicArray,
    repeats: int | Array,
    *,
    axis: int | None = None,
) -> AlgebraicArray:
    """Repeat elements of an array.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    repeats : int or Array
        Number of repetitions for each element.
    axis : int or None, optional
        Axis along which to repeat. ``None`` flattens first.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.repeat(x.data, repeats, axis=axis))


def reshape(
    x: AlgebraicArray,
    shape: tuple[int, ...],
    *,
    copy: bool | None = None,
) -> AlgebraicArray:
    """Give a new shape to an array without changing its data.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    shape : tuple of int
        New shape. One entry may be ``-1`` to infer that dimension.
    copy : bool or None, optional
        Whether to copy the data (``None`` means copy only if needed).
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.reshape(x.data, shape, copy=copy)
    return x._wrap(result)


def roll(
    x: AlgebraicArray,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> AlgebraicArray:
    """Roll array elements along an axis.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    shift : int or tuple of int
        Number of places by which elements are shifted.
    axis : int or tuple of int or None, optional
        Axis or axes along which to roll.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.roll(x.data, shift, axis=axis))


def squeeze(
    x: AlgebraicArray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> AlgebraicArray:
    """Remove size-1 dimensions.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int or tuple of int or None, optional
        Size-1 axes to remove. ``None`` removes all size-1 axes.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.squeeze(x.data, axis=axis))


def stack(arrays: Sequence[AlgebraicArray], *, axis: int = 0) -> AlgebraicArray:
    """Join a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays : sequence of AlgebraicArray
        Arrays with the same semiring and identical shapes.
    axis : int, default 0
        Position of the new axis in the output array.
    """
    arr_list = list(arrays)
    validate_semiring(*arr_list)
    xp = array_api_compat.array_namespace(arr_list[0].data)
    result: Array = xp.stack([a.data for a in arr_list], axis=axis)
    return arr_list[0]._wrap(result)


def tile(x: AlgebraicArray, repetitions: int | tuple[int, ...]) -> AlgebraicArray:
    """Tile an array by repeating it the given number of times.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    repetitions : int or tuple of int
        Number of repetitions along each axis.
    """
    xp = array_api_compat.array_namespace(x.data)
    result = xp.tile(x.data, repetitions)
    return x._wrap(result)


def unstack(x: AlgebraicArray, *, axis: int = 0) -> Sequence[AlgebraicArray]:
    """Split an array into a list of arrays along *axis*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int, default 0
        Axis to unstack.

    Returns
    -------
    tuple of AlgebraicArray
        One array per slice along *axis*.
    """
    xp = array_api_compat.array_namespace(x.data)
    slices = tuple(xp.unstack(x.data, axis=axis))
    return tuple(x._wrap(s) for s in slices)


def where(condition: Array, x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Select elements from *x* or *y* depending on *condition*.

    Parameters
    ----------
    condition : Array
        Raw boolean array (not wrapped).
    x : AlgebraicArray
        Values selected where *condition* is ``True``.
    y : AlgebraicArray
        Values selected where *condition* is ``False``.

    Raises
    ------
    ValueError
        If *x* and *y* have different semirings.
    """
    validate_semiring(x, y)
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.where(condition, x.data, y.data)
    return x._wrap(result)


def unique_all(x: AlgebraicArray) -> UniqueAllResult:
    """Find unique values and their positions in *x*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.

    Returns
    -------
    UniqueAllResult
        Named tuple with fields: ``values`` (wrapped), ``indices``,
        ``inverse_indices``, and ``counts`` (all raw arrays).
    """
    xp = array_api_compat.array_namespace(x.data)
    if array_api_compat.is_torch_array(x.data):
        import torch

        flat_data = x.data.reshape(-1)
        unique, inverse_indices, counts = torch.unique(flat_data, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(inverse_indices, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        indices = ind_sorted[cum_sum]
        return UniqueAllResult(
            values=x._wrap(unique),
            indices=indices,
            inverse_indices=inverse_indices,
            counts=counts,
        )

    out = xp.unique_all(x.data)
    return UniqueAllResult(
        values=x._wrap(out.values),
        indices=out.indices,
        inverse_indices=out.inverse_indices,
        counts=out.counts,
    )


def unique_counts(x: AlgebraicArray) -> UniqueCountsResult:
    """Find unique values and their occurrence counts in *x*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.

    Returns
    -------
    UniqueCountsResult
        Named tuple with fields: ``values`` (wrapped) and ``counts`` (raw).
    """
    xp = array_api_compat.array_namespace(x.data)
    out = xp.unique_counts(x.data)
    return UniqueCountsResult(values=x._wrap(out.values), counts=out.counts)


def unique_inverse(x: AlgebraicArray) -> UniqueInverseResult:
    """Find unique values and the inverse permutation in *x*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.

    Returns
    -------
    UniqueInverseResult
        Named tuple with fields: ``values`` (wrapped) and
        ``inverse_indices`` (raw).
    """
    xp = array_api_compat.array_namespace(x.data)
    out = xp.unique_inverse(x.data)
    return UniqueInverseResult(values=x._wrap(out.values), inverse_indices=out.inverse_indices)


def unique_values(x: AlgebraicArray) -> AlgebraicArray:
    """Return the unique values in *x* (sorted).

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.unique_values(x.data)
    return x._wrap(result)


def take(
    x: AlgebraicArray,
    indices: Array | int,
    *,
    axis: int | None = None,
) -> AlgebraicArray:
    """Take elements from *x* at *indices*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    indices : Array or int
        Integer array of indices (or a scalar integer).
    axis : int or None, optional
        Axis along which to take. ``None`` treats *x* as flattened.
    """
    xp = array_api_compat.array_namespace(x.data)
    # Array API spec requires indices to be a 1-d array, not a plain int.
    scalar_index = isinstance(indices, int)
    if scalar_index:
        indices = xp.asarray([indices])
    result: Array = xp.take(x.data, indices, axis=axis)
    if scalar_index:
        squeeze_axis = axis if axis is not None else 0
        result = xp.squeeze(result, axis=squeeze_axis)
    return x._wrap(result)


def take_along_axis(
    x: AlgebraicArray,
    indices: Array,
    *,
    axis: int,
) -> AlgebraicArray:
    """Take values from *x* by matching *indices* along *axis*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    indices : Array
        Integer array of the same shape as *x* (or broadcastable).
    axis : int
        Axis along which to gather.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.take_along_axis(x.data, indices, axis=axis)
    return x._wrap(result)


def allclose(a: AlgebraicArray | Array, b: AlgebraicArray | Array | Number, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two arrays are element-wise equal within a tolerance.

    Parameters
    ----------
    a : AlgebraicArray or Array
        First input.
    b : AlgebraicArray or Array or Number
        Second input.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if all elements are close.
    """
    if isinstance(a, AlgebraicArray) and isinstance(b, AlgebraicArray):
        validate_semiring(a, b)
    a_data = maybe_unwrap(a)
    b_data = maybe_unwrap(b)

    xp = array_api_compat.array_namespace(a_data, b_data)
    if array_api_compat.is_jax_namespace(xp):
        import jax.numpy as jnp

        a_data = jnp.asarray(a_data)
        b_data = jnp.asarray(b_data)

        return bool(jnp.allclose(a_data, b_data, rtol=rtol, atol=atol).item())
    elif array_api_compat.is_torch_namespace(xp):
        import torch

        a_data = torch.asarray(a_data)
        b_data = torch.asarray(b_data)

        return bool(torch.allclose(a_data, b_data, rtol=rtol, atol=atol))
    elif array_api_compat.is_numpy_namespace(xp):
        import numpy as np

        a_data = np.asarray(a_data)
        b_data = np.asarray(b_data)

        return np.allclose(a_data, b_data, rtol=rtol, atol=atol)
    raise TypeError(f"Unknown array type: {type(a)}, {type(b)}")


def isclose(a: AlgebraicArray | Array, b: AlgebraicArray | Array | Number, *, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
    """Element-wise check if two arrays are within a tolerance.

    Parameters
    ----------
    a : AlgebraicArray or Array
        First input.
    b : AlgebraicArray or Array or Number
        Second input.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.

    Returns
    -------
    Array
        Boolean array of element-wise closeness.
    """
    if isinstance(a, AlgebraicArray) and isinstance(b, AlgebraicArray):
        validate_semiring(a, b)
    a_data = maybe_unwrap(a)
    b_data = maybe_unwrap(b)

    xp = array_api_compat.array_namespace(a_data, b_data)
    if array_api_compat.is_jax_namespace(xp):
        import jax.numpy as jnp

        a_data = jnp.asarray(a_data)
        b_data = jnp.asarray(b_data)

        return jnp.isclose(a_data, b_data, rtol=rtol, atol=atol)
    elif array_api_compat.is_torch_namespace(xp):
        import torch

        a_data = torch.asarray(a_data)
        b_data = torch.asarray(b_data)

        return torch.isclose(a_data, b_data, rtol=rtol, atol=atol)
    elif array_api_compat.is_numpy_namespace(xp):
        import numpy as np

        a_data = np.asarray(a_data)
        b_data = np.asarray(b_data)

        return np.isclose(a_data, b_data, rtol=rtol, atol=atol)
    raise TypeError(f"Unknown array type: {type(a)}, {type(b)}")


def equal(x: AlgebraicArray, y: AlgebraicArray | Number) -> Array:
    """Element-wise equality; returns a raw bool array.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    y : AlgebraicArray or Number
        Comparison value (semirings may differ, comparison is data-only).
    """
    xp = array_api_compat.array_namespace(x.data)
    y_data = y.data if isinstance(y, AlgebraicArray) else y
    result: Array = xp.equal(x.data, y_data)
    return result


def not_equal(x: AlgebraicArray, y: AlgebraicArray | Number) -> Array:
    """Element-wise inequality; returns a raw bool array.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    y : AlgebraicArray or Number
        Comparison value (semirings may differ, comparison is data-only).
    """
    xp = array_api_compat.array_namespace(x.data)
    y_data = y.data if isinstance(y, AlgebraicArray) else y
    result: Array = xp.not_equal(x.data, y_data)
    return result


def positive(x: AlgebraicArray) -> AlgebraicArray:
    """Identity operation (unary positive); returns a shallow copy.

    Equivalent to ``__pos__``.
    """
    return +x


def matrix_transpose(x: AlgebraicArray) -> AlgebraicArray:
    """Transpose of the last two dimensions.

    Parameters
    ----------
    x : AlgebraicArray
        Array with at least 2 dimensions.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.linalg.matrix_transpose(x.data)
    return x._wrap(result)


def diagonal(x: AlgebraicArray, /, *, offset: int = 0) -> AlgebraicArray:
    """Return the diagonal of a 2-D (or batched) array.

    Parameters
    ----------
    x : AlgebraicArray
        Array with at least 2 dimensions.
    offset : int, default 0
        Diagonal offset (0 = main diagonal).
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.linalg.diagonal(x.data, offset=offset)
    return x._wrap(result)
