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
from algebraic.types import Array
from algebraic.utils import validate_semiring


class UniqueAllResult(typing.NamedTuple):
    """Return type for `unique_all`."""

    values: AlgebraicArray
    """Unique values (wrapped AlgebraicArray)."""
    indices: Array
    """Indices of the first occurrence of each unique value in the input."""
    inverse_indices: Array
    """Indices such that `values[inverse_indices] == x.ravel()`."""
    counts: Array
    """Number of times each unique value appears."""


class UniqueCountsResult(typing.NamedTuple):
    """Return type for `unique_counts`."""

    values: AlgebraicArray
    """Unique values (wrapped AlgebraicArray)."""
    counts: Array
    """Number of times each unique value appears."""


class UniqueInverseResult(typing.NamedTuple):
    """Return type for `unique_inverse`."""

    values: AlgebraicArray
    """Unique values (wrapped AlgebraicArray)."""
    inverse_indices: Array
    """Indices such that `values[inverse_indices] == x.ravel()`."""


def broadcast_arrays(*arrays: AlgebraicArray) -> Sequence[AlgebraicArray]:
    """Broadcast a collection of arrays to a common shape.

    Args:
        *arrays: One or more arrays with the same semiring.

    Returns:
        Sequence[AlgebraicArray]: Each array broadcast to the common shape.
    """
    validate_semiring(*arrays)
    xp = array_api_compat.array_namespace(arrays[0].data)
    raw: list[Array] = list(xp.broadcast_arrays(*[a.data for a in arrays]))
    return tuple(arrays[0]._wrap(r) for r in raw)


def broadcast_to(x: AlgebraicArray, shape: tuple[int, ...]) -> AlgebraicArray:
    """Broadcast *x* to the given *shape*.

    Args:
        x: Input array.
        shape: Target shape (must be broadcast-compatible with `x.shape`).
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.broadcast_to(x.data, shape))


def concat(arrays: Sequence[AlgebraicArray], *, axis: int = 0) -> AlgebraicArray:
    """Concatenate arrays along an existing axis.

    Args:
        arrays: Sequence of arrays with the same semiring.
        axis: Axis along which to concatenate (default 0).
    """
    arr_list = list(arrays)
    validate_semiring(*arr_list)
    xp = array_api_compat.array_namespace(arr_list[0].data)
    result: Array = xp.concat([a.data for a in arr_list], axis=axis)
    return arr_list[0]._wrap(result)


def expand_dims(x: AlgebraicArray, *, axis: int) -> AlgebraicArray:
    """Insert a size-1 axis at position *axis*.

    Args:
        x: Input array.
        axis: Position of the new axis.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.expand_dims(x.data, axis=axis))


def flip(x: AlgebraicArray, *, axis: int | tuple[int, ...] | None = None) -> AlgebraicArray:
    """Reverse the order of elements along one or more axes.

    Args:
        x: Input array.
        axis: Axis or axes to flip. `None` flips all axes.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.flip(x.data, axis=axis))


def moveaxis(
    x: AlgebraicArray,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> AlgebraicArray:
    """Move axes of an array to new positions.

    Args:
        x: Input array.
        source: Original axis position(s).
        destination: Destination axis position(s).
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.moveaxis(x.data, source, destination))


def permute_dims(x: AlgebraicArray, axes: tuple[int, ...]) -> AlgebraicArray:
    """Permute array dimensions.

    Args:
        x: Input array.
        axes: Permutation of `range(x.ndim)`.
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

    Args:
        x: Input array.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat. `None` flattens first.
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

    Args:
        x: Input array.
        shape: New shape. One entry may be `-1` to infer that dimension.
        copy: Whether to copy the data (`None` means copy only if needed).
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

    Args:
        x: Input array.
        shift: Number of places by which elements are shifted.
        axis: Axis or axes along which to roll.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.roll(x.data, shift, axis=axis))


def squeeze(
    x: AlgebraicArray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> AlgebraicArray:
    """Remove size-1 dimensions.

    Args:
        x: Input array.
        axis: Size-1 axes to remove. `None` removes all size-1 axes.
    """
    xp = array_api_compat.array_namespace(x.data)
    return x._wrap(xp.squeeze(x.data, axis=axis))


def stack(arrays: Sequence[AlgebraicArray], *, axis: int = 0) -> AlgebraicArray:
    """Join a sequence of arrays along a new axis.

    Args:
        arrays: Sequence of arrays with the same semiring and identical shapes.
        axis: Position of the new axis in the output array.
    """
    arr_list = list(arrays)
    validate_semiring(*arr_list)
    xp = array_api_compat.array_namespace(arr_list[0].data)
    result: Array = xp.stack([a.data for a in arr_list], axis=axis)
    return arr_list[0]._wrap(result)


def tile(x: AlgebraicArray, repetitions: int | tuple[int, ...]) -> AlgebraicArray:
    """Tile an array by repeating it the given number of times.

    Args:
        x: Input array.
        repetitions: Number of repetitions along each axis.
    """
    xp = array_api_compat.array_namespace(x.data)
    result = xp.tile(x.data, repetitions)
    return x._wrap(result)


def unstack(x: AlgebraicArray, *, axis: int = 0) -> Sequence[AlgebraicArray]:
    """Split an array into a list of arrays along *axis*.

    Args:
        x: Input array.
        axis: Axis to unstack (default 0).

    Returns:
        list[AlgebraicArray]: One array per slice along *axis*.
    """
    xp = array_api_compat.array_namespace(x.data)
    slices = tuple(xp.unstack(x.data, axis=axis))
    return tuple(x._wrap(s) for s in slices)


def where(condition: Array, x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Select elements from *x* or *y* depending on *condition*.

    Args:
        condition: Raw boolean array (not wrapped).
        x: Values selected where *condition* is `True`.
        y: Values selected where *condition* is `False`.

    Raises:
        ValueError: If *x* and *y* have different semirings.
    """
    validate_semiring(x, y)
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.where(condition, x.data, y.data)
    return x._wrap(result)


def unique_all(x: AlgebraicArray) -> UniqueAllResult:
    """Find unique values and their positions in *x*.

    Args:
        x: Input array.

    Returns:
        UniqueAllResult: A dataclass with fields: `values` (wrapped), `indices`,
            `inverse_indices`, and `counts` (all raw arrays).
    """
    xp = array_api_compat.array_namespace(x.data)
    out = xp.unique_all(x.data)
    return UniqueAllResult(
        values=x._wrap(out.values),
        indices=out.indices,
        inverse_indices=out.inverse_indices,
        counts=out.counts,
    )


def unique_counts(x: AlgebraicArray) -> UniqueCountsResult:
    """Find unique values and their occurrence counts in *x*.

    Args:
        x: Input array.

    Returns:
        UniqueCountsResult: A dataclass with fields: `values` (wrapped) and `counts` (raw).
    """
    xp = array_api_compat.array_namespace(x.data)
    out = xp.unique_counts(x.data)
    return UniqueCountsResult(values=x._wrap(out.values), counts=out.counts)


def unique_inverse(x: AlgebraicArray) -> UniqueInverseResult:
    """Find unique values and the inverse permutation in *x*.

    Args:
        x: Input array.

    Returns:
        UniqueInverseResult: A dataclass with fields: `values` (wrapped) and
            `inverse_indices` (raw).
    """
    xp = array_api_compat.array_namespace(x.data)
    out = xp.unique_inverse(x.data)
    return UniqueInverseResult(values=x._wrap(out.values), inverse_indices=out.inverse_indices)


def unique_values(x: AlgebraicArray) -> AlgebraicArray:
    """Return the unique values in *x* (sorted).

    Args:
        x: Input array.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.unique_values(x.data)
    return x._wrap(result)


def take(
    x: AlgebraicArray,
    indices: Array,
    *,
    axis: int | None = None,
) -> AlgebraicArray:
    """Take elements from *x* at *indices*.

    Args:
        x: Input array.
        indices: Integer array of indices.
        axis: Axis along which to take. `None` treats *x* as flattened.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.take(x.data, indices, axis=axis)
    return x._wrap(result)


def take_along_axis(
    x: AlgebraicArray,
    indices: Array,
    *,
    axis: int,
) -> AlgebraicArray:
    """Take values from *x* by matching *indices* along *axis*.

    Args:
        x: Input array.
        indices: Integer array of the same shape as *x* (or broadcastable).
        axis: Axis along which to gather.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.take_along_axis(x.data, indices, axis=axis)
    return x._wrap(result)


def equal(x: AlgebraicArray, y: AlgebraicArray) -> Array:
    """Element-wise equality; returns a raw bool array.

    Args:
        x: Input arrays (semirings may differ -- comparison is data-only).
        y: Input arrays (semirings may differ -- comparison is data-only).
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.equal(x.data, y.data)
    return result


def not_equal(x: AlgebraicArray, y: AlgebraicArray) -> Array:
    """Element-wise inequality; returns a raw bool array.

    Args:
        x: Input arrays (semirings may differ -- comparison is data-only).
        y: Input arrays (semirings may differ -- comparison is data-only).
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.not_equal(x.data, y.data)
    return result


def positive(x: AlgebraicArray) -> AlgebraicArray:
    """Identity operation (unary positive); returns a shallow copy.

    Equivalent to `__pos__`.
    """
    return +x


def matrix_transpose(x: AlgebraicArray) -> AlgebraicArray:
    """Transpose of the last two dimensions.

    Args:
        x: Array with at least 2 dimensions.
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.linalg.matrix_transpose(x.data)
    return x._wrap(result)


def diagonal(x: AlgebraicArray, /, *, offset: int = 0) -> AlgebraicArray:
    """Return the diagonal of a 2-D (or batched) array.

    Args:
        x: Array with at least 2 dimensions.
        offset: Diagonal offset (0 = main diagonal).
    """
    xp = array_api_compat.array_namespace(x.data)
    result: Array = xp.linalg.diagonal(x.data, offset=offset)
    return x._wrap(result)
