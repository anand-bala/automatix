from __future__ import annotations

import typing
from collections.abc import Sequence

from plum import Dispatcher

if typing.TYPE_CHECKING:
    from algebraic import AlgebraicArray
from algebraic.types import Array, Number, is_array

dispatch = Dispatcher()


def normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    """Return a sorted tuple of non-negative axis indices.

    Args:
        axis: `None` (all axes), a single `int`, or a sequence of `int`s.
            Negative values are resolved modulo *ndim*.
        ndim: Number of dimensions of the array being operated on.
    """
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(sorted(a % ndim for a in axis))


def validate_semiring(*arrays: "AlgebraicArray") -> None:
    """Raise `ValueError` if any two inputs have different semiring instances.

    Args:
        *arrays: One or more `AlgebraicArray` instances.

    Raises:
        ValueError: If two arrays carry different semirings (compared with `==`).
    """
    if len(arrays) < 2:
        return
    first = arrays[0].semiring
    for arr in arrays[1:]:
        if arr.semiring != first:
            raise ValueError(f"All AlgebraicArray inputs must share the same semiring; got {first!r} and {arr.semiring!r}.")


def asanyarray(x: "AlgebraicArray" | Array | Number) -> Array:
    """Convert an object to an array.

    If scalar or unsupported type, will convert to NumPy array.
    """
    from algebraic import AlgebraicArray

    if isinstance(x, AlgebraicArray):
        return x.data
    if is_array(x):
        return x
    # default to numpy array
    import numpy as np

    return np.asanyarray(x)


def maybe_unwrap(x: "AlgebraicArray" | Array | Number) -> Array | Number:
    from algebraic import AlgebraicArray

    if isinstance(x, AlgebraicArray):
        return x.data
    if is_array(x):
        return x

    return x
