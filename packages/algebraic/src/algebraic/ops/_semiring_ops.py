"""Semiring-affected array operations for AlgebraicArray.

All functions in this module use the semiring's `add` and `mul` operations
rather than standard arithmetic. They are implemented on top of the three
abstract primitives defined in `algebraic.array.base.AlgebraicArray`:
`reduce`, `scan`, and `dot_general`.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Any

import array_api_compat
from typing_extensions import overload

from algebraic.array.base import AlgebraicArray
from algebraic.spec import is_ring
from algebraic.types import Array
from algebraic.utils import dispatch, validate_semiring


def add(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring addition.

    Args:
        x: Arrays with the same semiring.
        y: Arrays with the same semiring.
    """
    validate_semiring(x, y)
    return x + y


def multiply(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring multiplication.

    Args:
        x: Arrays with the same semiring.
        y: Arrays with the same semiring.
    """
    validate_semiring(x, y)
    return x * y


def subtract(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring subtraction (requires a Ring).

    Args:
        x: Arrays with the same semiring. The semiring must be a Ring (must
            have an `additive_inverse` operation).
        y: Arrays with the same semiring. The semiring must be a Ring (must
            have an `additive_inverse` operation).

    Raises:
        NotImplementedError: If `x.semiring` is not a Ring.
    """
    validate_semiring(x, y)
    return x - y


def negative(x: AlgebraicArray) -> AlgebraicArray:
    """Element-wise negation.

    Uses `additive_inverse` for Rings or `complement` for Boolean /
    De Morgan algebras.

    Raises:
        NotImplementedError: If the semiring supports neither operation.
    """
    return -x


def square(x: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring square (`x * x`)."""
    return x * x


@dispatch.abstract
def sum(  # noqa: A001  (intentional shadowing of built-in)
    x: AlgebraicArray, /, *, axis: int | Sequence[int] | None = None, keepdims: bool = False
) -> AlgebraicArray:
    """Reduce *x* using the semiring's addition along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    raise NotImplementedError


@dispatch.abstract
def prod(
    x: AlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> AlgebraicArray:
    """Reduce *x* using the semiring's multiplication along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    raise NotImplementedError


@dispatch.abstract
def cumulative_sum(
    x: AlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> AlgebraicArray:
    """Inclusive prefix sum along *axis* using the semiring's addition.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a zero slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    raise NotImplementedError


@dispatch.abstract
def cumulative_prod(
    x: AlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> AlgebraicArray:
    """Inclusive prefix product along *axis* using the semiring's multiplication.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a one slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    raise NotImplementedError


def matmul(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Matrix multiplication using semiring operations.

    Equivalent to the `@` operator; delegates to `AlgebraicArray.__matmul__`.
    """
    validate_semiring(x, y)
    return x @ y


def vecdot(x: AlgebraicArray, y: AlgebraicArray, /, *, axis: int = -1) -> AlgebraicArray:
    """Inner (dot) product of two arrays contracted along *axis*.

    Args:
        x: Arrays with identical shapes except possibly along *axis*.
        y: Arrays with identical shapes except possibly along *axis*.
        axis: The axis along which to contract (default `-1`).
    """
    validate_semiring(x, y)
    ndim = x.ndim
    ax = axis % ndim
    batch = tuple(i for i in range(ndim) if i != ax)
    return x.dot_general(y, (((ax,), (ax,)), (batch, batch)))


@overload
def tensordot(
    x: AlgebraicArray,
    y: AlgebraicArray,
    /,
    *,
    axes: int,
) -> AlgebraicArray: ...


@overload
def tensordot(
    x: AlgebraicArray,
    y: AlgebraicArray,
    /,
    *,
    axes: tuple[Sequence[int], Sequence[int]],
) -> AlgebraicArray: ...


def tensordot(
    x: AlgebraicArray,
    y: AlgebraicArray,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> AlgebraicArray:
    """Generalised tensor contraction using semiring operations.

    Args:
        x: Input array.
        y: Input array.
        axes: `int` n -- contract the last *n* axes of *x* with the first *n*
            axes of *y* (`axes=2` is equivalent to standard matrix multiply
            for 2-D arrays). Alternatively, `(lhs_axes, rhs_axes)` -- explicit
            contracting axis sequences.
    """
    validate_semiring(x, y)
    if isinstance(axes, int):
        n = axes
        lhs_contract = tuple(range(x.ndim - n, x.ndim))
        rhs_contract = tuple(range(n))
    else:
        lhs_contract = tuple(axes[0])
        rhs_contract = tuple(axes[1])
    return x.dot_general(y, ((lhs_contract, rhs_contract), ((), ())))


def trace(x: AlgebraicArray, /, *, offset: int = 0) -> AlgebraicArray:
    """Sum of diagonal elements using the semiring's addition.

    Args:
        x: 2-D (or batched) square-ish array.
        offset: Diagonal offset (0 = main diagonal; positive = above; negative = below).
    """
    xp = array_api_compat.array_namespace(x.data)
    diag_data: Array = xp.linalg.diagonal(x.data, offset=offset)
    return typing.cast(AlgebraicArray, sum(x._wrap(diag_data), axis=-1))


def outer(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Outer product of two 1-D arrays using semiring multiplication.

    Args:
        x: 1-D array. The result has shape `(x.shape[0], y.shape[0])`.
        y: 1-D array. The result has shape `(x.shape[0], y.shape[0])`.
    """
    validate_semiring(x, y)
    # No contracting dims, no batch dims — result[i, j] = x[i] * y[j].
    return x.dot_general(y, (((), ()), ((), ())))


def matrix_power(x: AlgebraicArray, n: int) -> AlgebraicArray:
    """Raise a square matrix to the non-negative integer power *n*.

    Uses binary exponentiation (O(log n) multiplications).

    Args:
        x: Square 2-D array.
        n: Non-negative integer exponent. `n=0` is not supported because constructing
            an identity matrix requires backend-specific creation functions.

    Raises:
        ValueError: If *n* is negative.
        NotImplementedError: If *n* is zero.
    """
    if n < 0:
        raise ValueError(f"matrix_power requires n >= 0; got n={n}")
    if n == 0:
        raise NotImplementedError(
            "matrix_power n=0 requires creation of an identity matrix, which depends on backend-specific creation functions."
        )

    result: AlgebraicArray | None = None
    base = x
    exp = n
    while exp > 0:
        if exp & 1:
            result = base if result is None else result @ base
        base = base @ base
        exp >>= 1
    # result is None only when n==0, which is ruled out above.
    assert result is not None
    return result


def diff(
    x: AlgebraicArray,
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: AlgebraicArray | None = None,
    append: AlgebraicArray | None = None,
) -> AlgebraicArray:
    """Discrete differences along *axis* (requires a Ring).

    Computes the *n*-th-order forward difference: `out[i] = x[i+1] - x[i]`.

    Args:
        x: Input array.
        n: Order of the difference (default 1).
        axis: Axis along which differences are computed (default `-1`).
        prepend: Values to prepend to *x* along *axis* before computing differences.
        append: Values to append to *x* along *axis* before computing differences.

    Raises:
        NotImplementedError: If the semiring is not a Ring.
    """
    if not is_ring(x.semiring):
        raise NotImplementedError(
            f"diff requires a Ring with additive_inverse; semiring {type(x.semiring).__name__} does not support subtraction."
        )

    arrays: list[AlgebraicArray] = [x]
    if prepend is not None:
        arrays.append(prepend)
    if append is not None:
        arrays.append(append)
    if len(arrays) > 1:
        validate_semiring(*arrays)

    xp = array_api_compat.array_namespace(x.data)

    # Optionally prepend / append values.
    result = x
    if prepend is not None or append is not None:
        parts: list[Array] = []
        if prepend is not None:
            parts.append(prepend.data)
        parts.append(result.data)
        if append is not None:
            parts.append(append.data)
        ax_pre = axis % x.ndim
        result = x._wrap(xp.concat(parts, axis=ax_pre))

    for _ in range(n):
        ndim = result.ndim
        ax = axis % ndim
        idx_tail: list[Any] = [slice(None)] * ndim
        idx_tail[ax] = slice(1, None)
        idx_head: list[Any] = [slice(None)] * ndim
        idx_head[ax] = slice(None, -1)
        result = result[tuple(idx_tail)] - result[tuple(idx_head)]

    return result
