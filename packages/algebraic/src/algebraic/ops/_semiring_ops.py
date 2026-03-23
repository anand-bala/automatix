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
from algebraic.spec import Semiring, is_ring
from algebraic.types import AccumulationFn, Array, BinaryOp, ScanFn

K = typing.TypeVar("K", bound=Semiring)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
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


def _make_prefix_scan_fn(binary_op: BinaryOp) -> ScanFn:
    """Return a `ScanFn` that accumulates elements using *binary_op*.

    The returned function has signature `(carry, x) -> (new_carry, output)`
    where `new_carry == output == binary_op(carry, x)`.  This produces a
    standard inclusive prefix scan when passed to `AlgebraicArray.scan`.
    """

    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        new_carry: Array = binary_op(carry, x)
        return new_carry, new_carry

    # cast: plain function satisfies ScanFn protocol; mypy struggles with
    # Callable vs Protocol when parameter types differ in their supertype chains.
    return typing.cast(ScanFn, scan_fn)


# ---------------------------------------------------------------------------
# Elementwise operations (delegate to dunders)
# ---------------------------------------------------------------------------


def add(x: AlgebraicArray[K], y: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Element-wise semiring addition.

    Args:
        x: Arrays with the same semiring.
        y: Arrays with the same semiring.
    """
    return x + y


def multiply(x: AlgebraicArray[K], y: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Element-wise semiring multiplication.

    Args:
        x: Arrays with the same semiring.
        y: Arrays with the same semiring.
    """
    return x * y


def subtract(x: AlgebraicArray[K], y: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Element-wise semiring subtraction (requires a Ring).

    Args:
        x: Arrays with the same semiring. The semiring must be a Ring (must
            have an `additive_inverse` operation).
        y: Arrays with the same semiring. The semiring must be a Ring (must
            have an `additive_inverse` operation).

    Raises:
        NotImplementedError: If `x.semiring` is not a Ring.
    """
    return x - y


def negative(x: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Element-wise negation.

    Uses `additive_inverse` for Rings or `complement` for Boolean /
    De Morgan algebras.

    Raises:
        NotImplementedError: If the semiring supports neither operation.
    """
    return -x


def square(x: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Element-wise semiring square (`x * x`)."""
    return x * x


# ---------------------------------------------------------------------------
# Statistical reductions (use abstract ``reduce``)
# ---------------------------------------------------------------------------


def sum(  # noqa: A001  (intentional shadowing of built-in)
    x: AlgebraicArray[K],
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> AlgebraicArray[K]:
    """Reduce *x* using the semiring's addition along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    dims = _normalize_axes(axis, x.ndim)
    zero = x._wrap(x.semiring.zeros(()))
    # cast: BinaryOp is contravariant-compatible with AccumulationFn but
    # mypy does not resolve this structural subtyping automatically.
    result = x.reduce(typing.cast(AccumulationFn, x.semiring.add), [zero], [x], dimensions=dims)
    if keepdims:
        xp = array_api_compat.array_namespace(result.data)
        new_data: Array = result.data
        for dim in sorted(dims):
            new_data = xp.expand_dims(new_data, axis=dim)
        result = result._wrap(new_data)
    return result


def prod(
    x: AlgebraicArray[K],
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> AlgebraicArray[K]:
    """Reduce *x* using the semiring's multiplication along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    dims = _normalize_axes(axis, x.ndim)
    one = x._wrap(x.semiring.ones(()))
    # cast: same reasoning as in `sum`.
    result = x.reduce(typing.cast(AccumulationFn, x.semiring.mul), [one], [x], dimensions=dims)
    if keepdims:
        xp = array_api_compat.array_namespace(result.data)
        new_data: Array = result.data
        for dim in sorted(dims):
            new_data = xp.expand_dims(new_data, axis=dim)
        result = result._wrap(new_data)
    return result


# ---------------------------------------------------------------------------
# Cumulative operations (use abstract ``scan``)
# ---------------------------------------------------------------------------


def cumulative_sum(
    x: AlgebraicArray[K],
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> AlgebraicArray[K]:
    """Inclusive prefix sum along *axis* using the semiring's addition.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a zero slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    xp = array_api_compat.array_namespace(x.data)
    ndim = x.ndim
    ax = axis % ndim

    # Permute so the scan axis is leading.
    if ax != 0:
        perm = (ax,) + tuple(i for i in range(ndim) if i != ax)
        x_permuted = x._wrap(xp.permute_dims(x.data, perm))
    else:
        perm = None
        x_permuted = x

    # Initial carry: zero of shape matching a single slice along the leading axis.
    slice_shape = x_permuted.data.shape[1:]
    zero_init = x._wrap(x.semiring.zeros(slice_shape))

    scan_fn = _make_prefix_scan_fn(x.semiring.add)
    result_permuted = x.scan(scan_fn, zero_init, [x_permuted])

    # Optionally prepend the initial zero slice.
    if include_initial:
        zero_slice: Array = xp.expand_dims(x.semiring.zeros(slice_shape), axis=0)
        combined: Array = xp.concat([zero_slice, result_permuted.data], axis=0)
        result_permuted = result_permuted._wrap(combined)

    # Permute back if we moved the axis.
    if perm is not None:
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result_data: Array = xp.permute_dims(result_permuted.data, tuple(inv_perm))
        return x._wrap(result_data)

    return result_permuted


def cumulative_prod(
    x: AlgebraicArray[K],
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> AlgebraicArray[K]:
    """Inclusive prefix product along *axis* using the semiring's multiplication.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a one slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    xp = array_api_compat.array_namespace(x.data)
    ndim = x.ndim
    ax = axis % ndim

    # Permute so the scan axis is leading.
    if ax != 0:
        perm = (ax,) + tuple(i for i in range(ndim) if i != ax)
        x_permuted = x._wrap(xp.permute_dims(x.data, perm))
    else:
        perm = None
        x_permuted = x

    # Initial carry: one of shape matching a single slice along the leading axis.
    slice_shape = x_permuted.data.shape[1:]
    one_init = x._wrap(x.semiring.ones(slice_shape))

    scan_fn = _make_prefix_scan_fn(x.semiring.mul)
    result_permuted = x.scan(scan_fn, one_init, [x_permuted])

    # Optionally prepend the initial one slice.
    if include_initial:
        one_slice: Array = xp.expand_dims(x.semiring.ones(slice_shape), axis=0)
        combined: Array = xp.concat([one_slice, result_permuted.data], axis=0)
        result_permuted = result_permuted._wrap(combined)

    # Permute back if we moved the axis.
    if perm is not None:
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result_data: Array = xp.permute_dims(result_permuted.data, tuple(inv_perm))
        return x._wrap(result_data)

    return result_permuted


# ---------------------------------------------------------------------------
# Linear algebra (use abstract ``dot_general``)
# ---------------------------------------------------------------------------


def matmul(x: AlgebraicArray[K], y: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Matrix multiplication using semiring operations.

    Equivalent to the `@` operator; delegates to `AlgebraicArray.__matmul__`.
    """
    return x @ y


def vecdot(x: AlgebraicArray[K], y: AlgebraicArray[K], /, *, axis: int = -1) -> AlgebraicArray[K]:
    """Inner (dot) product of two arrays contracted along *axis*.

    Args:
        x: Arrays with identical shapes except possibly along *axis*.
        y: Arrays with identical shapes except possibly along *axis*.
        axis: The axis along which to contract (default `-1`).
    """
    ndim = x.ndim
    ax = axis % ndim
    batch = tuple(i for i in range(ndim) if i != ax)
    return x.dot_general(y, (((ax,), (ax,)), (batch, batch)))


@overload
def tensordot(
    x: AlgebraicArray[K],
    y: AlgebraicArray[K],
    /,
    *,
    axes: int,
) -> AlgebraicArray[K]: ...


@overload
def tensordot(
    x: AlgebraicArray[K],
    y: AlgebraicArray[K],
    /,
    *,
    axes: tuple[Sequence[int], Sequence[int]],
) -> AlgebraicArray[K]: ...


def tensordot(
    x: AlgebraicArray[K],
    y: AlgebraicArray[K],
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> AlgebraicArray[K]:
    """Generalised tensor contraction using semiring operations.

    Args:
        x: Input array.
        y: Input array.
        axes: `int` n -- contract the last *n* axes of *x* with the first *n*
            axes of *y* (`axes=2` is equivalent to standard matrix multiply
            for 2-D arrays). Alternatively, `(lhs_axes, rhs_axes)` -- explicit
            contracting axis sequences.
    """
    if isinstance(axes, int):
        n = axes
        lhs_contract = tuple(range(x.ndim - n, x.ndim))
        rhs_contract = tuple(range(n))
    else:
        lhs_contract = tuple(axes[0])
        rhs_contract = tuple(axes[1])
    return x.dot_general(y, ((lhs_contract, rhs_contract), ((), ())))


# ---------------------------------------------------------------------------
# linalg extensions (built on the above + array_api_compat)
# ---------------------------------------------------------------------------


def trace(x: AlgebraicArray[K], /, *, offset: int = 0) -> AlgebraicArray[K]:
    """Sum of diagonal elements using the semiring's addition.

    Args:
        x: 2-D (or batched) square-ish array.
        offset: Diagonal offset (0 = main diagonal; positive = above; negative = below).
    """
    xp = array_api_compat.array_namespace(x.data)
    diag_data: Array = xp.linalg.diagonal(x.data, offset=offset)
    return sum(x._wrap(diag_data), axis=-1)


def outer(x: AlgebraicArray[K], y: AlgebraicArray[K]) -> AlgebraicArray[K]:
    """Outer product of two 1-D arrays using semiring multiplication.

    Args:
        x: 1-D array. The result has shape `(x.shape[0], y.shape[0])`.
        y: 1-D array. The result has shape `(x.shape[0], y.shape[0])`.
    """
    # No contracting dims, no batch dims — result[i, j] = x[i] * y[j].
    return x.dot_general(y, (((), ()), ((), ())))


def matrix_power(x: AlgebraicArray[K], n: int) -> AlgebraicArray[K]:
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
            "matrix_power n=0 requires creation of an identity matrix, "
            "which depends on backend-specific creation functions."
        )

    result: AlgebraicArray[K] | None = None
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


def cross(x: AlgebraicArray[K], y: AlgebraicArray[K], /, *, axis: int = -1) -> AlgebraicArray[K]:
    """3-D cross product using semiring operations (requires a Ring).

    Args:
        x: Array whose size along *axis* is exactly 3.
        y: Array whose size along *axis* is exactly 3.
        axis: The axis that indexes the three components (default `-1`).

    Raises:
        NotImplementedError: If the semiring is not a Ring (no `additive_inverse`).
        ValueError: If the size along *axis* is not 3.
    """
    if not is_ring(x.semiring):
        raise NotImplementedError(
            f"cross product requires a Ring with additive_inverse; "
            f"semiring {type(x.semiring).__name__} does not support subtraction."
        )
    ndim = x.ndim
    ax = axis % ndim
    if x.data.shape[ax] != 3:
        raise ValueError(f"cross product requires size-3 axis; got shape {x.data.shape} with axis={ax}")

    def _slice(arr: AlgebraicArray[K], i: int) -> AlgebraicArray[K]:
        """Extract the i-th element along *ax*."""
        idx: list[Any] = [slice(None)] * arr.ndim
        idx[ax] = i
        return arr[tuple(idx)]

    x0, x1, x2 = _slice(x, 0), _slice(x, 1), _slice(x, 2)
    y0, y1, y2 = _slice(y, 0), _slice(y, 1), _slice(y, 2)

    c0 = x1 * y2 - x2 * y1
    c1 = x2 * y0 - x0 * y2
    c2 = x0 * y1 - x1 * y0

    xp = array_api_compat.array_namespace(x.data)
    result_data: Array = xp.stack([c0.data, c1.data, c2.data], axis=ax)
    return x._wrap(result_data)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def diff(
    x: AlgebraicArray[K],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: AlgebraicArray[K] | None = None,
    append: AlgebraicArray[K] | None = None,
) -> AlgebraicArray[K]:
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
            f"diff requires a Ring with additive_inverse; "
            f"semiring {type(x.semiring).__name__} does not support subtraction."
        )

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
