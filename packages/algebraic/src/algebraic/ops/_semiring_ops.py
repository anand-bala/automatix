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

if typing.TYPE_CHECKING:
    from opt_einsum.contract import OptimizeKind, _MemoryLimit


def add(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring addition.

    Parameters
    ----------
    x : AlgebraicArray
        Left operand.
    y : AlgebraicArray
        Right operand (must share the same semiring as *x*).
    """
    validate_semiring(x, y)
    return x + y


def multiply(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring multiplication.

    Parameters
    ----------
    x : AlgebraicArray
        Left operand.
    y : AlgebraicArray
        Right operand (must share the same semiring as *x*).
    """
    validate_semiring(x, y)
    return x * y


def subtract(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring subtraction (requires a :class:`~algebraic.spec.Ring`).

    Parameters
    ----------
    x : AlgebraicArray
        Left operand. The semiring must be a Ring (must have an
        ``additive_inverse`` operation).
    y : AlgebraicArray
        Right operand (must share the same semiring as *x*).

    Raises
    ------
    NotImplementedError
        If ``x.semiring`` is not a Ring.
    """
    validate_semiring(x, y)
    return x - y


def negative(x: AlgebraicArray) -> AlgebraicArray:
    """Element-wise negation.

    Uses ``additive_inverse`` for Rings or ``complement`` for Boolean /
    De Morgan algebras.

    Raises
    ------
    NotImplementedError
        If the semiring supports neither operation.
    """
    return -x


def square(x: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring square (``x * x``)."""
    return x * x


@dispatch.abstract
def sum(  # noqa: A001  (intentional shadowing of built-in)
    x: AlgebraicArray, /, *, axis: int | Sequence[int] | None = None, keepdims: bool = False
) -> AlgebraicArray:
    """Reduce *x* using the semiring's addition along *axis*.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int or sequence of int or None, optional
        Axis or axes to reduce. ``None`` reduces all axes.
    keepdims : bool, default False
        When ``True``, reduced axes are replaced by size-1 dimensions.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> a = algebraic.array([1.0, 2.0, 3.0], semiring=sr, backend="numpy")
    >>> algebraic.sum(a).data
    array(1.)
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

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int or sequence of int or None, optional
        Axis or axes to reduce. ``None`` reduces all axes.
    keepdims : bool, default False
        When ``True``, reduced axes are replaced by size-1 dimensions.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> a = algebraic.array([1.0, 2.0, 3.0], semiring=sr, backend="numpy")
    >>> algebraic.prod(a).data
    array(6.)
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

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int, default 0
        Axis along which to scan.
    include_initial : bool, default False
        When ``True``, prepend a zero slice before the scan output so that
        ``result.shape[axis] == x.shape[axis] + 1``.
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

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    axis : int, default 0
        Axis along which to scan.
    include_initial : bool, default False
        When ``True``, prepend a one slice before the scan output so that
        ``result.shape[axis] == x.shape[axis] + 1``.
    """
    raise NotImplementedError


def matmul(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Matrix multiplication using semiring operations.

    Equivalent to the ``@`` operator; delegates to
    :meth:`AlgebraicArray.__matmul__`.

    Parameters
    ----------
    x : AlgebraicArray
        Left operand.
    y : AlgebraicArray
        Right operand (must share the same semiring as *x*).

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> A = algebraic.array([[0.0, 1.0], [2.0, 3.0]], semiring=sr, backend="numpy")
    >>> B = algebraic.array([[4.0, 5.0], [6.0, 7.0]], semiring=sr, backend="numpy")
    >>> C = algebraic.matmul(A, B)
    >>> C.data  # doctest: +SKIP
    array([[ 6.,  7.],
           [ 6.,  7.]])
    """
    validate_semiring(x, y)
    return x @ y


def vecdot(x: AlgebraicArray, y: AlgebraicArray, /, *, axis: int = -1) -> AlgebraicArray:
    """Inner (dot) product of two arrays contracted along *axis*.

    Parameters
    ----------
    x : AlgebraicArray
        Left operand.
    y : AlgebraicArray
        Right operand (must have the same shape as *x* except possibly along
        *axis*).
    axis : int, default -1
        The axis along which to contract.
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

    Parameters
    ----------
    x : AlgebraicArray
        Left operand.
    y : AlgebraicArray
        Right operand.
    axes : int or tuple of (sequence of int, sequence of int), default 2
        Contract the last *n* axes of *x* with the first *n* axes of *y* (``axes=2`` is
        equivalent to standard matrix multiply for 2-D arrays). Alternatively, explicit
        contracting axis sequences ``(lhs_axes, rhs_axes)``.
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

    Parameters
    ----------
    x : AlgebraicArray
        2-D (or batched) square-ish array.
    offset : int, default 0
        Diagonal offset (0 = main diagonal; positive = above; negative = below).
    """
    xp = array_api_compat.array_namespace(x.data)
    diag_data: Array = xp.linalg.diagonal(x.data, offset=offset)
    return typing.cast(AlgebraicArray, sum(x._wrap(diag_data), axis=-1))


def outer(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
    """Outer product of two 1-D arrays using semiring multiplication.

    Parameters
    ----------
    x : AlgebraicArray
        1-D array.
    y : AlgebraicArray
        1-D array. The result has shape ``(x.shape[0], y.shape[0])``.
    """
    validate_semiring(x, y)
    # No contracting dims, no batch dims, result[i, j] = x[i] * y[j].
    return x.dot_general(y, (((), ()), ((), ())))


def matrix_power(x: AlgebraicArray, n: int) -> AlgebraicArray:
    """Raise a square matrix to the non-negative integer power *n*.

    Uses binary exponentiation (O(log n) multiplications).

    Parameters
    ----------
    x : AlgebraicArray
        Square 2-D array.
    n : int
        Non-negative integer exponent. ``n=0`` is not supported because
        constructing an identity matrix requires backend-specific creation
        functions.

    Raises
    ------
    ValueError
        If *n* is negative.
    NotImplementedError
        If *n* is zero.
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


def _take_diagonal_axes(x: AlgebraicArray, axis1: int, axis2: int) -> AlgebraicArray:
    """Extract the diagonal along two axes, placing the result at the lower axis position."""
    from algebraic.ops._passthrough import diagonal, moveaxis, permute_dims

    if axis1 > axis2:
        axis1, axis2 = axis2, axis1
    perm = tuple(i for i in range(x.ndim) if i not in (axis1, axis2)) + (axis1, axis2)
    y = permute_dims(x, perm)
    y = diagonal(y)
    return moveaxis(y, -1, axis1)


def _collapse_repeated_labels(x: AlgebraicArray, subs: str) -> tuple[AlgebraicArray, str]:
    """Diagonalise repeated indices within a single operand."""
    labels = list(subs)
    for c in list(dict.fromkeys(labels)):
        while labels.count(c) > 1:
            i1 = labels.index(c)
            i2 = labels.index(c, i1 + 1)
            x = _take_diagonal_axes(x, i1, i2)
            labels.pop(i2)
    return x, "".join(labels)


def _reduce_unneeded_axes(x: AlgebraicArray, subs: str, keep: set[str]) -> tuple[AlgebraicArray, str]:
    """Sum-reduce axes whose labels are not in *keep*."""
    axes = tuple(i for i, c in enumerate(subs) if c not in keep)
    if axes:
        x = typing.cast(AlgebraicArray, sum(x, axis=list(axes)))
        subs = "".join(c for i, c in enumerate(subs) if i not in axes)
    return x, subs


def _execute_unary_einsum_step(x: AlgebraicArray, in_sub: str, out_sub: str) -> AlgebraicArray:
    """Handle a single-operand einsum step (diagonal, trace, reduction, transpose)."""
    from algebraic.ops._passthrough import permute_dims

    x, in_sub = _collapse_repeated_labels(x, in_sub)
    x, in_sub = _reduce_unneeded_axes(x, in_sub, set(out_sub))
    if in_sub != out_sub:
        perm = tuple(in_sub.index(c) for c in out_sub)
        x = permute_dims(x, perm)
    return x


def _execute_binary_einsum_step(
    lhs: AlgebraicArray,
    rhs: AlgebraicArray,
    lhs_sub: str,
    rhs_sub: str,
    out_sub: str,
) -> AlgebraicArray:
    """Handle a two-operand einsum step via ``dot_general``."""
    from algebraic.ops._passthrough import permute_dims

    lhs, lhs_sub = _collapse_repeated_labels(lhs, lhs_sub)
    rhs, rhs_sub = _collapse_repeated_labels(rhs, rhs_sub)

    out_set = set(out_sub)
    # lhs_labels = set(lhs_sub)
    rhs_labels = set(rhs_sub)

    # Pre-reduce labels that appear in only one operand and are absent from output.
    lhs, lhs_sub = _reduce_unneeded_axes(lhs, lhs_sub, rhs_labels | out_set)
    rhs, rhs_sub = _reduce_unneeded_axes(rhs, rhs_sub, set(lhs_sub) | out_set)

    lhs_set = set(lhs_sub)

    batch_labels = [c for c in lhs_sub if c in rhs_sub and c in out_set]
    contract_labels = [c for c in lhs_sub if c in rhs_sub and c not in out_set]
    lhs_free_labels = [c for c in lhs_sub if c not in rhs_sub]
    rhs_free_labels = [c for c in rhs_sub if c not in lhs_set]

    lhs_batch = tuple(lhs_sub.index(c) for c in batch_labels)
    rhs_batch = tuple(rhs_sub.index(c) for c in batch_labels)
    lhs_contract = tuple(lhs_sub.index(c) for c in contract_labels)
    rhs_contract = tuple(rhs_sub.index(c) for c in contract_labels)

    result = lhs.dot_general(
        rhs,
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)),
    )

    # dot_general output order: batch + lhs_free + rhs_free
    result_sub = "".join(batch_labels + lhs_free_labels + rhs_free_labels)
    if result_sub != out_sub:
        perm = tuple(result_sub.index(c) for c in out_sub)
        result = permute_dims(result, perm)
    return result


def _execute_einsum_step(step_eq: str, step_ops: list[AlgebraicArray]) -> AlgebraicArray:
    """Dispatch a single contraction step produced by ``opt_einsum``."""
    lhs_rhs, out_sub = step_eq.split("->")
    input_subs = lhs_rhs.split(",")

    if len(step_ops) == 1:
        return _execute_unary_einsum_step(step_ops[0], input_subs[0], out_sub)
    if len(step_ops) == 2:
        return _execute_binary_einsum_step(step_ops[0], step_ops[1], input_subs[0], input_subs[1], out_sub)
    raise NotImplementedError(f"Expected a unary or pairwise contraction step, got {len(step_ops)} operands.")


def einsum(
    subscripts: str,
    *operands: AlgebraicArray,
    optimize: OptimizeKind = "auto",
    memory_limit: _MemoryLimit = None,
) -> AlgebraicArray:
    """Evaluate an Einstein summation using semiring operations.

    Uses ``opt_einsum`` to find an efficient contraction path, then executes
    each pairwise step with ``dot_general`` (semiring-aware contraction).

    Parameters
    ----------
    subscripts : str
        Einsum subscript string (e.g. ``"ij,jk->ik"``).
    *operands : AlgebraicArray
        Input arrays. All must share the same semiring.
    optimize : str or list, default "auto"
        Contraction path optimisation strategy passed to
        ``opt_einsum.contract_path``. ``"auto"`` lets ``opt_einsum`` choose.
    memory_limit : int or None, optional
        Memory limit for the optimiser (bytes). ``None`` means unlimited.

    Returns
    -------
    AlgebraicArray
        The contracted result.

    Examples
    --------
    >>> import algebraic
    >>> sr = algebraic.semirings.tropical_semiring(minplus=True)
    >>> A = algebraic.array([[0.0, 1.0], [2.0, 3.0]], semiring=sr, backend="numpy")
    >>> B = algebraic.array([[4.0, 5.0], [6.0, 7.0]], semiring=sr, backend="numpy")
    >>> C = algebraic.einsum("ij,jk->ik", A, B)
    >>> C.data  # doctest: +SKIP
    array([[ 6.,  7.],
           [ 6.,  7.]])
    """
    import opt_einsum

    if not operands:
        raise ValueError("einsum requires at least one operand.")
    validate_semiring(*operands)

    _, path_info = opt_einsum.contract_path(
        subscripts,
        *(op.shape for op in operands),
        shapes=True,
        optimize=optimize,
        use_blas=False,
        memory_limit=memory_limit,
    )

    work: list[AlgebraicArray] = list(operands)
    for inds, _idx_rm, step_eq, _remaining, _do_blas in path_info.contraction_list:
        step_ops = [work.pop(i) for i in inds]
        work.append(_execute_einsum_step(step_eq, step_ops))

    assert len(work) == 1
    return work[0]


def diff(
    x: AlgebraicArray,
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: AlgebraicArray | None = None,
    append: AlgebraicArray | None = None,
) -> AlgebraicArray:
    """Discrete differences along *axis* (requires a :class:`~algebraic.spec.Ring`).

    Computes the *n*-th-order forward difference: ``out[i] = x[i+1] - x[i]``.

    Parameters
    ----------
    x : AlgebraicArray
        Input array.
    n : int, default 1
        Order of the difference.
    axis : int, default -1
        Axis along which differences are computed.
    prepend : AlgebraicArray or None, optional
        Values to prepend to *x* along *axis* before computing differences.
    append : AlgebraicArray or None, optional
        Values to append to *x* along *axis* before computing differences.

    Raises
    ------
    NotImplementedError
        If the semiring is not a Ring.
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
