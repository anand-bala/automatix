"""NumPy backend implementation of `AlgebraicArray`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from typing_extensions import Self, override

from algebraic._backend_mixins import DataclassReplaceMixin
from algebraic._better_abc import frozen
from algebraic.array._dot_plan import DotPlan
from algebraic.array._reduction_helpers import reduce_axes, scan_axis
from algebraic.array.base import AlgebraicArray
from algebraic.spec import Semiring
from algebraic.types import Array, MatmulFn, Number, VdotFn
from algebraic.utils import dispatch, normalize_axes


@frozen()
class NumpyAlgebraicArray(DataclassReplaceMixin, AlgebraicArray):
    """NumPy backend implementation of `AlgebraicArray`."""

    data: np.ndarray
    semiring: Semiring
    _vdot: VdotFn | None = None
    _matmul: MatmulFn | None = None

    @override
    def _wrap(self, data: Array | Number) -> Self:
        data = np.asarray(data)
        return self._replace_attr("data", data)

    @override
    def dot_general(
        self,
        other: Self,
        dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> Self:
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

        # vdot fast path
        if (
            self._vdot is not None
            and self.data.ndim == 1
            and other.data.ndim == 1
            and len(lhs_contract) == 1
            and len(rhs_contract) == 1
            and len(lhs_batch) == 0
            and len(rhs_batch) == 0
            and lhs_contract[0] == 0
            and rhs_contract[0] == 0
        ):
            return self._wrap(self._vdot(self.data, other.data))

        # matmul fast path
        if (
            self._matmul is not None
            and self.data.ndim == 2
            and other.data.ndim == 2
            and len(lhs_contract) == 1
            and len(rhs_contract) == 1
            and len(lhs_batch) == 0
            and len(rhs_batch) == 0
            and lhs_contract[0] == 1
            and rhs_contract[0] == 0
        ):
            return self._wrap(self._matmul(self.data, other.data))

        # General case
        plan = DotPlan.plan(self.data.shape, other.data.shape, dimension_numbers)
        semiring = self.semiring

        lhs_transposed = np.transpose(self.data, plan.lhs_perm)
        rhs_transposed = np.transpose(other.data, plan.rhs_perm)

        if plan.n_batch > 0:
            lhs_reshaped = lhs_transposed.reshape(plan.batch_size, plan.lhs_free_size, plan.contract_size)
            rhs_reshaped = rhs_transposed.reshape(plan.batch_size, plan.rhs_free_size, plan.contract_size)

            products = np.asarray(semiring.mul(lhs_reshaped[:, :, None, :], rhs_reshaped[:, None, :, :]))
        else:
            lhs_reshaped = lhs_transposed.reshape(plan.lhs_free_size, plan.contract_size)
            rhs_reshaped = rhs_transposed.reshape(plan.rhs_free_size, plan.contract_size)

            products = np.asarray(semiring.mul(lhs_reshaped[:, None, :], rhs_reshaped[None, :, :]))

        # Reduce along contract dimension (last dim) using semiring.add
        slices = [products[..., i] for i in range(products.shape[-1])]
        result = np.full_like(slices[0], semiring.zero)
        for s in slices:
            result = np.asarray(semiring.add(result, s))

        if plan.output_shape:
            result = result.reshape(plan.output_shape)
        else:
            result = result.squeeze()

        return self._wrap(result)


@dispatch
def sum(  # noqa: A001  (intentional shadowing of built-in)
    x: NumpyAlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> NumpyAlgebraicArray:
    """Reduce *x* using the semiring's addition along *axis*."""
    dims = normalize_axes(axis, x.ndim)
    zero = np.asarray(x.semiring.zero, dtype=x.dtype)

    result = reduce_axes(x.data, x.semiring.add, zero, dims)

    if keepdims:
        for dim in sorted(dims):
            result = np.expand_dims(result, axis=dim)
    return x._wrap(result)


@dispatch
def prod(
    x: NumpyAlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> NumpyAlgebraicArray:
    """Reduce *x* using the semiring's multiplication along *axis*."""
    dims = normalize_axes(axis, x.ndim)
    one = np.asarray(x.semiring.one, dtype=x.dtype)

    result = reduce_axes(x.data, x.semiring.mul, one, dims)

    if keepdims:
        for dim in sorted(dims):
            result = np.expand_dims(result, axis=dim)
    return x._wrap(result)


@dispatch
def cumulative_sum(
    x: NumpyAlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> NumpyAlgebraicArray:
    """Inclusive prefix sum along *axis* using the semiring's addition.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a zero slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    axis = axis % x.ndim
    zero = np.asarray(x.semiring.zero, dtype=x.dtype)
    result = scan_axis(x.data, x.semiring.add, zero, axis, include_initial)
    return x._wrap(result)


@dispatch
def cumulative_prod(
    x: NumpyAlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> NumpyAlgebraicArray:
    """Inclusive prefix product along *axis* using the semiring's multiplication.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a one slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    axis = axis % x.ndim
    one = np.asarray(x.semiring.one, dtype=x.dtype)
    result = scan_axis(x.data, x.semiring.mul, one, axis, include_initial)
    return x._wrap(result)
