"""NumPy backend implementation of `AlgebraicArray`."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from typing_extensions import Self, override

from algebraic._better_abc import frozen
from algebraic.array.base import AlgebraicArray
from algebraic.ops.utils import dispatch, normalize_axes
from algebraic.spec import Semiring
from algebraic.types import Array, MatmulFn, Number, VdotFn


@frozen()
class NumpyAlgebraicArray(AlgebraicArray):
    """NumPy backend implementation of `AlgebraicArray`."""

    data: np.ndarray
    semiring: Semiring
    _vdot: VdotFn | None = None
    _matmul: MatmulFn | None = None

    @override
    def _wrap(self, data: Array | Number) -> Self:
        data = np.asarray(data)
        return dataclasses.replace(self, data=data)

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
        lhs_data = self.data
        rhs_data = other.data
        semiring = self.semiring

        lhs_ndim = lhs_data.ndim
        rhs_ndim = rhs_data.ndim

        lhs_free = tuple(i for i in range(lhs_ndim) if i not in lhs_contract and i not in lhs_batch)
        rhs_free = tuple(i for i in range(rhs_ndim) if i not in rhs_contract and i not in rhs_batch)

        lhs_perm = lhs_batch + lhs_free + lhs_contract
        rhs_perm = rhs_batch + rhs_free + rhs_contract

        lhs_transposed = np.transpose(lhs_data, lhs_perm)
        rhs_transposed = np.transpose(rhs_data, rhs_perm)

        n_batch = len(lhs_batch)
        n_lhs_free = len(lhs_free)
        n_rhs_free = len(rhs_free)

        batch_shape = tuple(lhs_transposed.shape[i] for i in range(n_batch))
        lhs_free_shape = tuple(lhs_transposed.shape[i] for i in range(n_batch, n_batch + n_lhs_free))
        rhs_free_shape = tuple(rhs_transposed.shape[i] for i in range(n_batch, n_batch + n_rhs_free))

        batch_size = math.prod(batch_shape)
        lhs_free_size = math.prod(lhs_free_shape)
        rhs_free_size = math.prod(rhs_free_shape)
        contract_size = math.prod(tuple(lhs_transposed.shape[i] for i in range(n_batch + n_lhs_free, lhs_transposed.ndim)))

        lhs_reshaped: np.ndarray
        rhs_reshaped: np.ndarray
        if n_batch > 0:
            lhs_reshaped = lhs_transposed.reshape(batch_size, lhs_free_size, contract_size)
            rhs_reshaped = rhs_transposed.reshape(batch_size, rhs_free_size, contract_size)

            lhs_expanded = lhs_reshaped[:, :, None, :]
            rhs_expanded = rhs_reshaped[:, None, :, :]

            products = np.asarray(semiring.mul(lhs_expanded, rhs_expanded))

            # Reduce along contract dimension (last dim) using semiring.add
            slices = [products[..., i] for i in range(products.shape[-1])]
            result = np.full_like(slices[0], semiring.zero)
            for s in slices:
                result = np.asarray(semiring.add(result, s))

            output_shape = batch_shape + lhs_free_shape + rhs_free_shape
            if output_shape:
                result = result.reshape(output_shape)
            else:
                result = result.squeeze()
        else:
            lhs_reshaped = lhs_transposed.reshape(lhs_free_size, contract_size)
            rhs_reshaped = rhs_transposed.reshape(rhs_free_size, contract_size)

            lhs_expanded = lhs_reshaped[:, None, :]
            rhs_expanded = rhs_reshaped[None, :, :]

            products = np.asarray(semiring.mul(lhs_expanded, rhs_expanded))

            # Reduce along contract dimension (last dim) using semiring.add
            slices = [products[:, :, i] for i in range(products.shape[-1])]
            result = np.full_like(slices[0], semiring.zero)
            for s in slices:
                result = np.asarray(semiring.add(result, s))

            output_shape = lhs_free_shape + rhs_free_shape
            if output_shape:
                result = result.reshape(output_shape)
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

    result = np.asarray(x.data)
    for dim in sorted(dims, reverse=True):
        slices = [np.take(result, i, axis=dim) for i in range(result.shape[dim])]
        acc = np.broadcast_to(zero, slices[0].shape).copy()
        for s in slices:
            acc = np.asarray(x.semiring.add(acc, s))
        result = acc

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

    result = x.data
    for dim in sorted(dims, reverse=True):
        slices = [np.take(result, i, axis=dim) for i in range(result.shape[dim])]
        acc = np.broadcast_to(one, slices[0].shape).copy()
        for s in slices:
            acc = np.asarray(x.semiring.mul(acc, s))
        result = acc

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
    semiring = x.semiring
    zero = np.asarray(semiring.zero, dtype=x.dtype)
    n = x.data.shape[axis]

    slices = [np.take(x.data, i, axis=axis) for i in range(n)]
    outputs: list[np.ndarray] = []
    acc = np.broadcast_to(zero, slices[0].shape).copy() if n > 0 else zero
    for s in slices:
        acc = np.asarray(semiring.add(acc, s))
        outputs.append(acc)

    scanned = np.stack(outputs, axis=axis) if outputs else x.data

    if include_initial:
        shape = list(scanned.shape)
        shape[axis] = 1
        zero_slice = np.full(shape, zero, dtype=x.dtype)
        scanned = np.concatenate([zero_slice, scanned], axis=axis)
    return x._wrap(scanned)


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
    semiring = x.semiring
    one = np.asarray(semiring.one, dtype=x.dtype)
    n = x.data.shape[axis]

    slices = [np.take(x.data, i, axis=axis) for i in range(n)]
    outputs: list[np.ndarray] = []
    acc = np.broadcast_to(one, slices[0].shape).copy() if n > 0 else one
    for s in slices:
        acc = np.asarray(semiring.mul(acc, s))
        outputs.append(acc)

    scanned = np.stack(outputs, axis=axis) if outputs else x.data

    if include_initial:
        shape = list(scanned.shape)
        shape[axis] = 1
        one_slice = np.full(shape, one, dtype=x.dtype)
        scanned = np.concatenate([one_slice, scanned], axis=axis)
    return x._wrap(scanned)
