"""Torch backend implementation of `AlgebraicArray`."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from typing_extensions import Self, override

from algebraic.array.base import AlgebraicArray
from algebraic.spec import Semiring
from algebraic.types import Array, MatmulFn, Number, VdotFn
from algebraic.utils import dispatch, normalize_axes


class TorchAlgebraicArray(nn.Module, AlgebraicArray):
    """Torch backend implementation of `AlgebraicArray`.

    Uses `torch.nn.Module` so that `data` is registered as a buffer,
    enabling `state_dict()` and `.to(device)` support. The semiring is
    stored as a plain attribute (not a parameter or buffer).
    """

    data: torch.Tensor
    semiring: Semiring

    _vdot: VdotFn | None = None
    _matmul: MatmulFn | None = None

    def __init__(
        self,
        data: torch.Tensor,
        semiring: Semiring,
        *,
        _vdot: VdotFn | None = None,
        _matmul: MatmulFn | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.data: torch.Tensor = data
        self.semiring: Semiring = semiring
        self._vdot = _vdot
        self._matmul = _matmul

    @override
    def _wrap(self, data: Array | Number) -> Self:
        data = torch.asarray(data)
        return type(self)(data, self.semiring, _vdot=self._vdot, _matmul=self._matmul)

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
        lhs_data = torch.asarray(self.data)
        rhs_data = torch.asarray(other.data)
        semiring = self.semiring

        lhs_ndim = lhs_data.ndim
        rhs_ndim = rhs_data.ndim

        lhs_free = tuple(i for i in range(lhs_ndim) if i not in lhs_contract and i not in lhs_batch)
        rhs_free = tuple(i for i in range(rhs_ndim) if i not in rhs_contract and i not in rhs_batch)

        lhs_perm = lhs_batch + lhs_free + lhs_contract
        rhs_perm = rhs_batch + rhs_free + rhs_contract

        lhs_transposed = torch.permute(lhs_data, lhs_perm)
        rhs_transposed = torch.permute(rhs_data, rhs_perm)

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

        if n_batch > 0:
            lhs_reshaped = lhs_transposed.reshape(batch_size, lhs_free_size, contract_size)
            rhs_reshaped = rhs_transposed.reshape(batch_size, rhs_free_size, contract_size)

            lhs_expanded = lhs_reshaped[:, :, None, :]
            rhs_expanded = rhs_reshaped[:, None, :, :]

            products = torch.asarray(semiring.mul(lhs_expanded, rhs_expanded))

            # Reduce along contract dimension (last dim, index 3) using semiring.add
            slices = products.unbind(-1)
            result = torch.full_like(slices[0], semiring.zero)
            for s in slices:
                result = torch.asarray(semiring.add(result, s))

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

            products = torch.asarray(semiring.mul(lhs_expanded, rhs_expanded))

            # Reduce along contract dimension (last dim, index 2) using semiring.add
            slices = products.unbind(-1)
            result = torch.full_like(slices[0], semiring.zero)
            for s in slices:
                result = torch.asarray(semiring.add(result, s))

            output_shape = lhs_free_shape + rhs_free_shape
            if output_shape:
                result = result.reshape(output_shape)
            else:
                result = result.squeeze()

        return self._wrap(result)


@dispatch
def sum(  # noqa: A001  (intentional shadowing of built-in)
    x: TorchAlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> TorchAlgebraicArray:
    """Reduce *x* using the semiring's addition along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    dims = normalize_axes(axis, x.ndim)
    zero = torch.as_tensor(x.semiring.zero, dtype=x.dtype)

    result = torch.asarray(x.data)
    for dim in sorted(dims, reverse=True):
        slices = result.unbind(dim)
        acc = zero.expand_as(slices[0]).clone()
        for s in slices:
            acc = torch.asarray(x.semiring.add(acc, s))
        result = acc

    if keepdims:
        for dim in sorted(dims):
            result = result.unsqueeze(dim)
    return x._wrap(result)


@dispatch
def prod(
    x: TorchAlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> TorchAlgebraicArray:
    """Reduce *x* using the semiring's multiplication along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    dims = normalize_axes(axis, x.ndim)
    one = torch.as_tensor(x.semiring.one, dtype=x.dtype)

    result = torch.asarray(x.data)
    for dim in sorted(dims, reverse=True):
        slices = result.unbind(dim)
        acc = one.expand_as(slices[0]).clone()
        for s in slices:
            acc = torch.asarray(x.semiring.mul(acc, s))
        result = acc

    if keepdims:
        for dim in sorted(dims):
            result = result.unsqueeze(dim)
    return x._wrap(result)


@dispatch
def cumulative_sum(
    x: TorchAlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> TorchAlgebraicArray:
    """Inclusive prefix sum along *axis* using the semiring's addition.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a zero slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    axis = axis % x.ndim
    semiring = x.semiring
    zero = torch.as_tensor(semiring.zero, dtype=x.dtype)
    data = torch.asarray(x.data)
    n = data.shape[axis]

    slices = data.unbind(axis)
    outputs: list[torch.Tensor] = []
    acc = zero.expand_as(slices[0]).clone() if n > 0 else zero
    for s in slices:
        acc = torch.asarray(semiring.add(acc, s))
        outputs.append(acc)

    scanned = torch.stack(outputs, dim=axis) if outputs else data

    if include_initial:
        shape = list(scanned.shape)
        shape[axis] = 1
        zero_slice = torch.full(shape, semiring.zero, dtype=x.dtype)
        scanned = torch.cat([zero_slice, scanned], dim=axis)
    return x._wrap(scanned)


@dispatch
def cumulative_prod(
    x: TorchAlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> TorchAlgebraicArray:
    """Inclusive prefix product along *axis* using the semiring's multiplication.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a one slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    axis = axis % x.ndim
    semiring = x.semiring
    one = torch.as_tensor(semiring.one, dtype=x.dtype)
    data = torch.asarray(x.data)
    n = data.shape[axis]

    slices = data.unbind(axis)
    outputs: list[torch.Tensor] = []
    acc = one.expand_as(slices[0]).clone() if n > 0 else one
    for s in slices:
        acc = torch.asarray(semiring.mul(acc, s))
        outputs.append(acc)

    scanned = torch.stack(outputs, dim=axis) if outputs else data

    if include_initial:
        shape = list(scanned.shape)
        shape[axis] = 1
        one_slice = torch.full(shape, semiring.one, dtype=x.dtype)
        scanned = torch.cat([one_slice, scanned], dim=axis)
    return x._wrap(scanned)
