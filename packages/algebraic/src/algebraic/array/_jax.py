"""JAX backend implementation of `AlgebraicArray`."""

from __future__ import annotations

import math
import typing
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self, override

from algebraic.array.base import AlgebraicArray
from algebraic.spec import Semiring
from algebraic.types import Array, MatmulFn, Number, VdotFn
from algebraic.utils import dispatch, normalize_axes
from algebraic.utils.jax import EqxMeta


class JaxAlgebraicArray(eqx.Module, AlgebraicArray, metaclass=EqxMeta):
    """JAX backend implementation of `AlgebraicArray`.

    Uses `equinox.Module` for JAX pytree compatibility and
    `jax.lax` primitives for `reduce`, `scan`, and `dot_general`.
    """

    data: jax.Array
    semiring: Semiring = eqx.field(static=True)
    _vdot: VdotFn | None = eqx.field(static=True, default=None, kw_only=True)
    _matmul: MatmulFn | None = eqx.field(static=True, default=None, kw_only=True)

    @override
    def _wrap(self, data: Array | Number) -> Self:
        data = jnp.asarray(data)
        return typing.cast(Self, eqx.tree_at(lambda t: t.data, self, data))

    @override
    def dot_general(
        self,
        other: Self,
        dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> Self:
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

        # vdot fast path: 1D dot product
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

        # matmul fast path: 2D matrix multiplication
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

        lhs_transposed = jnp.transpose(lhs_data, lhs_perm)
        rhs_transposed = jnp.transpose(rhs_data, rhs_perm)

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

        zero_typed = jnp.asarray(semiring.zero, dtype=lhs_data.dtype)

        if n_batch > 0:
            lhs_reshaped = lhs_transposed.reshape(batch_size, lhs_free_size, contract_size)
            rhs_reshaped = rhs_transposed.reshape(batch_size, rhs_free_size, contract_size)

            lhs_expanded = lhs_reshaped[:, :, None, :]
            rhs_expanded = rhs_reshaped[:, None, :, :]

            products = semiring.mul(lhs_expanded, rhs_expanded)

            result = jax.lax.reduce(products, zero_typed, semiring.add, (3,))

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

            products = semiring.mul(lhs_expanded, rhs_expanded)

            result = jax.lax.reduce(products, zero_typed, semiring.add, (2,))

            output_shape = lhs_free_shape + rhs_free_shape
            if output_shape:
                result = result.reshape(output_shape)
            else:
                result = result.squeeze()

        return self._wrap(result)


@dispatch
def sum(  # noqa: A001  (intentional shadowing of built-in)
    x: JaxAlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> JaxAlgebraicArray:
    """Reduce *x* using the semiring's addition along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    dims = normalize_axes(axis, x.ndim)
    axis = normalize_axes(axis, x.ndim)
    zero = jnp.asarray(x.semiring.zero, dtype=x.dtype)

    result = jax.lax.reduce(x.data, zero, x.semiring.add, dimensions=axis)
    if keepdims:
        for dim in sorted(dims):
            result = jnp.expand_dims(result, axis=dim)
    result = x._wrap(result)
    return result


@dispatch
def prod(
    x: JaxAlgebraicArray,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> JaxAlgebraicArray:
    """Reduce *x* using the semiring's multiplication along *axis*.

    Args:
        x: Input array.
        axis: Axis or axes to reduce. `None` reduces all axes.
        keepdims: When `True`, reduced axes are replaced by size-1 dimensions.
    """
    dims = normalize_axes(axis, x.ndim)
    axis = normalize_axes(axis, x.ndim)
    one = jnp.asarray(x.semiring.one, dtype=x.dtype)

    result = jax.lax.reduce(x.data, one, x.semiring.mul, dimensions=axis)
    if keepdims:
        for dim in sorted(dims):
            result = jnp.expand_dims(result, axis=dim)
    result = x._wrap(result)
    return result


@dispatch
def cumulative_sum(
    x: JaxAlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> JaxAlgebraicArray:
    """Inclusive prefix sum along *axis* using the semiring's addition.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a zero slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    axis = axis % x.ndim
    semiring = x.semiring
    zero = jnp.asarray(semiring.zero)

    # Perform inclusive prefix scan along the specified axis
    scanned = jax.lax.associative_scan(semiring.add, x.data, axis=axis)

    if include_initial:
        # Create zero slice along the axis
        shape = list(scanned.shape)
        shape[axis] = 1
        zero_slice = jnp.full(shape, zero, dtype=x.data.dtype)

        # Prepend along the axis
        scanned = jnp.concatenate([zero_slice, scanned], axis=axis)
    return x._wrap(scanned)


@dispatch
def cumulative_prod(
    x: JaxAlgebraicArray,
    /,
    *,
    axis: int = 0,
    include_initial: bool = False,
) -> JaxAlgebraicArray:
    """Inclusive prefix product along *axis* using the semiring's multiplication.

    Args:
        x: Input array.
        axis: Axis along which to scan (default 0).
        include_initial: When `True`, prepend a one slice before the scan output so that
            `result.shape[axis] == x.shape[axis] + 1`.
    """
    axis = axis % x.ndim
    semiring = x.semiring
    one = jnp.asarray(semiring.one)

    # Perform inclusive prefix scan along the specified axis
    scanned = jax.lax.associative_scan(semiring.mul, x.data, axis=axis)

    if include_initial:
        # Create one slice along the axis
        shape = list(scanned.shape)
        shape[axis] = 1
        one_slice = jnp.full(shape, one, dtype=x.data.dtype)

        # Prepend along the axis
        scanned = jnp.concatenate([one_slice, scanned], axis=axis)
    return x._wrap(scanned)
