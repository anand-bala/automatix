"""JAX backend implementation of `AlgebraicArray`."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self, override

from algebraic._backend_mixins import EqxReplaceMixin
from algebraic.array._dot_plan import DotPlan
from algebraic.array.base import AlgebraicArray
from algebraic.spec import Semiring
from algebraic.types import Array, MatmulFn, Number, VdotFn
from algebraic.utils import dispatch, normalize_axes
from algebraic.utils.jax import EqxMeta


class JaxAlgebraicArray(eqx.Module, EqxReplaceMixin, AlgebraicArray, metaclass=EqxMeta):
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
        return self._replace_attr("data", data)

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
        plan = DotPlan.plan(self.data.shape, other.data.shape, dimension_numbers)
        semiring = self.semiring

        lhs_transposed = jnp.transpose(self.data, plan.lhs_perm)
        rhs_transposed = jnp.transpose(other.data, plan.rhs_perm)

        zero_typed = jnp.asarray(semiring.zero, dtype=self.data.dtype)

        if plan.n_batch > 0:
            lhs_reshaped = lhs_transposed.reshape(plan.batch_size, plan.lhs_free_size, plan.contract_size)
            rhs_reshaped = rhs_transposed.reshape(plan.batch_size, plan.rhs_free_size, plan.contract_size)

            products = semiring.mul(lhs_reshaped[:, :, None, :], rhs_reshaped[:, None, :, :])
            result = jax.lax.reduce(products, zero_typed, semiring.add, (3,))
        else:
            lhs_reshaped = lhs_transposed.reshape(plan.lhs_free_size, plan.contract_size)
            rhs_reshaped = rhs_transposed.reshape(plan.rhs_free_size, plan.contract_size)

            products = semiring.mul(lhs_reshaped[:, None, :], rhs_reshaped[None, :, :])
            result = jax.lax.reduce(products, zero_typed, semiring.add, (2,))

        if plan.output_shape:
            result = result.reshape(plan.output_shape)
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
