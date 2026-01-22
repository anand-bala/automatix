# mypy: disable-error-code="no-any-return,no-untyped-call,type-arg"
from __future__ import annotations

import dataclasses
import math
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.core
import jax.extend.core
import jax.lax as lax
import jax.numpy as jnp
import quax
from jaxtyping import Array, ArrayLike, DTypeLike, Shaped
from typing_extensions import final, override

import algebraic.numpy as alge
from algebraic.spec import MatmulFn, Semiring, VdotFn, has_complement, is_ring

if TYPE_CHECKING:
    from ._index_update import _IndexUpdateHelper

K = typing.TypeVar("K", bound=Semiring)


def _as_array(data: Shaped[ArrayLike | AlgebraicArray[K], "*shape"]) -> Shaped[Array, "*shape"]:
    if isinstance(data, AlgebraicArray):
        data = data.data
    return jnp.asarray(data)


@final
class AlgebraicArray(quax.ArrayValue, typing.Generic[K]):
    """A multidimensional array with elements from a semiring.


    This array overrides multiplication and addition to be defined with respect to the
    corresponding semiring.
    """

    data: Shaped[Array, "..."]
    semiring: K = eqx.field(static=True)

    _vdot: None | VdotFn = eqx.field(static=True)
    _matmul: None | MatmulFn = eqx.field(static=True)

    def __init__(
        self,
        data: Shaped[ArrayLike | AlgebraicArray[K], "..."],
        semiring: K,
        *,
        _vdot: None | VdotFn = None,
        _matmul: None | MatmulFn = None,
    ) -> None:
        data = _as_array(data)
        if not eqx.is_array(data):
            raise TypeError(f"AlgebraicArray has non-Array data: {type(data)=}")
        self.data = data
        self.semiring = semiring
        self._vdot = _vdot
        self._matmul = _matmul

    @override
    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.data.shape, self.data.dtype)

    @override
    def materialise(self) -> Shaped[Array, "..."]:
        return self.data

    @override
    @staticmethod
    def default(
        primitive: jax.extend.core.Primitive, values: Sequence[ArrayLike | quax.Value], params: dict[str, Any]
    ) -> Array | AlgebraicArray[K] | Sequence[Array | AlgebraicArray[K]]:
        """Default handler for unregistered JAX primitives.

        This allows AlgebraicArray to work with JAX operations that don't need
        special semiring handling (like reshaping, indexing, broadcasting, etc.).
        """
        # Extract data from AlgebraicArray arguments
        unwrapped_args: list[ArrayLike | AlgebraicArray[K]] = []
        semiring: K | None = None
        for arg in values:
            if eqx.is_array_like(arg):
                unwrapped_args.append(typing.cast(ArrayLike, arg))
            elif isinstance(arg, AlgebraicArray):
                unwrapped_args.append(arg.data)
                assert isinstance(arg.semiring, Semiring)
                semiring = typing.cast(
                    K,
                    arg.semiring,
                )
            elif isinstance(arg, quax.Value):
                raise ValueError("`AlgebraicArray` cannot be used in conjuction with other Quax types.")
            else:
                raise AssertionError()  # should never happen

        # Call the primitive with unwrapped data
        result = primitive.bind(*unwrapped_args, **params)

        # If we had an AlgebraicArray input, wrap the result(s)
        if semiring is not None:
            if primitive.multiple_results:
                # Primitive returns multiple values - wrap each array result
                # TODO: verify if this is correct.
                return [AlgebraicArray(r, semiring) if eqx.is_array(r) else r for r in result]
            elif eqx.is_array(result):
                # Single result - wrap it
                return AlgebraicArray(result, semiring)
        # return as is
        return result

    def __add__(self, other: AlgebraicArray[K]) -> AlgebraicArray[K]:
        ret = alge.add(self, other)
        assert isinstance(ret, AlgebraicArray)
        return ret

    def sum(self, **kwargs: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        ret = alge.sum(self, **kwargs)
        assert isinstance(ret, AlgebraicArray)
        return ret

    def __mul__(self, other: AlgebraicArray[K]) -> AlgebraicArray[K]:
        ret = alge.multiply(self, other)
        assert isinstance(ret, AlgebraicArray)
        return ret

    def prod(self, **kwargs: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        ret = alge.prod(self, **kwargs)
        assert isinstance(ret, AlgebraicArray)
        return ret

    def __getitem__(self, idx: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        data = self.data[idx]
        return dataclasses.replace(self, data=data)

    @property
    def at(self) -> _IndexUpdateHelper[K]:
        from ._index_update import _IndexUpdateHelper

        return _IndexUpdateHelper(self)

    def __matmul__(self, other: AlgebraicArray[K]) -> AlgebraicArray[K]:
        return alge.matmul(self, other)


@quax.register(lax.add_p)
def _(lhs: AlgebraicArray, rhs: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring addition."""
    if eqx.tree_equal(lhs.semiring, rhs.semiring) is False:
        raise ValueError(
            f"Cannot add AlgebraicArrays with different semirings. lhs semiring: {lhs.semiring}, rhs semiring: {rhs.semiring}"
        )
    result_data = lhs.semiring.add(lhs.data, rhs.data)
    return dataclasses.replace(lhs, data=result_data)


@quax.register(lax.mul_p)
def _(lhs: AlgebraicArray, rhs: AlgebraicArray) -> AlgebraicArray:
    """Element-wise semiring multiplication."""
    if eqx.tree_equal(lhs.semiring, rhs.semiring) is False:
        raise ValueError(
            "Cannot multiply AlgebraicArrays with different semirings. "
            f"lhs semiring: {lhs.semiring}, rhs semiring: {rhs.semiring}"
        )
    result_data = lhs.semiring.mul(lhs.data, rhs.data)
    return dataclasses.replace(lhs, data=result_data)


@quax.register(lax.sub_p)
def _(lhs: AlgebraicArray, rhs: AlgebraicArray) -> AlgebraicArray:
    """Element-wise subtraction (only for Rings with additive inverse)."""
    if eqx.tree_equal(lhs.semiring, rhs.semiring) is False:
        raise ValueError(
            "Cannot subtract AlgebraicArrays with different semirings. "
            f"lhs semiring: {lhs.semiring}, rhs semiring: {rhs.semiring}"
        )

    semiring = lhs.semiring
    # Check if semiring has additive inverse (is a Ring)
    if not is_ring(semiring):
        raise TypeError(
            f"Subtraction requires a Ring with additive_inverse. "
            f"Semiring {type(lhs.semiring).__name__} does not support subtraction."
        )

    # Compute: lhs + (-rhs)
    neg_rhs = semiring.additive_inverse(rhs.data)
    result_data = lhs.semiring.add(lhs.data, neg_rhs)
    return dataclasses.replace(lhs, data=result_data)


@quax.register(lax.neg_p)
def _(a: AlgebraicArray) -> AlgebraicArray:
    """Negation (for Rings) or complement (for Boolean/DeMorgan algebras)."""
    semiring = a.semiring

    # Try additive_inverse first (for Rings)
    if is_ring(semiring):
        result_data = semiring.additive_inverse(a.data)
        return dataclasses.replace(a, data=result_data)

    # Try complement (for Boolean/DeMorgan/Heyting/Stone algebras)
    if has_complement(semiring):
        result_data = semiring.complement(a.data)
        return dataclasses.replace(a, data=result_data)

    raise TypeError(
        f"Negation requires either additive_inverse (Ring) or complement (Boolean algebra). "
        f"Semiring {type(semiring).__name__} has neither."
    )


@quax.register(lax.transpose_p)
def _(a: AlgebraicArray, *, permutation: Sequence[int]) -> AlgebraicArray:
    data = lax.transpose(a.data, permutation=permutation)
    return dataclasses.replace(a, data=data)


@quax.register(lax.reduce_sum_p)
def _(
    a: AlgebraicArray,
    *,
    axes: tuple[int, ...],
    out_sharding: jax.sharding.Sharding | jax.sharding.PartitionSpec | None = None,
) -> AlgebraicArray:
    """Reduce array using semiring addition along specified axes."""
    # Cast zero to match data dtype
    zero_typed = jnp.asarray(a.semiring.zero, dtype=a.data.dtype)
    result_data = jax.lax.reduce(
        a.data,
        zero_typed,
        a.semiring.add,
        axes,
    )
    return dataclasses.replace(a, data=result_data)


@quax.register(lax.reduce_prod_p)
def _(
    a: AlgebraicArray,
    *,
    axes: tuple[int, ...],
    out_sharding: jax.sharding.Sharding | jax.sharding.PartitionSpec | None = None,
) -> AlgebraicArray:
    """Reduce array using semiring multiplication along specified axes."""
    # Cast one to match data dtype
    one_typed = jnp.asarray(a.semiring.one, dtype=a.data.dtype)
    result_data = jax.lax.reduce(
        a.data,
        one_typed,
        a.semiring.mul,
        axes,
    )
    return dataclasses.replace(a, data=result_data)


@quax.register(lax.cumsum_p)
def _(a: AlgebraicArray, *, axis: int, reverse: bool) -> AlgebraicArray:
    """Cumulative sum using semiring addition."""
    # Use associative_scan with semiring.add
    result_data = lax.associative_scan(a.semiring.add, a.data, axis=axis, reverse=reverse)
    return dataclasses.replace(a, data=result_data)


@quax.register(lax.cumprod_p)
def _(a: AlgebraicArray, *, axis: int, reverse: bool) -> AlgebraicArray:
    """Cumulative product using semiring multiplication."""
    # Use associative_scan with semiring.mul
    result_data = lax.associative_scan(a.semiring.mul, a.data, axis=axis, reverse=reverse)
    return dataclasses.replace(a, data=result_data)


@quax.register(lax.scatter_add_p)
def _(
    operand: AlgebraicArray,
    scatter_indices: Array,
    updates: Array | AlgebraicArray,
    *,
    update_jaxpr: Any,  # noqa: ANN401
    update_consts: Sequence[Any],
    dimension_numbers: lax.ScatterDimensionNumbers,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: lax.GatherScatterMode,
) -> AlgebraicArray:
    """Scatter with accumulation using semiring addition.

    This operation combines existing values with updates using semiring.add.
    """
    # Extract updates data if it's an AlgebraicArray
    updates_data = updates.data if isinstance(updates, AlgebraicArray) else updates

    # Gather the existing values at scatter indices
    gather_dnums = lax.GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,  # type: ignore[arg-type]
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,  # type: ignore[arg-type]
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims,  # type: ignore[arg-type]
    )
    slice_sizes = list(operand.data.shape)
    for dim in dimension_numbers.inserted_window_dims:
        slice_sizes[dim] = 1

    existing_values = lax.gather(
        operand.data,
        scatter_indices,
        dimension_numbers=gather_dnums,
        slice_sizes=tuple(slice_sizes),
        mode=mode,
    )

    # Combine existing values with updates using semiring.add
    combined_updates = operand.semiring.add(existing_values, updates_data)

    # Scatter the combined values back (using basic scatter, not scatter_add)
    result_data = lax.scatter(
        operand.data,
        scatter_indices,
        combined_updates,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
    )
    return dataclasses.replace(operand, data=result_data)


@quax.register(lax.scatter_mul_p)
def _(
    operand: AlgebraicArray,
    scatter_indices: Array,
    updates: Array | AlgebraicArray,
    *,
    update_jaxpr: Any,  # noqa: ANN401
    update_consts: Sequence[Any],
    dimension_numbers: lax.ScatterDimensionNumbers,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: lax.GatherScatterMode,
) -> AlgebraicArray:
    """Scatter with accumulation using semiring multiplication.

    This operation combines existing values with updates using semiring.mul.
    """
    # Extract updates data if it's an AlgebraicArray
    updates_data = updates.data if isinstance(updates, AlgebraicArray) else updates

    # Gather the existing values at scatter indices
    gather_dnums = lax.GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,  # type: ignore[arg-type]
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,  # type: ignore[arg-type]
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims,  # type: ignore[arg-type]
    )
    slice_sizes = list(operand.data.shape)
    for dim in dimension_numbers.inserted_window_dims:
        slice_sizes[dim] = 1

    existing_values = lax.gather(
        operand.data,
        scatter_indices,
        dimension_numbers=gather_dnums,
        slice_sizes=tuple(slice_sizes),
        mode=mode,
    )

    # Combine existing values with updates using semiring.mul
    combined_updates = operand.semiring.mul(existing_values, updates_data)

    # Scatter the combined values back (using basic scatter, not scatter_mul)
    result_data = lax.scatter(
        operand.data,
        scatter_indices,
        combined_updates,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
    )
    return dataclasses.replace(operand, data=result_data)


@quax.register(lax.dot_general_p)
def _(
    lhs: AlgebraicArray,
    rhs: AlgebraicArray,
    *,
    dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    precision: lax.Precision | None = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: jax.sharding.Sharding | jax.sharding.PartitionSpec | None = None,
) -> AlgebraicArray:
    """Compute generalized dot product using semiring operations.

    This implements dot_general for AlgebraicArray, using the semiring's
    multiplication and addition operations instead of standard arithmetic.
    """
    # Validate semiring compatibility
    if eqx.tree_equal(lhs.semiring, rhs.semiring) is False:
        raise ValueError(
            "Cannot perform dot_general on AlgebraicArrays with different semirings. "
            f"lhs semiring: {lhs.semiring}, rhs semiring: {rhs.semiring}"
        )

    # Parse dimension_numbers
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    # Check for special case: vector dot product (vdot)
    # vdot applies when: 1D arrays, single contracting dimension, no batch dims
    if (
        lhs._vdot is not None
        and lhs.data.ndim == 1
        and rhs.data.ndim == 1
        and len(lhs_contract) == 1
        and len(rhs_contract) == 1
        and len(lhs_batch) == 0
        and len(rhs_batch) == 0
        and lhs_contract[0] == 0
        and rhs_contract[0] == 0
    ):
        result_data = lhs._vdot(lhs.data, rhs.data)
        return dataclasses.replace(lhs, data=result_data)

    # Check for special case: matrix multiplication (matmul)
    # matmul applies when: 2D arrays, standard matmul pattern, no batch dims
    if (
        lhs._matmul is not None
        and lhs.data.ndim == 2
        and rhs.data.ndim == 2
        and len(lhs_contract) == 1
        and len(rhs_contract) == 1
        and len(lhs_batch) == 0
        and len(rhs_batch) == 0
        and lhs_contract[0] == 1
        and rhs_contract[0] == 0
    ):
        result_data = lhs._matmul(lhs.data, rhs.data)
        return dataclasses.replace(lhs, data=result_data)

    # General case: implement using semiring operations
    lhs_data = lhs.data
    rhs_data = rhs.data
    semiring = lhs.semiring

    # Get all dimension indices
    lhs_ndim = lhs_data.ndim
    rhs_ndim = rhs_data.ndim

    # Determine free dimensions (not contracted or batched)
    lhs_free = tuple(i for i in range(lhs_ndim) if i not in lhs_contract and i not in lhs_batch)
    rhs_free = tuple(i for i in range(rhs_ndim) if i not in rhs_contract and i not in rhs_batch)

    # Build permutation to arrange: [batch, free, contract]
    lhs_perm = lhs_batch + lhs_free + lhs_contract
    rhs_perm = rhs_batch + rhs_free + rhs_contract

    # Transpose arrays
    lhs_transposed = jnp.transpose(lhs_data, lhs_perm)
    rhs_transposed = jnp.transpose(rhs_data, rhs_perm)

    # Calculate dimension sizes
    n_batch = len(lhs_batch)
    n_lhs_free = len(lhs_free)
    n_rhs_free = len(rhs_free)
    _n_contract = len(lhs_contract)

    # Get shapes after transposition
    batch_shape = tuple(lhs_transposed.shape[i] for i in range(n_batch))
    lhs_free_shape = tuple(lhs_transposed.shape[i] for i in range(n_batch, n_batch + n_lhs_free))
    rhs_free_shape = tuple(rhs_transposed.shape[i] for i in range(n_batch, n_batch + n_rhs_free))
    contract_shape = tuple(lhs_transposed.shape[i] for i in range(n_batch + n_lhs_free, lhs_transposed.ndim))

    # Flatten batch dimensions
    batch_size = math.prod(batch_shape)
    lhs_free_size = math.prod(lhs_free_shape)
    rhs_free_size = math.prod(rhs_free_shape)
    contract_size = math.prod(contract_shape)

    # Reshape to [batch_size, lhs_free_size, contract_size] and [batch_size, rhs_free_size, contract_size]
    if n_batch > 0:
        lhs_reshaped = lhs_transposed.reshape(batch_size, lhs_free_size, contract_size)
        rhs_reshaped = rhs_transposed.reshape(batch_size, rhs_free_size, contract_size)

        # Expand dimensions for broadcasting:
        # lhs: [batch, lhs_free, 1, contract]
        # rhs: [batch, 1, rhs_free, contract]
        lhs_expanded = lhs_reshaped[:, :, None, :]
        rhs_expanded = rhs_reshaped[:, None, :, :]

        # Element-wise multiply using semiring: [batch, lhs_free, rhs_free, contract]
        products = semiring.mul(lhs_expanded, rhs_expanded)

        # Sum along contract dimension using semiring addition
        # Cast zero to match products dtype
        zero_typed = jnp.asarray(semiring.zero, dtype=products.dtype)
        result = jax.lax.reduce(
            products,
            zero_typed,
            semiring.add,
            (3,),  # Reduce along contract dimension
        )

        # Reshape to final output shape: [batch_shape..., lhs_free_shape..., rhs_free_shape...]
        output_shape = batch_shape + lhs_free_shape + rhs_free_shape
        if output_shape:
            result = result.reshape(output_shape)
        else:
            result = result.squeeze()
    else:
        # No batch dimensions
        lhs_reshaped = lhs_transposed.reshape(lhs_free_size, contract_size)
        rhs_reshaped = rhs_transposed.reshape(rhs_free_size, contract_size)

        # Expand dimensions for broadcasting:
        # lhs: [lhs_free, 1, contract]
        # rhs: [1, rhs_free, contract]
        lhs_expanded = lhs_reshaped[:, None, :]
        rhs_expanded = rhs_reshaped[None, :, :]

        # Element-wise multiply: [lhs_free, rhs_free, contract]
        products = semiring.mul(lhs_expanded, rhs_expanded)

        # Sum along contract dimension
        # Cast zero to match products dtype
        zero_typed = jnp.asarray(semiring.zero, dtype=products.dtype)
        result = jax.lax.reduce(
            products,
            zero_typed,
            semiring.add,
            (2,),  # Reduce along contract dimension
        )

        # Reshape to final output shape
        output_shape = lhs_free_shape + rhs_free_shape
        if output_shape:
            result = result.reshape(output_shape)
        else:
            result = result.squeeze()

    return dataclasses.replace(lhs, data=result)
