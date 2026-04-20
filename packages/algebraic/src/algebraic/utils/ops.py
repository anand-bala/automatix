"""Dispatchable common utility operations"""

from __future__ import annotations

import dataclasses
import math
import typing
from typing import Any

import array_api_compat

from algebraic.types import Array, BinaryOp, Scalar, is_jax_array, is_numpy_array, is_torch_array


def reduce_axes(
    data: Array,
    identity: Scalar,
    semiring_op: BinaryOp,
    *,
    axis: tuple[int, ...],
    keepdims: bool = False,
) -> Array:
    """Reduce *data* along *axis* using a binary semiring operation.

    Parameters
    ----------
    data : Array
        Input array.
    identity : Array
        Identity element for the operation (e.g. ``zero`` or ``one``).
    semiring_op : callable
        Binary operation (e.g. ``semiring.add`` or ``semiring.mul``).
    axis : tuple of int
        Sorted, non-negative axis indices to reduce over.
    keepdims : bool
        When ``True``, reduced axes are replaced by size-1 dimensions.
    Returns
    -------
    Array
        Reduced array.
    """
    xp = array_api_compat.array_namespace(data)

    result: Array
    if is_jax_array(data):
        import jax
        import jax.numpy as jnp

        identity = jnp.asarray(identity, dtype=data.dtype, device=data.device)
        result = jax.lax.reduce(data, identity, semiring_op, dimensions=axis)
    elif is_torch_array(data) or is_numpy_array(data):
        import numpy.typing as npt
        import torch

        acc: torch.Tensor | npt.NDArray[Any]
        device = array_api_compat.device(data)

        identity = xp.asarray(identity, dtype=data.dtype, device=device)

        result = data
        for dim in sorted(axis, reverse=True):
            n = result.shape[dim]
            slices = [_take(result, i, dim) for i in range(n)]
            acc = xp.broadcast_to(identity, slices[0].shape)
            for s in slices:
                acc = typing.cast(torch.Tensor | npt.NDArray[Any], semiring_op(acc, s))
            result = acc
    else:
        raise NotImplementedError(f"Unsupported array type {type(data)}")
    if keepdims:
        for dim in sorted(axis):
            result = xp.expand_dims(result, axis=dim)
    return result


def prefix_scan_axis(
    data: Array,
    identity: Scalar,
    semiring_op: BinaryOp,
    *,
    axis: int,
    include_initial: bool,
) -> Array:
    """Inclusive prefix scan along *axis* using a binary semiring operation.

    Parameters
    ----------
    data : Array
        Input array
    identity : Scalar
        Identity element for the operation.
    semiring_op : callable
        Binary operation (e.g. ``semiring.add`` or ``semiring.mul``).
    axis : int
        Non-negative axis along which to scan.
    include_initial : bool
        When ``True``, prepend an identity slice so that the output has
        one extra element along *axis*.

    Returns
    -------
    Array
        Scanned array.
    """
    xp = array_api_compat.array_namespace(data, identity)
    dtype = data.dtype

    scanned: Array
    if is_torch_array(data) or is_numpy_array(data):
        import numpy as np
        import torch

        n = data.shape[axis]
        slices = [_take(data, i, axis) for i in range(n)]
        outputs: list[Any] = []

        acc: torch.Tensor | np.ndarray
        initial: torch.Tensor | np.ndarray

        device = array_api_compat.device(data)
        initial = xp.asarray(identity, dtype=data.dtype, device=device)

        acc = xp.broadcast_to(initial, slices[0].shape) if n > 0 else initial
        for s in slices:
            acc = typing.cast(torch.Tensor | np.ndarray, semiring_op(acc, s))
            outputs.append(acc)

        scanned = xp.stack(outputs, axis=axis) if outputs else data
    elif is_jax_array(data):
        import jax

        scanned = jax.lax.associative_scan(semiring_op, data, axis=axis)
    else:
        raise NotImplementedError(f"Unknown array type {type(data)}")

    if include_initial:
        shape = list(scanned.shape)
        shape[axis] = 1
        id_slice: Array = xp.full(tuple(shape), identity, dtype=dtype)
        scanned = xp.concat(id_slice, scanned, axis)
    return scanned


def _take(data: Array, index: int, axis: int) -> Array:
    """Extract a single slice along *axis*."""
    if array_api_compat.is_torch_array(data):
        return data.select(axis, index)
    import numpy as np

    return np.take(data, index, axis=axis)  # type: ignore[no-any-return]


@dataclasses.dataclass(frozen=True, slots=True)
class DotPlan:
    """Pre-computed shapes and permutations for a ``dot_general`` call.

    All three backends (JAX, NumPy, Torch) share the same planning logic;
    only the actual kernel (transpose, reshape, multiply, reduce) differs.
    """

    lhs_perm: tuple[int, ...]
    rhs_perm: tuple[int, ...]
    n_batch: int
    n_lhs_free: int
    n_rhs_free: int
    batch_shape: tuple[int, ...]
    lhs_free_shape: tuple[int, ...]
    rhs_free_shape: tuple[int, ...]
    batch_size: int
    lhs_free_size: int
    rhs_free_size: int
    contract_size: int
    output_shape: tuple[int, ...]

    @staticmethod
    def plan(
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        dimension_numbers: tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], tuple[int, ...]],
        ],
    ) -> "DotPlan":
        """Compute the permutation/reshape plan for a ``dot_general`` call.

        Parameters
        ----------
        lhs_shape : tuple[int, ...]
            Shape of the left-hand-side array.
        rhs_shape : tuple[int, ...]
            Shape of the right-hand-side array.
        dimension_numbers : tuple
            Nested tuple of ``((contracting_dims), (batch_dims))`` for each
            operand, following the same layout as ``jax.lax.dot_general``.

        Returns
        -------
        DotPlan
            Pre-computed plan with all shapes, permutations, and sizes.
        """
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

        lhs_ndim = len(lhs_shape)
        rhs_ndim = len(rhs_shape)

        lhs_free = tuple(i for i in range(lhs_ndim) if i not in lhs_contract and i not in lhs_batch)
        rhs_free = tuple(i for i in range(rhs_ndim) if i not in rhs_contract and i not in rhs_batch)

        lhs_perm = lhs_batch + lhs_free + lhs_contract
        rhs_perm = rhs_batch + rhs_free + rhs_contract

        n_batch = len(lhs_batch)
        n_lhs_free = len(lhs_free)
        n_rhs_free = len(rhs_free)

        # We need shapes after transposing to compute reshape sizes.
        # Since transpose just reorders dims, we can compute directly:
        lhs_t_shape = tuple(lhs_shape[i] for i in lhs_perm)
        rhs_t_shape = tuple(rhs_shape[i] for i in rhs_perm)

        batch_shape = lhs_t_shape[:n_batch]
        lhs_free_shape = lhs_t_shape[n_batch : n_batch + n_lhs_free]
        rhs_free_shape = rhs_t_shape[n_batch : n_batch + n_rhs_free]

        batch_size = math.prod(batch_shape)
        lhs_free_size = math.prod(lhs_free_shape)
        rhs_free_size = math.prod(rhs_free_shape)
        contract_size = math.prod(lhs_t_shape[n_batch + n_lhs_free :])

        output_shape = batch_shape + lhs_free_shape + rhs_free_shape

        return DotPlan(
            lhs_perm=lhs_perm,
            rhs_perm=rhs_perm,
            n_batch=n_batch,
            n_lhs_free=n_lhs_free,
            n_rhs_free=n_rhs_free,
            batch_shape=batch_shape,
            lhs_free_shape=lhs_free_shape,
            rhs_free_shape=rhs_free_shape,
            batch_size=batch_size,
            lhs_free_size=lhs_free_size,
            rhs_free_size=rhs_free_size,
            contract_size=contract_size,
            output_shape=output_shape,
        )
