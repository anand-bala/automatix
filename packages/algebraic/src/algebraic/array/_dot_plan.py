"""Shape/permutation planning for ``dot_general`` across backends."""

from __future__ import annotations

import dataclasses
import math


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
    ) -> DotPlan:
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
