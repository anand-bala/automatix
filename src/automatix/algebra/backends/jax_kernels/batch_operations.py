"""Batch semiring operations optimized for polynomial coefficient accumulation.

This module provides efficient JAX implementations for accumulating polynomial
coefficients during substitution. All operations are JAX JIT and vmap compatible.

Core principle: Use JAX scatter-add (jnp.add.at) to accumulate multiple
coefficients into a result vector efficiently, then combine with semiring
addition.
"""

from typing import Type, cast

from jaxtyping import Array, Num

from automatix.algebra.spec import AbstractSemiring


def batch_accumulate_coefficients(
    semiring: Type[AbstractSemiring],
    current_coeffs: Num[Array, " {2**q}"],
    indices: Num[Array, " n"],
    values: Num[Array, " n"],
) -> Num[Array, " {2**q}"]:
    r"""Accumulate values into coefficient array using semiring addition.

    This is the core operation for polynomial substitution. When substituting
    state variables with successor polynomials, we accumulate like-terms by
    adding coefficients for the same monomial.

    Algorithm:
    1. For each (index, value) pair, sequentially update result[index] via
       result[index] = semiring.add(result[index], value)
    2. Multiple updates to the same index are accumulated using semiring
       addition, preserving algebraic properties.

    Parameters
    ----------
    semiring : Type[AbstractSemiring]
        The semiring defining the addition operation.
    current_coeffs : Num[Array, " {2**q}"]
        Current polynomial coefficient vector (one per monomial).
        Shape: (2^q,) where q is number of automaton states.
    indices : Num[Array, " n"]
        Monomial indices (integers in [0, 2^q-1]) identifying which
        coefficients to update. Shape: (n,)
    values : Num[Array, " n"]
        Coefficient values to add (via semiring.add).
        Shape: (n,) - must match indices shape.

    Returns
    -------
    Num[Array, " {2**q}"]
        Updated coefficient vector with accumulated values.
        Shape: (2^q,)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from automatix.algebra.backends.jax_ import LatticeAlgebra
    >>> coeffs = jnp.zeros(4)  # q=2, so 2^2=4 monomials
    >>> indices = jnp.array([0, 2, 2])  # Add to monomials 0 and 2 (twice)
    >>> values = jnp.array([1.0, 0.5, 0.3])
    >>> result = batch_accumulate_coefficients(LatticeAlgebra, coeffs, indices, values)
    >>> # For MaxMin semiring: [1.0, 0.0, max(0.5, 0.3), 0.0]
    >>> result
    Array([1. , 0. , 0.5, 0. ], dtype=float32)

    Notes
    -----
    - Time complexity: O(n) where n = len(indices)
    - Space complexity: O(2^q) for coefficient array
    - JIT compatible: uses jax.lax.fori_loop for control flow
    - vmap compatible: can be vectorized over batches of coefficients
    - Correctness: Preserves semiring algebraic properties by applying
      semiring.add() at each update, not relying on scalar addition
    """
    from jax import lax

    def update_step(i: int, acc: Num[Array, " {2**q}"]) -> Num[Array, " {2**q}"]:
        """Update accumulator by adding values[i] to acc[indices[i]] via semiring.add."""
        idx = indices[i]
        current_val = acc[idx]
        new_val = semiring.add(current_val, values[i])
        return acc.at[idx].set(new_val)

    # Use fori_loop for JIT compatibility
    result: Num[Array, " {2**q}"] = cast(
        Num[Array, " {2**q}"],
        lax.fori_loop(0, len(indices), update_step, current_coeffs),
    )

    return result


def batch_accumulate_with_multiplication(
    semiring: Type[AbstractSemiring],
    current_coeffs: Num[Array, " {2**q}"],
    indices: Num[Array, " n"],
    values: Num[Array, " n"],
    multiplier: Num[Array, ""],
) -> Num[Array, " {2**q}"]:
    r"""Accumulate values with pre-multiplication.

    This is used during polynomial substitution when the accumulated value
    must be multiplied by a factor (e.g., when combining terms from multiple
    successor polynomials).

    Computes: current_coeffs + (multiplier * values) [in semiring arithmetic]

    Algorithm:
    1. Multiply all values by multiplier using semiring.multiply()
    2. Call batch_accumulate_coefficients with scaled values

    Parameters
    ----------
    semiring : Type[AbstractSemiring]
        The semiring defining addition and multiplication.
    current_coeffs : Num[Array, " {2**q}"]
        Current polynomial coefficient vector.
        Shape: (2^q,)
    indices : Num[Array, " n"]
        Monomial indices for updating.
        Shape: (n,)
    values : Num[Array, " n"]
        Coefficient values.
        Shape: (n,)
    multiplier : Num[Array, ""]
        Factor to multiply all values before accumulation (semiring multiply).
        Must be a scalar.

    Returns
    -------
    Num[Array, " {2**q}"]
        Updated coefficients with semiring.multiply(multiplier, values) accumulated.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from automatix.algebra.backends.jax_ import LatticeAlgebra
    >>> coeffs = jnp.array([0.0, 1.0, 0.5, 0.0])
    >>> indices = jnp.array([1, 2])
    >>> values = jnp.array([2.0, 3.0])
    >>> multiplier = jnp.array(0.5)
    >>> result = batch_accumulate_with_multiplication(
    ...     LatticeAlgebra, coeffs, indices, values, multiplier
    ... )
    >>> # For MaxMin: multiply then max: [0.0, max(1.0, 0.5*2), max(0.5, 0.5*3), 0.0]
    """
    # Multiply all values by the multiplier in the semiring
    scaled_values = semiring.multiply(multiplier, values)

    # Accumulate scaled values
    return batch_accumulate_coefficients(semiring, current_coeffs, indices, scaled_values)


def batch_evaluate_monomials(
    semiring: Type[AbstractSemiring],
    coeffs: Num[Array, " {2**q}"],
    indices: Num[Array, " n"],
) -> Num[Array, " n"]:
    r"""Extract coefficients at specified indices.

    This is a simple gather operation, kept for symmetry with scatter
    (batch_accumulate_coefficients). Useful for extracting multiple monomial
    coefficients from a polynomial in a single vectorized call.

    Algorithm:
    1. Index into coeffs array using indices
    2. Return extracted values

    Parameters
    ----------
    semiring : Type[AbstractSemiring]
        The semiring (used for identity/type consistency).
    coeffs : Num[Array, " {2**q}"]
        Polynomial coefficient vector.
        Shape: (2^q,)
    indices : Num[Array, " n"]
        Monomial indices to retrieve.
        Shape: (n,) - must all be in [0, 2^q)

    Returns
    -------
    Num[Array, " n"]
        Coefficient values at specified indices.
        Shape: (n,)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from automatix.algebra.backends.jax_ import LatticeAlgebra
    >>> coeffs = jnp.array([1.0, 0.0, 0.5, 0.8])
    >>> indices = jnp.array([0, 2, 3])
    >>> result = batch_evaluate_monomials(LatticeAlgebra, coeffs, indices)
    >>> result
    Array([1.0, 0.5, 0.8], dtype=float32)

    Notes
    -----
    - Time complexity: O(n) where n = len(indices)
    - JIT compatible: simple array indexing
    - vmap compatible: can be vectorized over batches of indices
    """
    return coeffs[indices]


# REVIEW NEEDED: Consider whether to add a batch_multiply_coefficients function
# that efficiently multiplies all coefficients by a scalar. This might be useful
# for certain polynomial operations (e.g., scaling during substitution).

__all__ = [
    "batch_accumulate_coefficients",
    "batch_accumulate_with_multiplication",
    "batch_evaluate_monomials",
]
