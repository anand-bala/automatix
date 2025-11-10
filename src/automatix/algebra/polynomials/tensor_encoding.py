"""Polynomial evaluation algorithms and tensor-based encoding.

This module implements multilinear polynomial evaluation over semirings.
Currently provides Algorithm 1 (monomial basis enumeration), with support
for Algorithm 4 (recursive traversal) planned for future optimization.

Reference: Gillespie, B.R. "A Note on Multilinear Polynomial Evaluation" (2023)
"""

from typing import Dict, Type

import jax
from jaxtyping import Array, Num

from automatix.algebra.polynomials.ring_polynomials import MultilinearPolynomial
from automatix.algebra.spec import AbstractSemiring


def eval_algorithm_1(
    poly: MultilinearPolynomial,
    semiring: Type[AbstractSemiring],
    values: Dict[int, Num[Array, ""]],
) -> Num[Array, ""]:
    r"""Evaluate polynomial using Algorithm 1 (monomial basis enumeration).

    Algorithm 1 iterates over all 2^q monomials and evaluates each one:
        P(x) = sum_{alpha in {0,1}^q} c_alpha * product_{i: alpha_i=1} x_i

    This is the simplest and most direct evaluation method, suitable for:
    - Validation and testing
    - Small polynomials (q <= 15)
    - When code clarity is more important than performance

    Time complexity: O(q * 2^q) multiplications + O(2^q) additions
    Space complexity: O(1) temporary space (plus input sizes)

    Parameters
    ----------
    poly : MultilinearPolynomial
        The polynomial to evaluate.
    semiring : Type[AbstractSemiring]
        The semiring defining addition (oplus) and multiplication (otimes).
    values : Dict[int, Num[Array, ""]]
        Dictionary mapping state indices (0 to q-1) to their evaluation values.
        Each value must be a scalar in the semiring.

    Returns
    -------
    Num[Array, ""]
        The polynomial evaluated at the given point, a scalar in the semiring.

    Raises
    ------
    ValueError
        If values dictionary is missing entries for any state in [0, q-1].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from automatix.algebra.polynomials import MultilinearPolynomial
    >>> from automatix.algebra.backends.jax_ import LatticeAlgebra
    >>> from automatix.algebra.polynomials.tensor_encoding import eval_algorithm_1
    >>>
    >>> # Create polynomial: x_0 + x_1
    >>> poly = MultilinearPolynomial.zeros(num_states=2)
    >>> poly = poly.set_monomial((1, 0), 1.0)  # x_0 coefficient
    >>> poly = poly.set_monomial((0, 1), 1.0)  # x_1 coefficient
    >>>
    >>> # Evaluate at x_0 = 0.5, x_1 = 0.3
    >>> values = {0: jnp.array(0.5), 1: jnp.array(0.3)}
    >>> result = eval_algorithm_1(poly, LatticeAlgebra, values)
    >>> # Result should be max(0.5, 0.3) = 0.5 (for MaxMin semiring)
    >>> result
    Array(0.5, dtype=float32)

    Notes
    -----
    - Correctness is guaranteed for all distributive lattice semirings
    - Not optimized for large polynomials; use Algorithm 4 for q > 15
    - The Python loop over monomials is still JAX-compatible because
      it iterates over a fixed-size range (no dynamic shape dependencies)
    """
    # Validate that all states have values
    for i in range(poly.num_states):
        if i not in values:
            raise ValueError(f"Missing value for state q_{i}. Values must include all states in [0, {poly.num_states - 1}].")

    # Initialize accumulator to zero in the semiring
    # zeros() returns a scalar (shape ()) when called with empty/unit shape
    accumulator: Num[Array, ""] = semiring.zeros(())

    # Iterate over all monomials (2^q of them)
    for idx in range(2**poly.num_states):
        # Convert index to binary tuple (which states appear in this monomial)
        alpha = MultilinearPolynomial.int_to_alpha(idx, poly.num_states)

        # Get coefficient for this monomial
        coeff = poly.coefficients[idx]

        # Compute product of state variables in this monomial
        # product = x_{i1} * x_{i2} * ... where alpha_{ij} = 1
        monomial_product: Num[Array, ""] = semiring.ones(())
        for i, bit in enumerate(alpha):
            if bit == 1:
                monomial_product = semiring.multiply(monomial_product, values[i])

        # Accumulate: accumulator = accumulator + (coeff * monomial_product)
        term: Num[Array, ""] = semiring.multiply(coeff, monomial_product)
        accumulator = semiring.add(accumulator, term)

    return accumulator


def eval_algorithm_1_batch(
    poly: MultilinearPolynomial,
    semiring: Type[AbstractSemiring],
    evaluation_points: Num[Array, "batch_size num_states"],
) -> Num[Array, " batch_size"]:
    r"""Batch evaluate polynomial at multiple points using Algorithm 1.

    Evaluates the polynomial at each point in evaluation_points using vmap
    to vectorize over the batch dimension.

    Parameters
    ----------
    poly : MultilinearPolynomial
        The polynomial to evaluate.
    semiring : Type[AbstractSemiring]
        The semiring defining operations.
    evaluation_points : Num[Array, "batch_size q"]
        Matrix of evaluation points, where each row is (x_0, x_1, ..., x_{q-1}).

    Returns
    -------
    Num[Array, "batch_size"]
        Array of results, one per evaluation point.

    Examples
    --------
    >>> # Evaluate polynomial at 10 different points
    >>> evaluation_points = jnp.ones((10, poly.num_states))
    >>> results = eval_algorithm_1_batch(poly, LatticeAlgebra, evaluation_points)
    >>> results.shape
    (10,)
    """

    def single_eval(point: Num[Array, " {poly.num_states}"]) -> Num[Array, ""]:
        """Evaluate polynomial at a single point."""
        values = {i: point[i] for i in range(poly.num_states)}
        return eval_algorithm_1(poly, semiring, values)

    # Vectorize over the batch dimension (axis 0 of evaluation_points)
    batched_eval = jax.vmap(single_eval, in_axes=0)
    return batched_eval(evaluation_points)


# REVIEW NEEDED: Algorithm 4 implementation (recursive traversal)
# This would provide O(n) time with O(log n) space for large polynomials.
# Pseudocode from AFA_POLYNOMIAL_ARCHITECTURE.md suggests using jax.lax.scan
# for efficient recursive traversal. Deferred to v0.6.0 optimization phase.
#
# def eval_algorithm_4(poly, semiring, values):
#     """Recursive traversal evaluation (O(n) time, O(log n) space)."""
#     ...


__all__ = [
    "eval_algorithm_1",
    "eval_algorithm_1_batch",
]
