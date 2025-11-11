"""Polynomial substitution and like-term collection.

Implements polynomial substitution P(x) with x_i -> Q_i, where Q_i are
successor polynomials. This is the core operation for AFA state transitions.

Reference: Gillespie (2023) - Multilinear polynomial semantics
"""

import logging
from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Num

from automatix.algebra._compat import normalize_semiring
from automatix.algebra.kernels import AlgebraicStructure
from automatix.algebra.polynomials.ring_polynomials import MultilinearPolynomial
from automatix.algebra.spec import AbstractSemiring

logger = logging.getLogger(__name__)


def polynomial_multiply(
    poly1: MultilinearPolynomial,
    poly2: MultilinearPolynomial,
) -> MultilinearPolynomial:
    r"""Multiply two multilinear polynomials.

    Computes the product P1(x) * P2(x) where both are multilinear polynomials
    over the same set of variables using the distribute-all-terms approach.

    Algorithm:
    1. For each pair of monomials (m1, m2) from P1 and P2:
       - Merge state indices: result_alpha[i] = m1_alpha[i] | m2_alpha[i]
       - Multiply coefficients: c1 * c2 (using semiring multiply)
       - Accumulate into result at result_idx (using semiring add)
    2. Multilinear constraint is automatically satisfied: since we use bitwise OR,
       repeated indices map to the same monomial (x_i^2 = x_i)
    3. Like-term collection happens automatically via accumulation

    Example:
    >>> (1 + x_0) * (1 + x_1) = 1 + x_0 + x_1 + x_0*x_1
    Breaking down:
    - 1 * 1 = 1 (index 0)
    - 1 * x_1 = x_1 (index 2)
    - x_0 * 1 = x_0 (index 1)
    - x_0 * x_1 = x_0*x_1 (index 3)

    Parameters
    ----------
    poly1 : MultilinearPolynomial
        First polynomial (input).
    poly2 : MultilinearPolynomial
        Second polynomial (input).

    Returns
    -------
    MultilinearPolynomial
        Product P1 * P2 with coefficients in the same algebra.

    Raises
    ------
    ValueError
        If polynomials have different num_states.
    """
    if poly1.num_states != poly2.num_states:
        raise ValueError(f"Polynomials must have same number of states: {poly1.num_states} != {poly2.num_states}")

    num_states = poly1.num_states
    algebra = normalize_semiring(poly1.algebra)

    # Initialize result with zero polynomial
    result_coeffs = jnp.zeros(2**num_states)

    # Iterate over all pairs of monomials
    for idx1 in range(2**num_states):
        for idx2 in range(2**num_states):
            coeff1 = poly1.coefficients[idx1]
            coeff2 = poly2.coefficients[idx2]

            # Extract binary representation of which states appear in each monomial
            alpha1 = MultilinearPolynomial.int_to_alpha(idx1, num_states)
            alpha2 = MultilinearPolynomial.int_to_alpha(idx2, num_states)

            # Merge state sets: union (bitwise OR) for multilinear multiplication
            # This naturally enforces x_i^2 = x_i (idempotent)
            result_alpha = tuple(a1 | a2 for a1, a2 in zip(alpha1, alpha2))
            result_idx = MultilinearPolynomial.alpha_to_int(result_alpha)

            # Multiply coefficients using semiring kernel
            term_coeff = algebra.mul(
                jnp.array([coeff1]), jnp.array([coeff2])
            )[0]

            # Accumulate: result[result_idx] += term_coeff
            # Uses semiring addition for like-term collection
            current = result_coeffs[result_idx]
            result_coeffs = result_coeffs.at[result_idx].set(
                algebra.add(
                    jnp.array([current]), jnp.array([term_coeff])
                )[0]
            )

    return MultilinearPolynomial(
        algebra=algebra,
        coefficients=result_coeffs,
        num_states=num_states,
        max_degree=None,
    )


def polynomial_substitute(
    poly: MultilinearPolynomial,
    successors: Dict[int, MultilinearPolynomial],
) -> MultilinearPolynomial:
    r"""Substitute state variables with successor polynomials.

    Given P(x_0, ..., x_q) and successor polynomials Q_i,
    compute: P(Q_0, Q_1, ..., Q_q)

    Algorithm:
    1. For each monomial m in P with coefficient c_m:
       - Extract which states appear in this monomial: alpha tuple
       - For each state i where alpha[i] = 1:
         - Multiply together all Q_i (using polynomial_multiply)
       - Scale result by c_m (coefficient scaling)
       - Accumulate into result (like-term collection)

    2. Like-term collection:
       - Multiple monomials may map to same state set
       - Combine via semiring addition (automatic via accumulation)

    3. Missing successors:
       - Default to zero polynomial (rejecting sink semantics)
       - Log warning when successor missing

    Parameters
    ----------
    poly : MultilinearPolynomial
        Polynomial to substitute into.
    successors : Dict[int, MultilinearPolynomial]
        Mapping from state index to successor polynomial.
        Missing states default to zero polynomial with warning.

    Returns
    -------
    MultilinearPolynomial
        Result of substitution with like terms collected.

    Examples
    --------
    >>> # P[x_0 <- Q_0, x_1 <- Q_1]
    >>> poly = MultilinearPolynomial.ones(2)  # 1 + x_0 + x_1 + x_0*x_1
    >>> Q_0 = MultilinearPolynomial.from_monomial(2, (1, 0), 1.0)  # x_0
    >>> Q_1 = MultilinearPolynomial.ones(2)  # 1 (constant)
    >>> result = polynomial_substitute(poly, {0: Q_0, 1: Q_1})
    >>> # Monomials:
    >>> # (1,0) [x_0]: 1 * Q_0 = x_0
    >>> # (0,1) [x_1]: 1 * Q_1 = 1
    >>> # (1,1) [x_0*x_1]: 1 * Q_0 * Q_1 = x_0
    >>> # Result collects like terms
    """
    num_states = poly.num_states
    algebra = poly.algebra

    # Initialize result with zero polynomial
    result = MultilinearPolynomial.zeros(algebra, num_states)

    # Check for missing successors and warn
    for state_idx in range(num_states):
        if state_idx not in successors:
            logger.warning(f"No successor for state q_{state_idx}, using zero polynomial")

    for idx in range(2**num_states):
        coeff = poly.coefficients[idx]
        alpha = MultilinearPolynomial.int_to_alpha(idx, num_states)

        # Compute product of successors for states that appear in this monomial
        # Start with constant 1 polynomial
        term_poly = MultilinearPolynomial.ones(algebra, num_states)

        for state_i, bit in enumerate(alpha):
            if bit == 1:
                # This state appears in the monomial
                if state_i in successors:
                    # Multiply by its successor
                    term_poly = polynomial_multiply(term_poly, successors[state_i])
                else:
                    # Missing successor: multiply by zero (rejecting sink)
                    term_poly = MultilinearPolynomial.zeros(algebra, num_states)

        # Scale by the original coefficient
        scaled_poly = _polynomial_scalar_mult(coeff, term_poly, algebra)

        # Accumulate into result (like-term collection via semiring addition)
        result = _polynomial_add(result, scaled_poly, algebra)

    return result


def _polynomial_add(
    poly1: MultilinearPolynomial,
    poly2: MultilinearPolynomial,
    algebra: AlgebraicStructure,
) -> MultilinearPolynomial:
    """Add two polynomials element-wise using semiring addition.

    Parameters
    ----------
    poly1 : MultilinearPolynomial
        First polynomial.
    poly2 : MultilinearPolynomial
        Second polynomial.
    algebra : AlgebraicStructure
        The algebra for operations.

    Returns
    -------
    MultilinearPolynomial
        Sum of the two polynomials.
    """
    sum_coeffs = algebra.add(
        poly1.coefficients, poly2.coefficients
    )
    return MultilinearPolynomial(
        algebra=algebra,
        coefficients=sum_coeffs,
        num_states=poly1.num_states,
        max_degree=None,
    )


def _polynomial_scalar_mult(
    scalar: Num[Array, ""],
    poly: MultilinearPolynomial,
    algebra: AlgebraicStructure,
) -> MultilinearPolynomial:
    """Multiply all coefficients of a polynomial by a scalar.

    Parameters
    ----------
    scalar : Num[Array, ""]
        The scalar value (in the semiring).
    poly : MultilinearPolynomial
        The polynomial to scale.
    algebra : AlgebraicStructure
        The algebra for operations.

    Returns
    -------
    MultilinearPolynomial
        Scaled polynomial.
    """
    # Multiply scalar by all coefficients
    scaled_coeffs = algebra.mul(
        jnp.full_like(poly.coefficients, scalar),
        poly.coefficients,
    )
    return MultilinearPolynomial(
        algebra=algebra,
        coefficients=scaled_coeffs,
        num_states=poly.num_states,
        max_degree=poly.max_degree,
    )


__all__ = [
    "polynomial_multiply",
    "polynomial_substitute",
]
