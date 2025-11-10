"""Polynomial substitution and like-term collection.

Implements polynomial substitution P(x) with x_i -> Q_i, where Q_i are
successor polynomials. This is the core operation for AFA state transitions.

Reference: Gillespie (2023) - Multilinear polynomial semantics
"""

from typing import Dict, Type

import jax.numpy as jnp

from automatix.algebra.polynomials.ring_polynomials import MultilinearPolynomial
from automatix.algebra.spec import AbstractSemiring


def polynomial_multiply(
    poly1: MultilinearPolynomial,
    poly2: MultilinearPolynomial,
) -> MultilinearPolynomial:
    r"""Multiply two multilinear polynomials.

    Computes the product P1(x) * P2(x) where both are multilinear polynomials
    over the same set of variables.

    Algorithm:
    1. For each pair of monomials (m1, m2) from P1 and P2:
       - Multiply their union of states (AND the bitmasks)
       - Multiply their coefficients (using semiring multiply)
    2. Collect like terms (monomials with same state set)
    3. Combine coefficients using semiring add

    REVIEW NEEDED: Is this the correct semantic for polynomial multiplication?
    - Option A: Distribute all terms: (a*x + b) * (c*x + d) = ac*x^2 + ...
    - Option B: Kronecker product of coefficient vectors
    - Option C: Something else specific to multilinear polynomials?

    Parameters
    ----------
    poly1 : MultilinearPolynomial
        First polynomial (input).
    poly2 : MultilinearPolynomial
        Second polynomial (input).
    semiring : Type[AbstractSemiring]
        The semiring for operations.

    Returns
    -------
    MultilinearPolynomial
        Product P1 * P2.

    Raises
    ------
    ValueError
        If polynomials have different num_states.

    Examples
    --------
    >>> # PLACEHOLDER: Add concrete example once semantics clarified
    >>> poly1 = MultilinearPolynomial.from_monomial(2, (1, 0), 1.0)  # x_0
    >>> poly2 = MultilinearPolynomial.from_monomial(2, (0, 1), 1.0)  # x_1
    >>> product = polynomial_multiply(poly1, poly2, LatticeAlgebra)
    >>> # product should represent x_0 * x_1
    """
    if poly1.num_states != poly2.num_states:
        raise ValueError(f"Polynomials must have same number of states: {poly1.num_states} != {poly2.num_states}")

    # REVIEW NEEDED: Implementation approach
    # For now, provide skeleton with TODOs

    num_states = poly1.num_states
    result_coeffs = jnp.zeros(2**num_states)

    # Placeholder: iterate over all pairs of monomials
    for idx1 in range(2**num_states):
        for idx2 in range(2**num_states):
            coeff1 = poly1.coefficients[idx1]
            coeff2 = poly2.coefficients[idx2]

            # REVIEW NEEDED: How to combine monomials?
            # Currently a no-op - needs implementation
            # alpha1 = MultilinearPolynomial.int_to_alpha(idx1, num_states)
            # alpha2 = MultilinearPolynomial.int_to_alpha(idx2, num_states)
            # result_alpha = (alpha1[i] | alpha2[i] for i in range(num_states))
            # result_idx = MultilinearPolynomial.alpha_to_int(result_alpha)
            # result_coeffs[result_idx] += coeff1 * coeff2 (semiring ops)

    return MultilinearPolynomial(
        coefficients=result_coeffs,
        num_states=num_states,
        max_degree=None,
    )


def polynomial_substitute(
    poly: MultilinearPolynomial,
    semiring: Type[AbstractSemiring],
    successors: Dict[int, MultilinearPolynomial],
) -> MultilinearPolynomial:
    r"""Substitute state variables with successor polynomials.

    Given P(x_0, ..., x_q) and successor polynomials Q_i,
    compute: P(Q_0, Q_1, ..., Q_q)

    Algorithm:
    1. For each monomial m in P with coefficient c_m:
       - If m = x_{i1} * x_{i2} * ... (with specific states)
       - Compute product: Q_{i1} * Q_{i2} * ... (polynomial multiply)
       - Multiply result by coefficient c_m
       - Accumulate into result (with like-term collection)

    2. Like-term collection:
       - Multiple monomials may map to same state set
       - Combine via semiring.add() using batch_accumulate_coefficients

    Parameters
    ----------
    poly : MultilinearPolynomial
        Polynomial to substitute into.
    semiring : Type[AbstractSemiring]
        The semiring for operations.
    successors : Dict[int, MultilinearPolynomial]
        Mapping from state index to successor polynomial.
        REVIEW NEEDED: What if not all states have successors?

    Returns
    -------
    MultilinearPolynomial
        Result of substitution with like terms collected.

    Raises
    ------
    ValueError
        If successors missing for any state in P.

    Examples
    --------
    >>> # PLACEHOLDER: Add example after implementation
    >>> poly = MultilinearPolynomial.from_monomial(2, (1, 0), 1.0)  # x_0
    >>> succ_0 = MultilinearPolynomial.from_monomial(2, (0, 1), 1.0)  # Replace x_0 with x_1
    >>> succ_1 = MultilinearPolynomial.ones(2)  # Replace x_1 with 1
    >>> result = polynomial_substitute(poly, LatticeAlgebra, {0: succ_0, 1: succ_1})
    >>> # result should be the successor polynomial for x_0
    """
    # Validate successors
    for state_idx in range(poly.num_states):
        if state_idx not in successors:
            raise ValueError(f"Missing successor for state q_{state_idx}. All states must have successors.")

    # REVIEW NEEDED: Implementation approach
    # Currently provides skeleton

    result = MultilinearPolynomial.zeros(num_states=poly.num_states)

    for idx in range(2**poly.num_states):
        coeff = poly.coefficients[idx]
        alpha = MultilinearPolynomial.int_to_alpha(idx, poly.num_states)

        # REVIEW NEEDED: How to compute product of successors?
        # For states where alpha_i = 1, multiply Q_i together
        # This requires polynomial_multiply to work correctly

        # term_poly = compute_monomial_product(alpha, successors)
        # scale_by_coeff = polynomial scale by coeff
        # accumulate into result with like-term collection

        pass

    return result


__all__ = [
    "polynomial_multiply",
    "polynomial_substitute",
]
