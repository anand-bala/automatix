"""Multilinear polynomial representations over semirings.

This package provides core data structures and algorithms for polynomials
where indeterminants correspond to automaton states.

v0.6.0 components:
- ring_polynomials: MultilinearPolynomial data structure
- tensor_encoding: Evaluation algorithms (Algorithm 1, 4)
- substitution: Polynomial substitution with like-term collection (future)

Usage:
    from automatix.algebra.polynomials import MultilinearPolynomial

    # Create a polynomial
    poly = MultilinearPolynomial.from_monomial(
        num_states=3,
        alpha=(1, 1, 0),
        coefficient=1.0
    )

    # Evaluate or manipulate
    coeff = poly.get_monomial((1, 1, 0))
"""

from automatix.algebra.polynomials.ring_polynomials import MultilinearPolynomial

__all__ = ["MultilinearPolynomial"]
