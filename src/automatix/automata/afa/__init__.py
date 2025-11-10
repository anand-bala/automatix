"""Alternating Finite Automata (AFA) with multilinear polynomial semantics.

An AFA evaluates multilinear polynomials where indeterminants correspond to
automaton states and coefficients come from weight functions.

This module provides:
- AFA: Core AFA class (generic over alphabet, states, semiring)
- PolynomialAutomatonOperator: Executor for AFA on input sequences
- make_polynomial_automaton_operator: Factory function

v0.6.0 Status: Basic structure in place, full semantics deferred to optimization phase.

Reference:
    Gillespie, B.R. (2023). Multilinear Polynomial Evaluation over Semirings.
    AFA_POLYNOMIAL_ARCHITECTURE.md: Core polynomial semantics
    WEEK2_IMPLEMENTATION_PLAN.md: v0.6.0 implementation roadmap
"""

# REVIEW NEEDED: Implement actual imports once modules are ready
# from automatix.automata.afa.automaton import AFA
# from automatix.automata.afa.operators import (
#     PolynomialAutomatonOperator,
#     make_polynomial_automaton_operator,
# )

# __all__ = [
#     "AFA",
#     "PolynomialAutomatonOperator",
#     "make_polynomial_automaton_operator",
# ]
