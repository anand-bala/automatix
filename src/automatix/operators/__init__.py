"""Weighted automaton operators.

This module provides operators for weighted automata:

- MatrixOperator: Matrix-based operator for NFAs with weight functions
- PolynomialOperator: Polynomial-based operator for AFAs
- from_afa: Factory method to create PolynomialOperator from AFA
- from_ltl: Convenience method to create PolynomialOperator from LTL formula

"""

from automatix.operators.bdd_operator import BDDOperator as BDDOperator
from automatix.operators.matrix import MatrixOperator as MatrixOperator
from automatix.operators.polynomial import PolynomialOperator as PolynomialOperator
from automatix.operators.symbolic_polynomial import (
    SymbolicPolynomialOperator as SymbolicPolynomialOperator,
)
