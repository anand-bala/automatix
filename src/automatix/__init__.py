"""Automatix: A library for weighted automata and automaton operators.

Core exports:
- WeightFunction: Type alias for weight functions mapping (input, guard) to semiring values
- Predicate: Wrapper for predicate functions
- make_atomic_predicate_weight_function: Factory for weight functions from atomic predicates
- NFA: Nondeterministic finite automaton
- AFA: Alternating finite automaton
"""

from automatix.predicates import ExprWeightFn, Predicate
from automatix.weights import (
    Guard,
    InputSymbol,
    SemiringValue,
    WeightFunction,
)

__all__ = ["WeightFunction", "InputSymbol", "Guard", "SemiringValue", "Predicate", "ExprWeightFn"]
