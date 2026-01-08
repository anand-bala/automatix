"""Automatix: A library for weighted automata and automaton operators.

Core exports:
- WeightFunction: Type alias for weight functions mapping (input, guard) to semiring values
- Predicate: Wrapper for predicate functions
- make_atomic_predicate_weight_function: Factory for weight functions from atomic predicates
- NFA: Nondeterministic finite automaton
- AFA: Alternating finite automaton
"""

import automatix.automata as automata
import automatix.weights as weights
from automatix.spec import (
    Guard,
    WeightFunction,
)

__all__ = [
    "Guard",
    "WeightFunction",
    "automata",
    "weights",
]
