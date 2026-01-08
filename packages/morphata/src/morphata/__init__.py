"""Morphata: Flexible automata representations for regular and omega-regular languages.

This package provides:
- Pure structural automaton interfaces (Automaton, Domain, TransitionRelation)
- Acceptance condition expressions (morphata.acceptance)
- HOA format parser (morphata.hoaparser)
- Example implementations (morphata.examples)
"""

from morphata.spec import (
    AcceptanceCondition,
    AlternatingTransitions,
    Automaton,
    DeterministicTransitions,
    Domain,
    InitialState,
    NonDeterministicTransitions,
    TransitionRelation,
    UniversalTransitions,
)

__all__ = [
    "Domain",
    "InitialState",
    "AcceptanceCondition",
    "TransitionRelation",
    "DeterministicTransitions",
    "NonDeterministicTransitions",
    "UniversalTransitions",
    "AlternatingTransitions",
    "Automaton",
]
