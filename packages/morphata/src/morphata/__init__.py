"""Morphata: Flexible automata representations for regular and omega-regular languages.

This package provides:
- Graph-based automaton implementations (NFA, STREL)
- HOA format parser
- Acceptance condition expressions
- Base interfaces for automata structures
"""

from morphata.automata import NFA, STRELAutomaton
from morphata.spec import AbstractAutomaton, FiniteAcceptance, Guard, SizedAutomaton

__all__ = [
    "NFA",
    "STRELAutomaton",
    "AbstractAutomaton",
    "SizedAutomaton",
    "Guard",
    "FiniteAcceptance",
]
