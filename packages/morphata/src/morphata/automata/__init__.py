"""Graph-based automaton implementations.

This module provides concrete automaton implementations built on NetworkX graphs.
These are pure structural representations without weighted semantics.
"""

from morphata.automata.nfa import NFA
from morphata.automata.strel import STRELAutomaton

__all__ = ["NFA", "STRELAutomaton"]
