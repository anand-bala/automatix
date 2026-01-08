"""NFA re-export from morphata for backward compatibility.

The NFA implementation has been moved to morphata.automata.nfa.
This module re-exports it for backward compatibility with existing automatix code.
"""

from morphata.automata.nfa import NFA, NFAState

__all__ = ["NFA", "NFAState"]
