"""NFA re-export from morphata for backward compatibility.

The NFA implementation lives in morphata.examples.nfa.
This module re-exports it for backward compatibility with existing automatix code.
"""

from morphata.examples.nfa import NFA, NFAState

__all__ = ["NFA", "NFAState"]
