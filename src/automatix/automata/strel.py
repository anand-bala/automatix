"""STREL automaton re-export from morphata for backward compatibility.

The STREL implementation has been moved to morphata.automata.strel.
This module re-exports it for backward compatibility with existing automatix code.
"""

from morphata.automata.strel import STRELAutomaton

__all__ = ["STRELAutomaton"]
