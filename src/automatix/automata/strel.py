"""STREL automaton re-export from morphata for backward compatibility.

The STREL implementation lives in morphata.examples.strel.
This module re-exports it for backward compatibility with existing automatix code.
"""

from morphata.examples.strel import STRELAutomaton

__all__ = ["STRELAutomaton"]
