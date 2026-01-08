"""Automatix-specific interfaces extending morphata base.

This module provides automatix-specific extensions to the base automata interfaces
from morphata. It adds weighted semantics, semiring operations, and state-set-based
acceptance conditions for runtime checking.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array, ScalarLike
from morphata.spec import BoolExpr as Guard

# Re-export morphata base interfaces for backward compatibility
__all__ = [
    "Guard",
    "WeightFunction",
]


@runtime_checkable
class WeightFunction[In, AP](Protocol):
    """Weight function mapping (input, guard) to semiring value.

    A weight function implements lambda(x, Delta) from weighted automata theory:
    - Takes an input symbol x and guard expression Delta
    - Returns a weight in the target semiring
    - Used to compute transition weights in automaton operators

    Examples
    --------
    Simple constant weight function:

    >>> def constant(x, guard):
    ...     return 1.0

    Distance-based weight function:

    >>> def distance_weight(x, guard):
    ...     # Distance from x to satisfying guard
    ...     return compute_distance(x, guard)

    Predicate-based weight function:

    >>> def predicate_weight(x, guard):
    ...     # Evaluate guard with input x
    ...     return evaluate_guard(x, guard)
    """

    def __call__(self, x: In, guard: Guard[AP]) -> Array | ScalarLike: ...
