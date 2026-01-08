"""Automatix-specific interfaces extending morphata base.

This module provides automatix-specific extensions to the base automata interfaces
from morphata. It adds weighted semantics, semiring operations, and state-set-based
acceptance conditions for runtime checking.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Protocol, runtime_checkable

from jaxtyping import Array, ScalarLike

# Import and re-export base interfaces from morphata
from morphata.spec import (
    AbstractAutomaton as _BaseAutomaton,
    Guard,
    SizedAutomaton as _BaseSizedAutomaton,
)

import automatix.acc as acc

# Re-export morphata base interfaces for backward compatibility
# These are the pure structural interfaces without weighted semantics
__all__ = [
    "Guard",
    "AbstractAutomaton",
    "SizedAutomaton",
    "WeightFunction",
    "AcceptanceCondition",
]

# Automatix-specific: state-set based acceptance conditions for runtime checking
# This is different from morphata.acceptance which is expression-based for HOA specs
type AcceptanceCondition[Q: Hashable] = (
    acc.Finite[Q]
    | acc.Buchi[Q]
    | acc.CoBuchi[Q]
    | acc.GeneralizedBuchi[Q]
    | acc.GeneralizedCoBuchi[Q]
    | acc.Rabin[Q]
    | acc.Streett[Q]
    | acc.Muller[Q]
)

# Re-export morphata interfaces
# Note: These don't use automatix.AcceptanceCondition, they use morphata.spec.FiniteAcceptance
AbstractAutomaton = _BaseAutomaton
SizedAutomaton = _BaseSizedAutomaton


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
