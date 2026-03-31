"""Automatix-specific interfaces extending morphata base.

Provides automatix-specific extensions to the base automata interfaces from
morphata, adding weighted semantics via semiring-valued weight functions.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from algebraic.types import Array, Scalar
from morphata.spec import BoolExpr as Guard

# Reexport morphata base interfaces for backward compatibility
__all__ = [
    "Guard",
    "WeightFunction",
]


@runtime_checkable
class WeightFunction(Protocol):
    """Weight function mapping an input symbol and guard to a semiring value.

    Implements the transition-weight map :math:`\\lambda(x, \\Delta)` from
    weighted automata theory. Concretely, a :class:`WeightFunction` is any
    callable with the right signature -- plain functions, callable objects,
    and :class:`equinox.Module` instances all satisfy the protocol.

    Examples
    --------
    A constant weight function that always returns ``1.0``:

    >>> def constant(x, guard):
    ...     return 1.0

    A weight function that evaluates the guard against the input:

    >>> def predicate_weight(x, guard):
    ...     return evaluate_guard(x, guard)
    """

    def __call__(self, x: object, guard: Guard[Any]) -> Array | Scalar:
        """Evaluate the weight for an input symbol and transition guard.

        Parameters
        ----------
        x : object
            The input symbol observed at the current step.
        guard : Guard[Any]
            The boolean guard expression labelling the transition.

        Returns
        -------
        Array or Scalar
            The semiring weight assigned to this (input, guard) pair.
        """
        ...
